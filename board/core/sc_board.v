`timescale 1ns / 1ps
// SpecCore on the KV260 (D16 option D): the unchanged RV core + LLM unit, with board memories.
//
// Same behaviour as the sim Wrapper (core/rtl/nexys/Wrapper.v), cycle for cycle:
//  - IROM (64 KB, BRAM) and DMEM (1 MB, URAM, 64-bit rows) are clocked on the FALLING edge of the core
//    clock: the address is stable from the rising edge, the data is back before the next rising edge, so
//    the core sees the same single-cycle "async read" memories as in simulation.
//  - The model image (8 MB window at 0x2000_0000) lives in real DDR (a CMA buffer the A53 fills). Two
//    read paths: the weight stream (ws_addr -> ws_data, sequential prefetch window) and core loads (a
//    512-line x 64 B read cache). When the word the core may use this cycle is not there yet, the core
//    clock is paused (BUFGCE). Paused cycles do not exist for the core: its cycle counter, pipeline and
//    the LLM unit see exactly the simulated sequence. Wall time (clk cycles while running) includes them.
//  - The A53 talks to a 2 MB AXI-Lite window: registers, IROM, console, DMEM (memories only while stopped).
//
// Register map (byte offsets): 0x00 ID 0x5C0E0003 | 0x04 CTRL [0] run (write 1 = start, 0 = abort), [1] BP
//   | 0x08 STATUS [0] running [1] done | 0x0C exit code (SEVENSEG) | 0x10/0x14 DDR base lo/hi
//   | 0x18 console bytes | 0x20/24 core cycles | 0x28/2C wall cycles | 0x30/34 stream-wait cycles
//   | 0x38/3C load-wait cycles | 0x40 load-cache misses | 0x44 stream restarts
//   0x10000 IROM (64 KB) | 0x20000 console (8K words, one char each) | 0x100000 DMEM (1 MB)
module sc_board (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi:m_axi_ws:m_axi_ld, ASSOCIATED_RESET aresetn" *)
    input  wire         clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         aresetn,
    // AXI4-Lite slave (A53)
    input  wire [20:0]  s_axi_awaddr,
    input  wire         s_axi_awvalid,
    output reg          s_axi_awready,
    input  wire [31:0]  s_axi_wdata,
    input  wire [3:0]   s_axi_wstrb,
    input  wire         s_axi_wvalid,
    output reg          s_axi_wready,
    output reg  [1:0]   s_axi_bresp,
    output reg          s_axi_bvalid,
    input  wire         s_axi_bready,
    input  wire [20:0]  s_axi_araddr,
    input  wire         s_axi_arvalid,
    output reg          s_axi_arready,
    output reg  [31:0]  s_axi_rdata,
    output reg  [1:0]   s_axi_rresp,
    output reg          s_axi_rvalid,
    input  wire         s_axi_rready,
    // AXI4 read master: weight stream
    output reg  [39:0]  m_axi_ws_araddr,
    output wire [7:0]   m_axi_ws_arlen,
    output wire [2:0]   m_axi_ws_arsize,
    output wire [1:0]   m_axi_ws_arburst,
    output wire [3:0]   m_axi_ws_arcache,
    output wire [2:0]   m_axi_ws_arprot,
    output reg          m_axi_ws_arvalid,
    input  wire         m_axi_ws_arready,
    input  wire [127:0] m_axi_ws_rdata,
    input  wire [1:0]   m_axi_ws_rresp,
    input  wire         m_axi_ws_rlast,
    input  wire         m_axi_ws_rvalid,
    output wire         m_axi_ws_rready,
    // AXI4 read master: core loads from the model image
    output reg  [39:0]  m_axi_ld_araddr,
    output wire [7:0]   m_axi_ld_arlen,
    output wire [2:0]   m_axi_ld_arsize,
    output wire [1:0]   m_axi_ld_arburst,
    output wire [3:0]   m_axi_ld_arcache,
    output wire [2:0]   m_axi_ld_arprot,
    output reg          m_axi_ld_arvalid,
    input  wire         m_axi_ld_arready,
    input  wire [127:0] m_axi_ld_rdata,
    input  wire [1:0]   m_axi_ld_rresp,
    input  wire         m_axi_ld_rlast,
    input  wire         m_axi_ld_rvalid,
    output wire         m_axi_ld_rready
);
    assign m_axi_ws_arlen = 8'd3;    assign m_axi_ld_arlen = 8'd3;     // 4 beats x 16 B = 64 B
    assign m_axi_ws_arsize = 3'd4;   assign m_axi_ld_arsize = 3'd4;
    assign m_axi_ws_arburst = 2'b01; assign m_axi_ld_arburst = 2'b01;
    assign m_axi_ws_arcache = 4'b0011; assign m_axi_ld_arcache = 4'b0011;
    assign m_axi_ws_arprot = 3'b000; assign m_axi_ld_arprot = 3'b000;
    assign m_axi_ws_rready = 1'b1;   assign m_axi_ld_rready = 1'b1;

    initial begin                                   // power-up values (FPGA init) for the output regs
        m_axi_ws_arvalid = 1'b0; m_axi_ld_arvalid = 1'b0; m_axi_ws_araddr = 40'd0; m_axi_ld_araddr = 40'd0;
        s_axi_awready = 1'b0; s_axi_wready = 1'b0; s_axi_bvalid = 1'b0; s_axi_bresp = 2'b00;
        s_axi_arready = 1'b0; s_axi_rvalid = 1'b0; s_axi_rresp = 2'b00; s_axi_rdata = 32'd0;
    end
    // ---------------- control (clk domain) ----------------
    reg        running = 1'b0, done = 1'b0, bp = 1'b0, start = 1'b0, abort = 1'b0;
    reg [39:0] ddr_base = 40'd0;
    reg [63:0] core_cyc = 0, wall_cyc = 0, ws_wait = 0, ld_wait = 0;
    reg [31:0] ld_miss_n = 0, ws_restart_n = 0;
    wire       rst_core = ~running;
    wire       ws_miss, ld_miss;
    wire       ce = ~running | ~(ws_miss | ld_miss);
    wire       cclk;
    BUFGCE #(.CE_TYPE("SYNC")) u_cg (.I(clk), .CE(ce), .O(cclk));

    // ---------------- the core ----------------
    wire [31:0] PC, ALUResult, WriteData_out, WS_Addr;
    wire        MemRead;
    wire [3:0]  MemWrite_out;
    reg  [31:0] ReadData_in;
    wire [31:0] Instr, WS_Data;
    reg         BP_s1 = 1'b0, BP_Enable = 1'b0;
    always @(posedge cclk) begin BP_s1 <= bp; BP_Enable <= BP_s1; end

    RV #(.PC_INIT(32'h0040_0000)) RV1(cclk, rst_core, BP_Enable, Instr, ReadData_in, MemRead, MemWrite_out,
                                      PC, ALUResult, WriteData_out, WS_Addr, WS_Data);

    // address decode (same ranges as the sim Wrapper)
    wire dec_DMEM = ALUResult[31:20] == 12'h100;
    wire dec_DDR  = ALUResult[31:23] == 9'h040;                  // 0x2000_0000, 8 MB
    wire dec_MMIO = ALUResult[31:8]  == 24'hFFFF00;
    wire [7:0] moff = {ALUResult[7:2], 2'b00};
    wire MemWrite = |MemWrite_out;

    // MMIO (only what the firmware uses on the sim: UART TX, SEVENSEG, CYCLECOUNT, DIP, flags)
    reg [31:0] cycle_count = 32'd0, sevenseg = 32'd0;
    reg        done_c = 1'b0;
    reg [12:0] con_n = 13'd0;
    reg [31:0] con_mem [0:8191];
    always @(posedge cclk) begin
        cycle_count <= rst_core ? 32'd0 : cycle_count + 32'd1;
        if (rst_core) begin
            done_c <= 1'b0;
        end else if (!done_c && MemWrite_out[0] && dec_MMIO && moff == 8'h0C) begin   // UART TX
            if (WriteData_out[7:0] == 8'h04) done_c <= 1'b1;
            else if (con_n != 13'h1FFF) begin
                con_mem[con_n] <= {24'd0, WriteData_out[7:0]};
                con_n <= con_n + 13'd1;
            end
        end
        if (!rst_core && dec_MMIO && moff == 8'h80) begin                            // SEVENSEG
            if (MemWrite_out[0]) sevenseg[7:0]   <= WriteData_out[7:0];
            if (MemWrite_out[1]) sevenseg[15:8]  <= WriteData_out[15:8];
            if (MemWrite_out[2]) sevenseg[23:16] <= WriteData_out[23:16];
            if (MemWrite_out[3]) sevenseg[31:24] <= WriteData_out[31:24];
        end
        if (start) begin con_n <= 13'd0; sevenseg <= 32'd0; end
    end
    reg [31:0] ReadData_MMIO;
    always @(*) begin
        if (!dec_MMIO) ReadData_MMIO = 32'd0;
        else case (moff)
            8'h64: ReadData_MMIO = {16'd0, BP_Enable, 15'd0};   // DIP (bit 15 = BP)
            8'h08: ReadData_MMIO = 32'd1;                       // UART TX ready
            8'hA0: ReadData_MMIO = cycle_count;
            default: ReadData_MMIO = 32'd0;                     // PB, UART RX, ACCEL: nothing on the board
        endcase
    end

    // ---------------- IROM: 16K x 32 BRAM, falling edge ----------------
    reg         a_irom_we = 1'b0, a_dmem_we = 1'b0;
    reg  [20:0] a_addr = 21'd0;
    reg  [31:0] a_wdata = 32'd0;
    reg  [3:0]  a_wstrb = 4'd0;
    wire [31:0] irom_b;
    wire [31:0] irom_a;
    xpm_memory_tdpram #(
        .MEMORY_SIZE(16384 * 32), .MEMORY_PRIMITIVE("block"), .CLOCKING_MODE("common_clock"),
        .WRITE_DATA_WIDTH_A(32), .READ_DATA_WIDTH_A(32), .BYTE_WRITE_WIDTH_A(32), .ADDR_WIDTH_A(14),
        .WRITE_DATA_WIDTH_B(32), .READ_DATA_WIDTH_B(32), .BYTE_WRITE_WIDTH_B(32), .ADDR_WIDTH_B(14),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1), .WRITE_MODE_A("read_first"), .WRITE_MODE_B("read_first"),
        .MEMORY_INIT_FILE("none"), .USE_MEM_INIT(0), .ECC_MODE("no_ecc"), .SIM_ASSERT_CHK(0)
    ) u_irom (
        .clka(~cclk), .rsta(1'b0), .ena(1'b1), .regcea(1'b1), .wea(1'b0), .addra(PC[15:2]), .dina(32'd0),
        .douta(irom_a),
        .clkb(~cclk), .rstb(1'b0), .enb(1'b1), .regceb(1'b1), .web(a_irom_we), .addrb(a_addr[15:2]),
        .dinb(a_wdata), .doutb(irom_b),
        .sleep(1'b0), .injectsbiterra(1'b0), .injectdbiterra(1'b0), .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb()
    );
    // invalid fetch address -> NOP, as in the Wrapper (PC is stable through the cycle)
    assign Instr = (PC[31:16] == 16'h0040 && PC[1:0] == 2'b00) ? irom_a : 32'h00000013;

    // ---------------- DMEM: 128K x 64 URAM, byte writes, falling edge ----------------
    wire [63:0] dmem_a, dmem_b;
    wire [7:0]  dmem_wea = ALUResult[2] ? {(dec_DMEM ? MemWrite_out : 4'd0), 4'd0}
                                        : {4'd0, (dec_DMEM ? MemWrite_out : 4'd0)};
    wire [7:0]  dmem_web = a_dmem_we ? (a_addr[2] ? {a_wstrb, 4'd0} : {4'd0, a_wstrb}) : 8'd0;
    xpm_memory_tdpram #(
        .MEMORY_SIZE(131072 * 64), .MEMORY_PRIMITIVE("ultra"), .CLOCKING_MODE("common_clock"),
        .WRITE_DATA_WIDTH_A(64), .READ_DATA_WIDTH_A(64), .BYTE_WRITE_WIDTH_A(8), .ADDR_WIDTH_A(17),
        .WRITE_DATA_WIDTH_B(64), .READ_DATA_WIDTH_B(64), .BYTE_WRITE_WIDTH_B(8), .ADDR_WIDTH_B(17),
        .READ_LATENCY_A(1), .READ_LATENCY_B(1), .WRITE_MODE_A("no_change"), .WRITE_MODE_B("no_change"),
        .MEMORY_INIT_FILE("none"), .USE_MEM_INIT(0), .ECC_MODE("no_ecc"), .SIM_ASSERT_CHK(0)
    ) u_dmem (
        .clka(~cclk), .rsta(1'b0), .ena(1'b1), .regcea(1'b1), .wea(dmem_wea), .addra(ALUResult[19:3]),
        .dina({WriteData_out, WriteData_out}), .douta(dmem_a),
        .clkb(~cclk), .rstb(1'b0), .enb(1'b1), .regceb(1'b1), .web(dmem_web), .addrb(a_addr[19:3]),
        .dinb({a_wdata, a_wdata}), .doutb(dmem_b),
        .sleep(1'b0), .injectsbiterra(1'b0), .injectdbiterra(1'b0), .injectsbiterrb(1'b0), .injectdbiterrb(1'b0),
        .sbiterra(), .dbiterra(), .sbiterrb(), .dbiterrb()
    );
    // the row half is picked with the same ALUResult[2] (stable through the cycle)
    wire [31:0] ReadData_DMEM = ALUResult[2] ? dmem_a[63:32] : dmem_a[31:0];

    // ---------------- weight stream: sequential prefetch window over DDR ----------------
    // Window of 16 B beats [hd, hd+cnt) in a 32-beat ring; beats [hd+cnt, fb) are requested, not back yet.
    localparam WSN = 32;
    reg [127:0] wsb [0:WSN-1];
    reg [4:0]   ws_head = 5'd0;
    reg [5:0]   ws_cnt = 6'd0, ws_infl = 6'd0, ws_disc = 6'd0;
    reg [18:0]  ws_hd = 19'd0, ws_fb = 19'd0;     // beat numbers (byte offset >> 4) in the 8 MB window
    wire [18:0] ws_beat = WS_Addr[22:4];
    wire [18:0] ws_rel = ws_beat - ws_hd;
    wire        ws_hit = (ws_beat >= ws_hd) && (ws_rel < ws_cnt);
    // will arrive without a restart: already requested, or in the next burst (issued as soon as there is room)
    wire        ws_inexp = (ws_beat >= ws_hd) && (ws_beat < ws_fb + 19'd4);
    wire [127:0] ws_row = wsb[ws_head + ws_rel[4:0]];
    assign WS_Data = ws_row[WS_Addr[3:2]*32 +: 32];
    assign ws_miss = ~ws_hit;
    always @(posedge clk) begin : wsf
        reg [5:0] cnt, infl, disc;
        reg [4:0] head;
        cnt = ws_cnt; infl = ws_infl; disc = ws_disc; head = ws_head;
        // returning beats
        if (m_axi_ws_rvalid) begin
            infl = infl - 6'd1;
            if (disc != 0) disc = disc - 6'd1;
            else begin wsb[head + cnt[4:0]] <= m_axi_ws_rdata; cnt = cnt + 6'd1; end
        end
        if (m_axi_ws_arvalid && m_axi_ws_arready) m_axi_ws_arvalid <= 1'b0;
        if (!running || start) begin
            // idle: forget the window; responses still in flight are dropped as they come back
            disc = infl; cnt = 6'd0; ws_hd <= 19'd0; ws_fb <= 19'd0;
            if (start) ws_restart_n <= 32'd0;
        end else if (!ws_inexp) begin
            // the stream jumped (WSTART): restart at the 64 B block holding ws_addr
            disc = infl; cnt = 6'd0;
            ws_hd <= {ws_beat[18:2], 2'b00}; ws_fb <= {ws_beat[18:2], 2'b00};
            ws_restart_n <= ws_restart_n + 32'd1;
        end else begin
            // drop the beat behind the stream (the stream moves at most one word per cycle)
            if (ws_beat != ws_hd && cnt != 0) begin
                head = head + 5'd1; cnt = cnt - 6'd1; ws_hd <= ws_hd + 19'd1;
            end
            // keep the ring full: one 4-beat burst at a time
            if (!m_axi_ws_arvalid && (cnt + infl) <= WSN - 4) begin
                m_axi_ws_araddr <= ddr_base + {17'd0, ws_fb, 4'd0};
                m_axi_ws_arvalid <= 1'b1;
                infl = infl + 6'd4;
                ws_fb <= ws_fb + 19'd4;
            end
        end
        ws_cnt <= cnt; ws_infl <= infl; ws_disc <= disc; ws_head <= head;
    end

    // ---------------- core loads from the image: 512 lines x 64 B (32 KB), direct mapped ----------------
    reg [127:0] ldm [0:2047];                // {line, beat}
    reg [7:0]   ld_tag [0:511];
    reg [511:0] ld_val = 512'd0;
    reg         ld_busy = 1'b0;
    reg [1:0]   ld_beat = 2'd0;
    reg [8:0]   ld_line = 9'd0;
    wire [22:0] ld_off = ALUResult[22:0];
    wire        ld_hit = ld_val[ld_off[14:6]] && ld_tag[ld_off[14:6]] == ld_off[22:15];
    assign ld_miss = MemRead && dec_DDR && !ld_hit;
    wire [127:0] ld_row = ldm[{ld_off[14:6], ld_off[5:4]}];
    wire [31:0] ReadData_DDR = ld_row[ld_off[3:2]*32 +: 32];
    always @(posedge clk) begin
        if (m_axi_ld_arvalid && m_axi_ld_arready) m_axi_ld_arvalid <= 1'b0;
        if (!running || start) begin
            ld_val <= 512'd0;
            if (start) ld_miss_n <= 32'd0;
        end else if (!ld_busy && ld_miss) begin
            ld_busy <= 1'b1; ld_beat <= 2'd0; ld_line <= ld_off[14:6];
            ld_tag[ld_off[14:6]] <= ld_off[22:15];
            ld_val[ld_off[14:6]] <= 1'b0;
            m_axi_ld_araddr <= ddr_base + {17'd0, ld_off[22:6], 6'd0};
            m_axi_ld_arvalid <= 1'b1;
            ld_miss_n <= ld_miss_n + 32'd1;
        end
        if (ld_busy && m_axi_ld_rvalid) begin
            ldm[{ld_line, ld_beat}] <= m_axi_ld_rdata;
            ld_beat <= ld_beat + 2'd1;
            if (m_axi_ld_rlast) begin ld_busy <= 1'b0; ld_val[ld_line] <= running && !start; end
        end
    end

    always @(*) begin
        if (dec_DMEM)     ReadData_in = ReadData_DMEM;
        else if (dec_DDR) ReadData_in = ReadData_DDR;
        else              ReadData_in = ReadData_MMIO;
    end

    // ---------------- run control + counters ----------------
    always @(posedge clk) begin
        if (running) begin
            wall_cyc <= wall_cyc + 64'd1;
            if (ce && !done_c) core_cyc <= core_cyc + 64'd1;
            if (ws_miss) ws_wait <= ws_wait + 64'd1;
            else if (ld_miss) ld_wait <= ld_wait + 64'd1;
            if (done_c) begin running <= 1'b0; done <= 1'b1; end
        end
        if (start) begin
            running <= 1'b1; done <= 1'b0;
            core_cyc <= 0; wall_cyc <= 0; ws_wait <= 0; ld_wait <= 0;
        end
        if (abort) running <= 1'b0;
        if (!aresetn) begin running <= 1'b0; done <= 1'b0; end
    end

    // ---------------- AXI-Lite slave ----------------
    // One access at a time. Memory windows are only written while the core is stopped.
    reg        rd_pend = 1'b0, rd_pend2 = 1'b0;
    reg [31:0] con_q = 32'd0;
    always @(posedge clk) con_q <= con_mem[a_addr[14:2]];      // BRAM read port (clk side)
    reg [20:0] rd_addr = 21'd0;
    always @(posedge clk) begin
        start <= 1'b0; abort <= 1'b0;
        a_irom_we <= 1'b0; a_dmem_we <= 1'b0;
        s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_arready <= 1'b0;
        if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
        if (!aresetn) begin
            s_axi_bvalid <= 1'b0; s_axi_rvalid <= 1'b0; rd_pend <= 1'b0; rd_pend2 <= 1'b0;
        end else if (s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid && !s_axi_awready && !rd_pend && !rd_pend2) begin
            s_axi_awready <= 1'b1; s_axi_wready <= 1'b1; s_axi_bvalid <= 1'b1; s_axi_bresp <= 2'b00;
            a_addr <= s_axi_awaddr; a_wdata <= s_axi_wdata; a_wstrb <= s_axi_wstrb;
            if (s_axi_awaddr[20]) a_dmem_we <= !running;
            else if (s_axi_awaddr[20:16] == 5'h01) a_irom_we <= !running;
            else if (s_axi_awaddr[20:16] == 5'h00) begin
                case (s_axi_awaddr[7:0])
                    8'h04: begin bp <= s_axi_wdata[1]; if (s_axi_wdata[0]) start <= 1'b1; else abort <= 1'b1; end
                    8'h10: ddr_base[31:0] <= s_axi_wdata;
                    8'h14: ddr_base[39:32] <= s_axi_wdata[7:0];
                    default: ;
                endcase
            end
        end else if (s_axi_arvalid && !s_axi_rvalid && !s_axi_arready && !rd_pend && !rd_pend2) begin
            s_axi_arready <= 1'b1; rd_pend <= 1'b1; rd_addr <= s_axi_araddr; a_addr <= s_axi_araddr;
        end else if (rd_pend) begin
            rd_pend <= 1'b0; rd_pend2 <= 1'b1;      // one more cycle for the console BRAM (con_q)
        end else if (rd_pend2) begin
            // memories captured a_addr on a falling edge; their outputs are still held
            rd_pend2 <= 1'b0; s_axi_rvalid <= 1'b1; s_axi_rresp <= 2'b00;
            if (rd_addr[20]) s_axi_rdata <= rd_addr[2] ? dmem_b[63:32] : dmem_b[31:0];
            else if (rd_addr[20:16] == 5'h01) s_axi_rdata <= irom_b;
            else if (rd_addr[20:16] == 5'h02) s_axi_rdata <= con_q;
            else case (rd_addr[7:0])
                8'h00: s_axi_rdata <= 32'h5C0E0003;
                8'h04: s_axi_rdata <= {30'd0, bp, running};
                8'h08: s_axi_rdata <= {30'd0, done, running};
                8'h0C: s_axi_rdata <= sevenseg;
                8'h10: s_axi_rdata <= ddr_base[31:0];
                8'h14: s_axi_rdata <= {24'd0, ddr_base[39:32]};
                8'h18: s_axi_rdata <= {19'd0, con_n};
                8'h20: s_axi_rdata <= core_cyc[31:0];  8'h24: s_axi_rdata <= core_cyc[63:32];
                8'h28: s_axi_rdata <= wall_cyc[31:0];  8'h2C: s_axi_rdata <= wall_cyc[63:32];
                8'h30: s_axi_rdata <= ws_wait[31:0];   8'h34: s_axi_rdata <= ws_wait[63:32];
                8'h38: s_axi_rdata <= ld_wait[31:0];   8'h3C: s_axi_rdata <= ld_wait[63:32];
                8'h40: s_axi_rdata <= ld_miss_n;
                8'h44: s_axi_rdata <= ws_restart_n;
                default: s_axi_rdata <= 32'hDEADBEEF;
            endcase
        end
    end
endmodule
