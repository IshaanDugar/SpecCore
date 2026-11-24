`timescale 1ns / 1ps
// xsim test of board/core/sc_board.v before a bitstream build: the A53 side is AXI-Lite tasks, DDR is
// a model with random read latency (10-60 cycles) holding the model image (AA_DDR.mem).
// Loads AA_IROM.mem / AA_DMEM.mem through the AXI-Lite window, starts the core, waits for done,
// prints the console, exit code and counters. Compare with the same program's tb_sc output.
// Plusargs: +bp (branch predictor on), +maxcyc_N (timeout in clk cycles).
module tb_board;
    logic clk = 0, aresetn = 0;
    always #10 clk = ~clk;                                   // 50 MHz
    localparam logic [39:0] BASE = 40'h6000_0000;

    logic [20:0] awaddr, araddr; logic awvalid, wvalid, bready = 1, arvalid, rready = 1;
    logic [31:0] wdata; logic [3:0] wstrb;
    wire awready, wready, bvalid, arready, rvalid; wire [1:0] bresp, rresp; wire [31:0] rdata;
    wire [39:0] ws_araddr, ld_araddr; wire ws_arvalid, ld_arvalid;
    logic ws_arready, ld_arready, ws_rvalid, ld_rvalid, ws_rlast, ld_rlast;
    logic [127:0] ws_rdata, ld_rdata;

    sc_board dut (.clk, .aresetn,
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready), .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid), .s_axi_bready(bready), .s_axi_araddr(araddr), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready),
        .m_axi_ws_araddr(ws_araddr), .m_axi_ws_arlen(), .m_axi_ws_arsize(), .m_axi_ws_arburst(),
        .m_axi_ws_arcache(), .m_axi_ws_arprot(), .m_axi_ws_arvalid(ws_arvalid), .m_axi_ws_arready(ws_arready),
        .m_axi_ws_rdata(ws_rdata), .m_axi_ws_rresp(2'b00), .m_axi_ws_rlast(ws_rlast), .m_axi_ws_rvalid(ws_rvalid),
        .m_axi_ws_rready(),
        .m_axi_ld_araddr(ld_araddr), .m_axi_ld_arlen(), .m_axi_ld_arsize(), .m_axi_ld_arburst(),
        .m_axi_ld_arcache(), .m_axi_ld_arprot(), .m_axi_ld_arvalid(ld_arvalid), .m_axi_ld_arready(ld_arready),
        .m_axi_ld_rdata(ld_rdata), .m_axi_ld_rresp(2'b00), .m_axi_ld_rlast(ld_rlast), .m_axi_ld_rvalid(ld_rvalid),
        .m_axi_ld_rready());

    // ---------------- DDR model: in-order bursts of 4 x 16 B, random latency ----------------
    logic [31:0] ddr [0:2097151];
    function automatic logic [127:0] beat(input logic [39:0] a);
        int w = int'((a - BASE) >> 2);
        return {ddr[w + 3], ddr[w + 2], ddr[w + 1], ddr[w]};
    endfunction
    task automatic port(input bit ws);
        logic [39:0] q[$]; int due[$]; int t = 0;
        forever begin
            @(posedge clk); t++;
            if (ws) begin ws_arready <= $urandom_range(0, 3) != 0; ws_rvalid <= 0; ws_rlast <= 0; end
            else    begin ld_arready <= $urandom_range(0, 3) != 0; ld_rvalid <= 0; ld_rlast <= 0; end
            if (ws ? (ws_arvalid && ws_arready) : (ld_arvalid && ld_arready)) begin
                q.push_back(ws ? ws_araddr : ld_araddr); due.push_back(t + $urandom_range(10, 60));
            end
            if (q.size() && due[0] <= t) begin
                for (int b = 0; b < 4; b++) begin
                    if (ws) begin ws_rvalid <= 1; ws_rdata <= beat(q[0] + 16 * b); ws_rlast <= b == 3; end
                    else    begin ld_rvalid <= 1; ld_rdata <= beat(q[0] + 16 * b); ld_rlast <= b == 3; end
                    if (b < 3) begin
                        @(posedge clk); t++;
                        if (ws ? (ws_arvalid && ws_arready) : (ld_arvalid && ld_arready)) begin
                            q.push_back(ws ? ws_araddr : ld_araddr); due.push_back(t + $urandom_range(10, 60));
                        end
                    end
                end
                void'(q.pop_front()); void'(due.pop_front());
            end
        end
    endtask

    // ---------------- AXI-Lite ----------------
    task automatic wr(input logic [20:0] a, input logic [31:0] d);
        @(posedge clk); awaddr <= a; wdata <= d; wstrb <= 4'hF; awvalid <= 1; wvalid <= 1;
        do @(posedge clk); while (!(awready && wready));
        awvalid <= 0; wvalid <= 0;                         // bvalid came with awready (bready = 1)
    endtask
    task automatic rd(input logic [20:0] a, output logic [31:0] d);
        @(posedge clk); araddr <= a; arvalid <= 1;
        do @(posedge clk); while (!arready);
        arvalid <= 0;
        do @(posedge clk); while (!rvalid);
        d = rdata;
    endtask
    function automatic longint u64(input logic [31:0] lo, hi); return {hi, lo}; endfunction

    logic [31:0] irom [0:16383], dmem [0:262143];
    initial begin
        logic [31:0] v, lo, hi, n; int ni = 0, nd = 0, bad = 0, maxcyc; string s = "";
        awvalid = 0; wvalid = 0; arvalid = 0; ws_rvalid = 0; ld_rvalid = 0; ws_arready = 0; ld_arready = 0;
        foreach (irom[i]) irom[i] = 'x;
        foreach (dmem[i]) dmem[i] = 'x;
        $readmemh("AA_IROM.mem", irom); $readmemh("AA_DMEM.mem", dmem); $readmemh("AA_DDR.mem", ddr);
        if (!$value$plusargs("maxcyc_%d", maxcyc)) maxcyc = 3_000_000;
        fork port(1); port(0); join_none
        repeat (5) @(posedge clk); aresetn = 1;
        rd(21'h0, v); $display("[tbb] ID %08h", v);
        foreach (irom[i]) if (!$isunknown(irom[i])) begin wr(21'h10000 + 4 * i, irom[i]); ni++; end
        foreach (dmem[i]) if (!$isunknown(dmem[i])) begin wr(21'h100000 + 4 * i, dmem[i]); nd++; end
        for (int i = 0; i < ni; i += 97) begin rd(21'h10000 + 4 * i, v); bad += v !== irom[i]; end
        for (int i = 0; i < nd; i += 7) begin rd(21'h100000 + 4 * i, v); bad += v !== dmem[i]; end
        $display("[tbb] loaded IROM %0d words, DMEM %0d words, readback mismatches %0d", ni, nd, bad);
        wr(21'h10, BASE[31:0]); wr(21'h14, 32'(BASE[39:32]));
        wr(21'h04, $test$plusargs("bp") ? 32'h3 : 32'h1);
        for (int t = 0; ; t++) begin
            repeat (200) @(posedge clk);
            rd(21'h08, v);
            if (v[1]) break;
            if (t * 200 > maxcyc) begin
                $display("[tbb] TIMEOUT run %b ce %b wsmiss %b ldmiss %b PC %h WS %h hd %h fb %h cnt %0d infl %0d disc %0d arv %b ALU %h MemRead %b ldbusy %b corecyc %0d",
                    dut.running, dut.ce, dut.ws_miss, dut.ld_miss, dut.PC, dut.WS_Addr, dut.ws_hd, dut.ws_fb, dut.ws_cnt,
                    dut.ws_infl, dut.ws_disc, dut.m_axi_ws_arvalid, dut.ALUResult, dut.MemRead, dut.ld_busy, dut.core_cyc);
                $finish;
            end
        end
        rd(21'h18, n);
        for (int i = 0; i < n; i++) begin rd(21'h20000 + 4 * i, v); s = {s, string'(v[7:0])}; end
        $write("%s", s);
        rd(21'h0C, v); $display("[tbb] EXIT code %0d", $signed(v));
        rd(21'h20, lo); rd(21'h24, hi); $display("[tbb] core cycles %0d", u64(lo, hi));
        rd(21'h28, lo); rd(21'h2C, hi); $display("[tbb] wall cycles %0d", u64(lo, hi));
        rd(21'h30, lo); rd(21'h34, hi); $display("[tbb] stream-wait cycles %0d", u64(lo, hi));
        rd(21'h38, lo); rd(21'h3C, hi); $display("[tbb] load-wait cycles %0d", u64(lo, hi));
        rd(21'h40, v); $display("[tbb] load-cache misses %0d", v);
        rd(21'h44, v); $display("[tbb] stream restarts %0d", v);
        $finish;
    end
endmodule
