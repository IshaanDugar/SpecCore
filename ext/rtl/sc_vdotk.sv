// SpecCore LLM unit: K-lane int8 dot product next to the ALU, in the E stage.
//
// One instruction class, custom-0 opcode (0001011), R-type. funct3 picks the operation:
//   0 SETK    K = rs1 (active lanes, 1..LANES)
//   1 SETP    funct7[6] = 0: ptr[funct7[2:0]] = rs1;  funct7[6] = 1: every lane's ptr = rs1  (word index in the bank)
//             funct7[5] = 1: SETS, lane scale of lane funct7[2:0]: sm = rs1 (mantissa), se = rs2 (exponent)
//   2 VDOTR   w = rs1 (word from a register, "R1" in the speedup audit): for each active lane k:
//             acc[k] += dot4(w, bank[k][ptr[k]]); ptr[k] += 1.  One lw of a K/V word feeds all K lanes.
//   3 ACCRD   rd = acc[funct7[2:0]]; acc[funct7[2:0]] = 0
//             funct7[6] = 1: ACCQ, scaled readout (D13): rd = apply(acc[l], smul(sm[l], se[l], rs1, rs2), 12),
//             acc[l] = 0. Bit-exact with ref/qref.c smul() + apply32(). Multi-cycle: holds E for QCYC cycles.
//   4 VDOTK   w = pop(weight FIFO), then as VDOTR. Stalls while the FIFO is empty.
//   5 WPOP    rd = pop(weight FIFO). Stalls while the FIFO is empty.
//   6 BSTORE  bank[funct7[2:0]][rs1] = rs2  (fill a lane's activation bank, one word)
//   7 WSTART  weight stream: byte address rs1, rs2 words; empties the FIFO first
// dot4(a, b) = sum over the 4 bytes of signed(a.byte) * signed(b.byte).
// ACCQ arithmetic (the number system in model/qmodel.py; scale = mantissa m in [2^29, 2^30) or 0, exponent e):
//   p = sm * rs1 (64 bit); p == 0 -> 0;  p >= 2^59: m = p >> 30, e = se + rs2 - 30;  else m = p >> 29, e = se + rs2 - 29
//   sh = e - 12;  m == 0 or sh > 62 -> 0;  sh < 1 -> 0 (never happens in the model; SVA A7)
//   rd = bits [31:0] of (acc * m + 2^(sh-1)) >>> sh   (64-bit, arithmetic shift = round half up)
//
// Pipeline contract (docs/core_facts.md: an instruction that reaches E is never squashed):
//   - every state change happens on the clock edge where the instruction LEAVES E (commit = valid & advance);
//     while E is held nothing changes, so a stall cannot double-count.
//   - stall is raised by VDOTK/WPOP with an empty FIFO and by ACCQ while it computes; RV.v ORs it into
//     the MCycle Busy path (hold F, D, E; bubble into M).
//   - ACCQ latches rs1/rs2 (and acc, sm, se) in its FIRST E cycle: while E is held, M gets bubbles and the
//     producer of a forwarded operand retires from W, so rs1/rs2 can go stale later in the hold.
//   - ACCRD/WPOP results leave through the normal E result mux, so forwarding works unchanged.
// Weight stream: one word per cycle from the model image (ws_addr -> ws_data, async read in the
// sim Wrapper) whenever the FIFO has room. WS_GAP > 0 inserts idle cycles between words to model
// a slower memory (sensitivity runs).
module sc_vdotk #(
    parameter int LANES = 8,
    parameter int BANK_WORDS = 256,       // 1 KB per lane: enough for a 1024-wide int8 vector
    parameter int FIFO_DEPTH = 16,
    parameter int WS_GAP = 0
) (
    input  logic        clk,
    input  logic        rst,
    input  logic        valid,            // a custom-0 instruction is in E
    input  logic [2:0]  f3,
    input  logic [6:0]  f7,
    input  logic [31:0] rs1,
    input  logic [31:0] rs2,
    input  logic        advance,          // the instruction in E moves to M on this edge (~StallE)
    output logic [31:0] result,
    output logic        stall,
    output logic [31:0] ws_addr,
    input  logic [31:0] ws_data
);
    localparam int LW = $clog2(LANES);
    localparam int BW = $clog2(BANK_WORDS);
    localparam int FW = $clog2(FIFO_DEPTH);

    localparam logic [2:0] F_SETK = 3'd0, F_SETP = 3'd1, F_VDOTR = 3'd2, F_ACCRD = 3'd3,
                           F_VDOTK = 3'd4, F_WPOP = 3'd5, F_BSTORE = 3'd6, F_WSTART = 3'd7;

    // ---------------- state ----------------
    // Activation banks: one 1-D memory per lane (generate block below), so synthesis maps each to
    // LUTRAM (1 write port, 1 async read). A single 2-D array became 65,536 flip-flops and hit the
    // 6 GB synth memory cap (16 Nov). bank_rd[k] = bank k at ptr[k].
    logic [31:0] bank_rd [LANES];
    logic [BW-1:0] ptr [LANES];
    logic signed [31:0] acc [LANES];
    logic [LW:0] k_act;                   // 1..LANES
    logic [31:0] fifo [FIFO_DEPTH];
    logic [FW-1:0] rd_i, wr_i;
    logic [FW:0] count;
    logic [31:0] ws_ptr, ws_left;
    logic [7:0] gap;
    logic [31:0] sm [LANES];              // lane scales (SETS): mantissa, exponent
    logic signed [31:0] se [LANES];

    // ---------------- decode ----------------
    wire [LW-1:0] lane = f7[LW-1:0];
    wire needs_word = valid && (f3 == F_VDOTK || f3 == F_WPOP);
    wire empty = (count == 0);
    wire is_accq = valid && (f3 == F_ACCRD) && f7[6];
    localparam int QCYC = 5;              // ACCQ compute cycles (E held); commits in cycle QCYC+1
    logic [2:0] qst;                      // 0 idle, 1..QCYC computing, QCYC = result ready
    assign stall = (needs_word && empty) || (is_accq && qst != 3'(QCYC));
    wire commit = valid && advance;
    wire pop = commit && (f3 == F_VDOTK || f3 == F_WPOP);
    wire mac = commit && (f3 == F_VDOTK || f3 == F_VDOTR);
    wire [31:0] w = (f3 == F_VDOTR) ? rs1 : fifo[rd_i];

    // ---------------- activation banks (BSTORE writes on commit, read at ptr) ----------------
    for (genvar g = 0; g < LANES; g++) begin : g_bank
        logic [31:0] mem [BANK_WORDS];
        always_ff @(posedge clk)
            if (commit && f3 == F_BSTORE && lane == LW'(g)) mem[rs1[BW-1:0]] <= rs2;
        assign bank_rd[g] = mem[ptr[g]];
    end

    // ---------------- lanes: 4 int8 multipliers + adder per lane ----------------
    logic signed [31:0] dot [LANES];
    always_comb begin
        for (int k = 0; k < LANES; k++) begin
            logic [31:0] b;
            b = bank_rd[k];
            dot[k] = 32'sd0;
            for (int i = 0; i < 4; i++)
                dot[k] += $signed(w[8*i +: 8]) * $signed(b[8*i +: 8]);
        end
    end

    // ---------------- ACCQ: scaled readout, 5 registered stages ----------------
    // S1 latch operands (first E cycle) | S2 p = sm * rm | S3 normalise -> m, sh | S4 prod = acc * m |
    // S5 out = (prod + 2^(sh-1)) >>> sh. Every stage is registered so ACCQ adds no long path (8 ns clock).
    logic signed [31:0] q_v, q_se, q_re, q_sh;
    logic [31:0] q_sm, q_rm, q_m;
    logic [63:0] q_p;
    logic q_zero;
    logic signed [64:0] q_prod;
    logic [31:0] q_out;
    logic signed [64:0] q_rnd;
    always_comb q_rnd = q_prod + (65'sd1 <<< (q_sh[5:0] - 6'd1));
    always_ff @(posedge clk) begin
        if (rst) qst <= '0;
        else if (!valid) qst <= '0;                    // nothing in E (e.g. a flush): never keep a stale result
        else if (is_accq && qst == 0) begin           // S1: first cycle in E, operands are fresh here
            q_v <= acc[lane]; q_sm <= sm[lane]; q_se <= se[lane]; q_rm <= rs1; q_re <= rs2;
            qst <= 3'd1;
        end else if (qst == 3'd1) begin               // S2
            q_p <= 64'(q_sm) * 64'(q_rm);
            qst <= 3'd2;
        end else if (qst == 3'd2) begin               // S3
            logic signed [31:0] e;
            if (q_p >= 64'h0800_0000_0000_0000) begin q_m <= q_p[61:30]; e = q_se + q_re - 32'sd30; end
            else begin q_m <= q_p[60:29]; e = q_se + q_re - 32'sd29; end
            q_sh <= e - 32'sd12;
            q_zero <= (q_p == 0) || (q_p < 64'h2000_0000) || (e - 32'sd12 > 32'sd62) || (e - 32'sd12 < 32'sd1);
            qst <= 3'd3;
        end else if (qst == 3'd3) begin               // S4
            q_prod <= 65'(q_v) * $signed({33'd0, q_m});
            qst <= 3'd4;
        end else if (qst == 3'd4) begin               // S5
            q_out <= q_zero ? 32'd0 : 32'(q_rnd >>> q_sh[5:0]);
            qst <= 3'd5;
        end else if (qst == 3'(QCYC) && commit) qst <= '0;
    end

    assign result = (f3 == F_ACCRD) ? (f7[6] ? q_out : acc[lane]) : fifo[rd_i];

    // ---------------- weight stream -> FIFO ----------------
    wire wstart = commit && (f3 == F_WSTART);
    wire push = !wstart && (ws_left != 0) && (gap == 0) && (count < FIFO_DEPTH);
    assign ws_addr = ws_ptr;

    always_ff @(posedge clk) begin
        if (rst) begin
            rd_i <= '0; wr_i <= '0; count <= '0; ws_ptr <= '0; ws_left <= '0; gap <= '0;
        end else if (wstart) begin
            rd_i <= '0; wr_i <= '0; count <= '0;
            ws_ptr <= rs1; ws_left <= rs2; gap <= '0;
        end else begin
            if (push) begin
                fifo[wr_i] <= ws_data;
                wr_i <= wr_i + 1'b1;
                ws_ptr <= ws_ptr + 32'd4;
                ws_left <= ws_left - 32'd1;
                gap <= 8'(WS_GAP);
            end else if (gap != 0) begin
                gap <= gap - 8'd1;
            end
            if (pop) rd_i <= rd_i + 1'b1;
            count <= count + (push ? 1'b1 : 1'b0) - (pop ? 1'b1 : 1'b0);
        end
    end

    // ---------------- architectural state, changed only on commit ----------------
    always_ff @(posedge clk) begin
        if (rst) begin
            k_act <= 1;
            for (int k = 0; k < LANES; k++) begin ptr[k] <= '0; acc[k] <= '0; sm[k] <= '0; se[k] <= '0; end
        end else if (commit) begin
            case (f3)
                F_SETK: k_act <= (rs1 == 0) ? 1 : (rs1 > LANES) ? (LW+1)'(LANES) : rs1[LW:0];
                F_SETP:
                    if (f7[5]) begin sm[lane] <= rs1; se[lane] <= rs2; end
                    else if (f7[6]) for (int k = 0; k < LANES; k++) ptr[k] <= rs1[BW-1:0];
                    else ptr[lane] <= rs1[BW-1:0];
                F_ACCRD: acc[lane] <= '0;
                default: ;
            endcase
            if (mac)
                for (int k = 0; k < LANES; k++)
                    if (k < k_act) begin
                        acc[k] <= acc[k] + dot[k];
                        ptr[k] <= ptr[k] + 1'b1;
                    end
        end
    end

    // ---------------- assertions (each in plain words) ----------------
    // Failures are counted here; core/tb/tb_sc.sv adds this count to its own SVA total.
    int sva_fail = 0;
    // A1. Never pop an empty FIFO: a VDOTK/WPOP can only leave E when a word is there.
    a_no_pop_empty: assert property (@(posedge clk) disable iff (rst) pop |-> !empty)
        else begin sva_fail++; $error("sc_vdotk: pop from an empty FIFO"); end
    // A2. While stalled, the instruction must not commit (stall and advance are exclusive).
    a_stall_holds: assert property (@(posedge clk) disable iff (rst) stall |-> !advance)
        else begin sva_fail++; $error("sc_vdotk: advanced while stalled"); end
    // A3. The FIFO never holds more words than it has slots.
    a_count: assert property (@(posedge clk) disable iff (rst) count <= FIFO_DEPTH)
        else begin sva_fail++; $error("sc_vdotk: FIFO overflow"); end
    // A4. K is always a legal lane count.
    a_k: assert property (@(posedge clk) disable iff (rst) k_act >= 1 && k_act <= LANES)
        else begin sva_fail++; $error("sc_vdotk: illegal K"); end
    // A5. Hidden state moves only on commit: if any accumulator changed, a MAC or ACCRD committed
    //     on the previous edge (this is the "no double counting during a stall" rule).
    logic [31:0] acc0_q;
    always_ff @(posedge clk) acc0_q <= acc[0];
    a_acc_commit: assert property (@(posedge clk) disable iff (rst)
        (acc[0] != acc0_q) |-> $past(commit && (f3 == F_VDOTK || f3 == F_VDOTR || f3 == F_ACCRD)))
        else begin sva_fail++; $error("sc_vdotk: acc changed without a commit"); end
    // A6. An ACCQ leaves E only with its result ready (all compute cycles done).
    a_accq_ready: assert property (@(posedge clk) disable iff (rst) (is_accq && commit) |-> qst == 3'(QCYC))
        else begin sva_fail++; $error("sc_vdotk: ACCQ committed before its result was ready"); end
    // A7. ACCQ inputs are legal scales: both mantissas are 0 or in [2^29, 2^30), and the shift is >= 1
    //     whenever the product is not 0 (sh < 1 is undefined in the C reference, so a firmware bug;
    //     sh > 62 is defined: the result is 0).
    a_accq_legal: assert property (@(posedge clk) disable iff (rst) (qst == 3'd1) |->
        ((q_sm == 0 || q_sm[31:29] == 3'b001) && (q_rm == 0 || q_rm[31:29] == 3'b001)))
        else begin sva_fail++; $error("sc_vdotk: ACCQ scale mantissa out of range"); end
    a_accq_shift: assert property (@(posedge clk) disable iff (rst) (qst == 3'd3 && q_p >= 64'h2000_0000) |->
        q_sh >= 1)
        else begin sva_fail++; $error("sc_vdotk: ACCQ shift out of range"); end
    // A8. ACCQ takes its operands from its FIRST E cycle (the stale-forwarding rule above): one edge after
    //     an ACCQ starts, the latches hold exactly what rs1, rs2 and the accumulator were at the start.
    a_accq_latch: assert property (@(posedge clk) disable iff (rst) (is_accq && qst == 0) |=>
        q_rm == $past(rs1) && q_re == $past(rs2) && q_v == $past(acc[lane]))
        else begin sva_fail++; $error("sc_vdotk: ACCQ did not latch its first-cycle operands"); end
    // A9. The ACCQ machine only runs while an ACCQ sits in E.
    a_accq_owner: assert property (@(posedge clk) disable iff (rst) (qst != 0) |-> is_accq)
        else begin sva_fail++; $error("sc_vdotk: ACCQ state busy without an ACCQ in E"); end
endmodule
