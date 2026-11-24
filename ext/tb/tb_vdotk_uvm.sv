// UVM testbench for the LLM unit (ext/rtl/sc_vdotk.sv), standalone (no core).
//   python tools/sc.py uvm [--seed N] [--n N]
//
// What each part does (plain words):
//   vdotk_if     the wires between the "pipeline" and the unit: one instruction slot (valid, f3,
//                f7, rs1, rs2), the advance signal (the pipeline lets the instruction leave E),
//                the result, the stall, and the weight-stream read port.
//   vdotk_item   one instruction (op, lane, rs1, rs2) plus what came back (result, stall cycles).
//   vdotk_seq    random but legal programs: SETK, BSTORE, SETP, WSTART then VDOTK/WPOP never past
//                the end of the stream, VDOTR, ACCRD. Random idle gaps let the FIFO fill or run dry.
//                D13 (22 Nov): SETS (SETP f7[5]) and ACCQ (ACCRD f7[6]) with legal scales
//                (mantissa 0 or [2^29, 2^30), shifts >= 1, including > 62 for a zero result).
//   directed     corner-data accumulation before bank fill/random; short ACCQ hazards after random.
//   vdotk_drv    acts like E: puts the instruction on the bus, holds it while stall = 1,
//                raises advance for exactly one cycle when stall = 0 (the commit). From the 2nd held
//                cycle on it puts GARBAGE on rs1/rs2, like the core: during a hold M gets bubbles and
//                the producer of a forwarded operand retires, so only the first E cycle is trustworthy.
//                (Without this a unit that forgets to latch its operands would still pass.)
//   vdotk_mon    watches the bus; every commit becomes an item sent to the scoreboard.
//   vdotk_sb     a reference model of the unit (same rules as firmware/include/sc_isa.h): replays
//                every committed instruction and checks ACCRD and WPOP results.
//   vdotk_cov    functional coverage: every op, every K, every lane, ops that stalled first.
//   memory model the weight stream reads mem_word(addr), a fixed hash of the address, so the
//                scoreboard knows every word without sharing state with the RTL.
`timescale 1ns / 1ps
`include "uvm_macros.svh"
import uvm_pkg::*;

function automatic logic [31:0] mem_word(logic [31:0] addr);
    logic [31:0] x = addr * 32'h9E3779B1 + 32'h7F4A7C15;
    x ^= x >> 15; x *= 32'h2C1B3C6D; x ^= x >> 12;
    return x;
endfunction

interface vdotk_if (input logic clk);
    logic rst, valid, advance, stall;
    logic [2:0] f3;
    logic [6:0] f7;
    logic [31:0] rs1, rs2, result, ws_addr, ws_data;
    assign ws_data = mem_word(ws_addr);
endinterface

typedef enum logic [2:0] {SETK = 0, SETP = 1, VDOTR = 2, ACCRD = 3, VDOTK = 4, WPOP = 5, BSTORE = 6, WSTART = 7} op_e;

class vdotk_item extends uvm_sequence_item;
    rand op_e op;
    rand logic [6:0] f7;
    rand logic [31:0] rs1, rs2;
    rand int unsigned gap;            // idle cycles before this instruction
    logic [31:0] result;
    int stalled;                      // cycles held before the commit
    constraint c_gap { gap dist {0 := 6, [1:3] := 3, [4:20] := 1}; }
    `uvm_object_utils(vdotk_item)
    function new(string name = "vdotk_item"); super.new(name); endfunction
    function string convert2string();
        return $sformatf("%s f7=%0h rs1=%08h rs2=%08h -> %08h (stalled %0d)", op.name(), f7, rs1, rs2, result, stalled);
    endfunction
endclass

// Legal random program: the sequence tracks how many stream words are left so VDOTK/WPOP never
// wait forever (a real program never pops past the end of the matrix it started).
class vdotk_seq extends uvm_sequence #(vdotk_item);
    int n = 2000;
    function logic [31:0] rand_mant();      // a scale mantissa: 0 (10%) or in [2^29, 2^30)
        return ($urandom_range(0, 9) == 0) ? 32'd0 : (32'h2000_0000 | ($urandom & 32'h1FFF_FFFF));
    endfunction
    `uvm_object_utils(vdotk_seq)
    function new(string name = "vdotk_seq"); super.new(name); endfunction
    // Shared by the directed sequences: gap zero keeps dependent instructions adjacent in E.
    task send(op_e op, logic [6:0] f7, logic [31:0] rs1 = 0, logic [31:0] rs2 = 0);
        vdotk_item it = vdotk_item::type_id::create("directed");
        start_item(it);
        it.op = op; it.f7 = f7; it.rs1 = rs1; it.rs2 = rs2; it.gap = 0;
        finish_item(it);
    endtask
    task body();
        int left = 0;
        bit fresh = 0;                     // previous op was WSTART: the FIFO is empty right now
        vdotk_item it;
        for (int i = 0; i < n; i++) begin
            it = vdotk_item::type_id::create("it");
            start_item(it);
            if (i < 3) $display("PROG seq randomize item %0d @%0t", i, $time);
            if (fresh) begin               // hit the empty-FIFO stall with both VDOTK and WPOP
                if (!it.randomize() with { op inside {VDOTK, WPOP}; gap == 0; })
                    `uvm_fatal("SEQ", "randomize")
            end else if (left == 0) begin
                if (!it.randomize() with { op dist {SETK := 2, SETP := 2, VDOTR := 3, ACCRD := 4, BSTORE := 3, WSTART := 2};
                                           op == SETK -> rs1 inside {[0:9]};
                                           op == WSTART -> rs2 inside {[1:64]};
                                           op == SETP -> rs1 < 256; op == BSTORE -> rs1 < 256; })
                    `uvm_fatal("SEQ", "randomize")
            end else begin
                if (!it.randomize() with { op dist {VDOTK := 8, WPOP := 2, ACCRD := 3, SETP := 1, VDOTR := 1, BSTORE := 1};
                                           op == SETP -> rs1 < 256; op == BSTORE -> rs1 < 256; })
                    `uvm_fatal("SEQ", "randomize")
            end
            if (it.op == WSTART) begin it.rs1 = {it.rs1[31:2], 2'b00}; left = it.rs2; end
            if (it.op == VDOTK || it.op == WPOP) left--;
            fresh = (it.op == WSTART);
            // legal f7 for the ops whose f7 bits mean something (lane in [2:0]; [6] / [5] select variants)
            if (it.op == SETP) begin
                int r = $urandom_range(0, 9);
                it.f7 = (r < 4) ? 7'(it.f7[2:0]) : (r < 6) ? 7'd64 : 7'(32 + it.f7[2:0]);   // SETP / SETPA / SETS
                if (it.f7[5]) begin it.rs1 = rand_mant(); it.rs2 = $urandom_range(21, 52); end
            end
            if (it.op == ACCRD) begin
                it.f7 = $urandom_range(0, 1) ? 7'(64 + it.f7[2:0]) : 7'(it.f7[2:0]);         // ACCQ / ACCRD
                if (it.f7[6]) begin
                    it.rs1 = rand_mant();
                    // Lane exponents are >= 21 (corner prelude, then SETS): never sh < 1.
                    // One in eight readouts uses sh > 62 in either normalisation branch.
                    it.rs2 = ($urandom_range(0, 7) == 0) ? $urandom_range(84, 110) : $urandom_range(22, 51);
                end
            end
            finish_item(it);
        end
    endtask
endclass

// A legal dot4 adds at most 65536, so reaching INT32 corners needs about 33K VDOTRs.
// Run before the normal bank fill, which restores random bank contents for the random program.
class vdotk_corner_seq extends vdotk_seq;
    `uvm_object_utils(vdotk_corner_seq)
    function new(string name = "vdotk_corner_seq"); super.new(name); endfunction
    task body();
        send(SETK, 0, 8);
        for (int l = 0; l < 8; l++) begin
            send(ACCRD, 7'(l));
            for (int a = 0; a < 256; a++)
                send(BSTORE, 7'(l), a, (l & 1) ? 32'h7F7F_7F7F : 32'h8080_8080);
            send(SETP, 7'(32 + l), (l & 2) ? 32'h3FFF_FFFF : 32'h2000_0000, 21);
        end
        send(SETP, 7'd64, 0);
        for (int sign_pass = 0; sign_pass < 2; sign_pass++) begin
            // Even lanes: 32767 * 65536 = 0x7fff0000 (INT32_MAX - 65535).
            // Odd lanes: 33026 * -65024 = 0x80000400 (INT32_MIN + 1024).
            repeat ((sign_pass == 0) ? 32767 : 259) send(VDOTR, 0, 32'h8080_8080);
            for (int l = sign_pass; l < 8; l += 2) begin
                int sh, re;
                case (l / 2)
                    0: sh = 1;
                    1: sh = 31;
                    2: sh = 32;
                    3: sh = 62;
                endcase
                re = sh + ((l & 2) ? 42 : 41) - 21;
                send(ACCRD, 7'(64 + l), (l & 2) ? 32'h3FFF_FFFF : 32'h2000_0000, re);
            end
        end
        // Leave accumulators/pointers/K as at reset; se = 21 also makes zero-mantissa shifts legal.
        for (int l = 0; l < 8; l++) send(ACCRD, 7'(l));
        send(SETP, 7'd64, 0);
        send(SETK, 0, 1);
    endtask
endclass

// Short post-random checks: every lane, both product branches at each shift range,
// zero in either scale operand, and SETS/VDOTR/ACCQ dependencies with no idle gaps.
class vdotk_accq_seq extends vdotk_seq;
    `uvm_object_utils(vdotk_accq_seq)
    function new(string name = "vdotk_accq_seq"); super.new(name); endfunction
    task body();
        send(SETK, 0, 8);
        for (int l = 0; l < 8; l++) begin
            send(ACCRD, 7'(l));
            send(BSTORE, 7'(l), 0, (l & 1) ? 32'h7F7F_7F7F : 32'h8080_8080);
        end
        send(SETP, 7'd64, 0);
        send(VDOTR, 0, 32'h8080_8080);       // all lanes nonzero before the first ACCQ
        send(SETK, 0, 1);                   // lanes 1..7 are deliberately >= K on readout
        for (int l = 0; l < 8; l++) begin
            logic [31:0] sm, rm;
            int re;
            sm = (l & 1) ? 32'h3FFF_FFFF : 32'h2000_0000;
            rm = sm;
            case (l)
                0: re = 21;                // sh = 1,  p < 2^59
                1: re = 52;                // sh = 31, p >= 2^59
                2: re = 52;                // sh = 32, p < 2^59
                3: re = 83;                // sh = 62, p >= 2^59
                4: re = 83;                // sh = 63, p < 2^59: result 0
                5: re = 84;                // sh = 63, p >= 2^59: result 0
                6: begin sm = 0; re = 21; end
                7: begin rm = 0; re = 21; end
            endcase
            send(SETP, 7'(32 + l), sm, 21);
            send(ACCRD, 7'(64 + l), rm, re); // SETS -> ACCQ
            send(ACCRD, 7'(64 + l), rm, re); // ACCQ -> ACCQ: cleared accumulator, result 0
        end
        send(SETP, 0, 0);                  // lane 0 points at its known corner bank word
        send(VDOTR, 0, 32'h8080_8080);
        send(ACCRD, 7'd64, 32'h2000_0000, 21); // VDOTR -> ACCQ, nonzero result
        send(ACCRD, 7'd64, 32'h2000_0000, 21); // second returns 0 with a nonzero scale
    endtask
endclass

// Bank fill: the banks are RAM (no reset), so like real firmware we write every word before the
// random program uses them. 8 lanes x 256 words of BSTORE, random data, no gaps.
class vdotk_fill_seq extends uvm_sequence #(vdotk_item);
    `uvm_object_utils(vdotk_fill_seq)
    function new(string name = "vdotk_fill_seq"); super.new(name); endfunction
    task body();
        vdotk_item it;
        for (int l = 0; l < 8; l++)
            for (int a = 0; a < 256; a++) begin
                it = vdotk_item::type_id::create("fill");
                start_item(it);
                it.op = BSTORE; it.f7 = 7'(l); it.rs1 = a; it.rs2 = $urandom; it.gap = 0;
                finish_item(it);
            end
    endtask
endclass

class vdotk_drv extends uvm_driver #(vdotk_item);
    virtual vdotk_if vif;
    `uvm_component_utils(vdotk_drv)
    function new(string name, uvm_component parent); super.new(name, parent); endfunction
    function void build_phase(uvm_phase phase);
        if (!uvm_config_db#(virtual vdotk_if)::get(this, "", "vif", vif)) `uvm_fatal("DRV", "no vif")
    endfunction
    task run_phase(uvm_phase phase);
        vif.valid <= 0; vif.advance <= 0;
        @(negedge vif.clk iff !vif.rst);
        $display("PROG drv out of reset @%0t", $time);
        for (int cnt = 0; ; cnt++) begin
            vdotk_item it;
            seq_item_port.get_next_item(it);
            if (cnt % 1000 == 0) $display("PROG drv item %0d %s @%0t", cnt, it.op.name(), $time);
            repeat (it.gap) begin vif.valid <= 0; vif.advance <= 1; @(negedge vif.clk); end
            vif.valid <= 1; vif.f3 <= it.op; vif.f7 <= it.f7; vif.rs1 <= it.rs1; vif.rs2 <= it.rs2;
            #1;
            while (vif.stall) begin
                vif.advance <= 0; @(negedge vif.clk);
                vif.rs1 <= $urandom; vif.rs2 <= $urandom;   // stale forwarded operands (see header)
                #1;
            end
            vif.advance <= 1;                         // commit on the next rising edge
            @(negedge vif.clk);
            // The instruction has left E: take it off the bus. (Bug found by seed 5 + WaveQ, 15 Nov:
            // without this the last WPOP stayed valid and "committed" again every cycle after the
            // sequence ended.) A next item's valid <= 1 in this same time step overrides this.
            vif.valid <= 0;
            seq_item_port.item_done();
        end
    endtask
endclass

class vdotk_mon extends uvm_monitor;
    virtual vdotk_if vif;
    uvm_analysis_port #(vdotk_item) ap;
    `uvm_component_utils(vdotk_mon)
    function new(string name, uvm_component parent); super.new(name, parent); ap = new("ap", this); endfunction
    function void build_phase(uvm_phase phase);
        if (!uvm_config_db#(virtual vdotk_if)::get(this, "", "vif", vif)) `uvm_fatal("MON", "no vif")
    endfunction
    task run_phase(uvm_phase phase);
        int held = 0;
        logic [31:0] r1, r2;                          // operands as seen in the FIRST E cycle
        forever begin
            @(posedge vif.clk);
            if (vif.rst || !vif.valid) begin held = 0; continue; end
            if (held == 0) begin r1 = vif.rs1; r2 = vif.rs2; end
            if (vif.stall) begin held++; continue; end
            if (vif.advance) begin                    // this edge is the commit
                vdotk_item it = vdotk_item::type_id::create("seen");
                it.op = op_e'(vif.f3); it.f7 = vif.f7; it.rs1 = r1; it.rs2 = r2;
                it.result = vif.result; it.stalled = held;
                ap.write(it);
                held = 0;
            end
        end
    endtask
endclass

// Reference model + checks. Same semantics as the C model in sc_isa.h.
class vdotk_sb extends uvm_subscriber #(vdotk_item);
    logic [31:0] bank [8][256];
    int ptr [8];
    int signed acc [8];
    logic [31:0] sm [8]; int signed se [8];   // lane scales (SETS)
    int k = 1;
    logic [31:0] ws_ptr; int ws_left;
    int checks = 0, errors = 0;
    `uvm_component_utils(vdotk_sb)
    function new(string name, uvm_component parent);
        super.new(name, parent);
        foreach (bank[i, j]) bank[i][j] = 0;
        foreach (ptr[i]) begin ptr[i] = 0; acc[i] = 0; sm[i] = 0; se[i] = 0; end
    endfunction
    // ACCQ reference = sc_isa.h sc_accq_() = ref/qref.c smul() + apply32() (the number system)
    function logic [31:0] accq(int l, logic [31:0] rm, int signed re);
        logic [63:0] p; logic [31:0] m; int signed e, sh; logic signed [64:0] x;
        p = 64'(sm[l]) * 64'(rm);
        if (p >= 64'h0800_0000_0000_0000) begin m = p[61:30]; e = se[l] + re - 30; end
        else begin m = p[60:29]; e = se[l] + re - 29; end
        sh = e - 12;
        if (m == 0 || sh > 62 || sh < 1) return 32'd0;
        x = 65'(acc[l]) * $signed({33'd0, m}) + (65'sd1 <<< (sh - 1));
        return 32'(x >>> sh);
    endfunction
    function int signed dot4(logic [31:0] a, logic [31:0] b);
        dot4 = 0;
        for (int i = 0; i < 4; i++) dot4 += $signed(a[8*i +: 8]) * $signed(b[8*i +: 8]);
    endfunction
    function void mac(logic [31:0] w);
        for (int l = 0; l < k; l++) begin acc[l] += dot4(w, bank[l][ptr[l]]); ptr[l] = (ptr[l] + 1) % 256; end
    endfunction
    function void chk(string what, logic [31:0] got, logic [31:0] want);
        checks++;
        if (got !== want) begin errors++; `uvm_error("SB", $sformatf("%s got %08h want %08h", what, got, want)) end
    endfunction
    function void write(vdotk_item t);
        int l = t.f7[2:0];
        case (t.op)
            SETK: k = (t.rs1 == 0) ? 1 : (t.rs1 > 8) ? 8 : t.rs1;
            SETP: if (t.f7[5]) begin sm[l] = t.rs1; se[l] = t.rs2; end
                  else if (t.f7[6]) foreach (ptr[i]) ptr[i] = t.rs1[7:0]; else ptr[l] = t.rs1[7:0];
            VDOTR: mac(t.rs1);
            ACCRD: begin
                if (t.f7[6]) chk($sformatf("ACCQ lane %0d", l), t.result, accq(l, t.rs1, t.rs2));
                else chk($sformatf("ACCRD lane %0d", l), t.result, acc[l]);
                acc[l] = 0;
            end
            VDOTK: begin mac(mem_word(ws_ptr)); ws_ptr += 4; ws_left--; end
            WPOP: begin chk("WPOP", t.result, mem_word(ws_ptr)); ws_ptr += 4; ws_left--; end
            BSTORE: bank[l][t.rs1[7:0]] = t.rs2;
            WSTART: begin ws_ptr = t.rs1; ws_left = t.rs2; end
        endcase
    endfunction
    function void report_phase(uvm_phase phase);
        `uvm_info("SB", $sformatf("scoreboard: %0d checks, %0d errors", checks, errors), UVM_NONE)
    endfunction
endclass

class vdotk_cov extends uvm_subscriber #(vdotk_item);
    // Kept small on purpose: a `with` filter on a cross whose bin is [1:$] made xsim expand ~2^31
    // values at time 0 and eat all memory (froze the laptop, 14/15 Nov). Use 1-bit / small ranges
    // and binsof() for ignore bins.
    op_e op; int lane, kk; bit st;
    bit [1:0] qv;                                  // 0 plain, 1 SETS, 2 ACCQ
    logic [31:0] sm [8]; int signed se [8];          // committed SETS state, as in the scoreboard
    int active_k = 1;
    bit [1:0] q_shift;                              // 0: 1..31, 1: 32..62, 2: >62
    bit q_p_ge59, q_zero_m, q_inactive;
    covergroup cg;
        cp_variant: coverpoint qv { bins plain = {0}; bins sets = {1}; bins accq = {2}; }
        cp_op: coverpoint op;
        cp_lane: coverpoint lane { bins l[] = {[0:7]}; }
        cp_k: coverpoint kk iff (op == SETK) { bins k[] = {[0:9]}; }
        cp_stall: coverpoint st { bins none = {0}; bins some = {1}; }
        x_op_stall: cross cp_op, cp_stall {
            ignore_bins never = binsof(cp_stall.some) && binsof(cp_op) intersect {SETK, SETP, VDOTR, BSTORE, WSTART}; }
        cp_accq_shift: coverpoint q_shift iff (qv == 2) {
            bins sh_1_31 = {0}; bins sh_32_62 = {1}; bins sh_gt62 = {2}; }
        cp_accq_product: coverpoint q_p_ge59 iff (qv == 2) {
            bins lt_2p59 = {0}; bins ge_2p59 = {1}; }
        cp_accq_mantissa: coverpoint q_zero_m iff (qv == 2) {
            bins nonzero = {0}; bins zero = {1}; }
        cp_accq_lane: coverpoint lane iff (qv == 2) { bins l[] = {[0:7]}; }
        cp_accq_inactive: coverpoint q_inactive iff (qv == 2) {
            bins below_k = {0}; bins at_or_above_k = {1}; }
        x_accq_shift_product: cross cp_accq_shift, cp_accq_product iff (qv == 2);
    endgroup
    `uvm_component_utils(vdotk_cov)
    function new(string name, uvm_component parent);
        super.new(name, parent);
        foreach (sm[i]) begin sm[i] = 0; se[i] = 0; end
        cg = new();
    endfunction
    function void write(vdotk_item t);
        logic [63:0] p; logic [31:0] m; int signed e, sh;
        op = t.op; lane = t.f7[2:0]; kk = (t.op == SETK) ? t.rs1 : 0; st = (t.stalled > 0);
        qv = (t.op == SETP && t.f7[5]) ? 2'd1 : (t.op == ACCRD && t.f7[6]) ? 2'd2 : 2'd0;
        if (t.op == SETK) active_k = (t.rs1 == 0) ? 1 : (t.rs1 > 8) ? 8 : t.rs1;
        if (qv == 1) begin sm[lane] = t.rs1; se[lane] = t.rs2; end
        if (qv == 2) begin
            // Same widths, normalisation branches and signed exponents as vdotk_sb::accq().
            p = 64'(sm[lane]) * 64'(t.rs1);
            if (p >= 64'h0800_0000_0000_0000) begin m = p[61:30]; e = se[lane] + $signed(t.rs2) - 30; end
            else begin m = p[60:29]; e = se[lane] + $signed(t.rs2) - 29; end
            sh = e - 12;
            q_shift = (sh >= 1 && sh <= 31) ? 2'd0 : (sh >= 32 && sh <= 62) ? 2'd1 : (sh > 62) ? 2'd2 : 2'd3;
            q_p_ge59 = (p >= 64'h0800_0000_0000_0000);
            q_zero_m = (m == 0);
            q_inactive = (lane >= active_k);
        end
        cg.sample();
    endfunction
    function void report_phase(uvm_phase phase);
        `uvm_info("COV", $sformatf("coverage: ops %0.1f%%, lanes %0.1f%%, K %0.1f%%, op x stall %0.1f%%, total %0.1f%%",
                  cg.cp_op.get_coverage(), cg.cp_lane.get_coverage(), cg.cp_k.get_coverage(),
                  cg.x_op_stall.get_coverage(), cg.get_coverage()), UVM_NONE)
        `uvm_info("COV", $sformatf("ACCQ coverage: shifts %0.1f%%, products %0.1f%%, zero mantissa %0.1f%%, lanes %0.1f%%, lane vs K %0.1f%%, shift x product %0.1f%%",
                  cg.cp_accq_shift.get_coverage(), cg.cp_accq_product.get_coverage(), cg.cp_accq_mantissa.get_coverage(),
                  cg.cp_accq_lane.get_coverage(), cg.cp_accq_inactive.get_coverage(), cg.x_accq_shift_product.get_coverage()), UVM_NONE)
    endfunction
endclass

class vdotk_env extends uvm_env;
    uvm_sequencer #(vdotk_item) sqr; vdotk_drv drv; vdotk_mon mon; vdotk_sb sb; vdotk_cov cov;
    `uvm_component_utils(vdotk_env)
    function new(string name, uvm_component parent); super.new(name, parent); endfunction
    function void build_phase(uvm_phase phase);
        sqr = uvm_sequencer#(vdotk_item)::type_id::create("sqr", this);
        drv = vdotk_drv::type_id::create("drv", this);
        mon = vdotk_mon::type_id::create("mon", this);
        sb = vdotk_sb::type_id::create("sb", this);
        cov = vdotk_cov::type_id::create("cov", this);
    endfunction
    function void connect_phase(uvm_phase phase);
        drv.seq_item_port.connect(sqr.seq_item_export);
        mon.ap.connect(sb.analysis_export);
        mon.ap.connect(cov.analysis_export);
    endfunction
endclass

class vdotk_test extends uvm_test;
    vdotk_env env;
    `uvm_component_utils(vdotk_test)
    function new(string name, uvm_component parent); super.new(name, parent); endfunction
    function void build_phase(uvm_phase phase); env = vdotk_env::type_id::create("env", this); endfunction
    task run_phase(uvm_phase phase);
        vdotk_fill_seq f = vdotk_fill_seq::type_id::create("f");
        vdotk_seq s = vdotk_seq::type_id::create("s");
        vdotk_corner_seq c = vdotk_corner_seq::type_id::create("c");
        vdotk_accq_seq q = vdotk_accq_seq::type_id::create("q");
        int n;
        if ($value$plusargs("n_%d", n)) s.n = n;
        phase.raise_objection(this);
        c.start(env.sqr);
        f.start(env.sqr);
        s.start(env.sqr);
        q.start(env.sqr);
        repeat (5) @(posedge env.drv.vif.clk);
        phase.drop_objection(this);
    endtask
    function void report_phase(uvm_phase phase);
        uvm_report_server rs = uvm_report_server::get_server();
        int errs = rs.get_severity_count(UVM_ERROR) + rs.get_severity_count(UVM_FATAL);
        $display("UVM %s: %0d UVM errors, %0d unit SVA failures",
                 (errs == 0 && tb_vdotk_uvm.dut.sva_fail == 0) ? "PASS" : "FAIL", errs, tb_vdotk_uvm.dut.sva_fail);
    endfunction
endclass

module tb_vdotk_uvm;
    logic clk = 0;
    always #5 clk = ~clk;
    vdotk_if vif(clk);
    sc_vdotk #(.LANES(8)) dut(
        .clk(clk), .rst(vif.rst), .valid(vif.valid), .f3(vif.f3), .f7(vif.f7), .rs1(vif.rs1), .rs2(vif.rs2),
        .advance(vif.advance), .result(vif.result), .stall(vif.stall), .ws_addr(vif.ws_addr), .ws_data(vif.ws_data));
    initial begin
        vif.rst = 1; repeat (3) @(posedge clk); vif.rst = 0;
    end
    // Windowed VCD for WaveQ: +vcd_T dumps T-200 ns .. T+200 ns only (T in ns), into uvm.vcd.
    initial begin
        int t;
        if ($value$plusargs("vcd_%d", t)) begin
            #((t > 200 ? t - 200 : 0) * 1ns);
            $dumpfile("uvm.vcd"); $dumpvars(0, tb_vdotk_uvm);
            #400ns; $dumpoff;
        end
    end
    // Watchdog: a legal program never needs this long; a hang ends the run instead of spinning.
    initial begin
        #50ms;
        $display("UVM FAIL: watchdog (50 ms simulated), the unit or the driver is stuck");
        $finish;
    end
    initial begin
        uvm_config_db#(virtual vdotk_if)::set(null, "*", "vif", vif);
        run_test("vdotk_test");
    end
endmodule
