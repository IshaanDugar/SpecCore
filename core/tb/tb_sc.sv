`timescale 1ns / 1ps
// SpecCore system testbench (SystemVerilog). Runs whatever AA_IROM.mem / AA_DMEM.mem hold
// (built by `python tools/sc.py fw`). Prints every UART byte the program sends.
// Program end = UART byte 0x04; exit code = SEVENSEG value.
//
// Plusargs (no "=" in them: cmd.exe splits on it):
//   +maxcyc_N   stop after N cycles (default 5,000,000)
//   +bp         branch predictor on (DIP[15])
//   +vcd        dump the core (Wrapper in GATE mode, all levels) to sim.vcd, for WaveQ
//
// SVA checks on the core run every cycle; the count is printed at the end ("[tb] SVA ...").
// Each property is one rule the pipeline must always obey.
// GATE disables RTL-only checks; UART, exit status and cycle counting use the Wrapper interface.
module tb_sc;
	logic CLK = 1'b0;
	logic RESET = 1'b1;
	logic [15:0] DIP = '0;
	logic [7:0] LED_OUT;
	logic [6:0] LED_PC;
	logic [31:0] SEVENSEGHEX;
	logic [7:0] UART_TX;
	logic UART_TX_valid, UART_RX_ack, OLED_Write;
	logic [6:0] OLED_Col;
	logic [5:0] OLED_Row;
	logic [23:0] OLED_Data;

	Wrapper dut(
		.DIP(DIP), .PB(3'd0), .LED_OUT(LED_OUT), .LED_PC(LED_PC), .SEVENSEGHEX(SEVENSEGHEX),
		.UART_TX(UART_TX), .UART_TX_ready(1'b1), .UART_TX_valid(UART_TX_valid),
		.UART_RX(8'd0), .UART_RX_valid(1'b0), .UART_RX_ack(UART_RX_ack),
		.OLED_Write(OLED_Write), .OLED_Col(OLED_Col), .OLED_Row(OLED_Row), .OLED_Data(OLED_Data),
		.ACCEL_Data(32'd0), .ACCEL_DReady(1'b0), .RESET(RESET), .CLK(CLK)
	);

	always #5 CLK = ~CLK;   // 100 MHz

	int cycles = 0;
	int max_cycles;
`ifndef GATE
	int sva_fail = 0;
`endif

	initial begin
		if (!$value$plusargs("maxcyc_%d", max_cycles)) max_cycles = 5_000_000;
		if ($test$plusargs("bp")) DIP[15] = 1'b1;
		if ($test$plusargs("vcd")) begin
			$dumpfile("sim.vcd");
`ifndef GATE
			$dumpvars(0, tb_sc.dut.RV1);
`else
			// Observe the testbench interface without dumping the large behavioral memories.
			$dumpvars(1, tb_sc);
`endif
		end
		$display("[tb] start, branch predictor %s, max %0d cycles", DIP[15] ? "ON" : "OFF", max_cycles);
`ifndef GATE
		#22 RESET = 1'b0;
`else
		$display("[tb] SVA checks disabled (GATE)");
		// glbl holds GSR for 100 ns. Apply two reset clocks after its release.
		#122 RESET = 1'b0;
`endif
	end

`ifndef GATE
	// ---------------- SVA: rules the core must obey ----------------
	// Short names for signals inside the core.
	wire [31:0] pc       = dut.RV1.PC_F;
	wire [3:0]  be       = dut.MemWrite_out;
	wire [31:0] addr     = dut.ALUResult;
	wire        busy     = dut.RV1.Busy_E;
	wire [31:0] pc_e     = dut.RV1.PC_E;
	wire        regwr_w  = dut.RV1.RegWrite_W;
	wire [4:0]  rd_w     = dut.RV1.rd_W;
	wire        load_w   = dut.RV1.MemtoReg_W;
	wire [31:0] result_w = dut.RV1.Result_W;
	wire        flushm   = dut.RV1.FlushM;

	default clocking cb @(posedge CLK); endclocking

	// 1. PC is never X (an X here means a branch/jump target came from an X register).
	a_pc_known: assert property (disable iff (RESET) !$isunknown(pc))
		else begin sva_fail++; $error("[sva] PC is X"); end

	// 2. Store byte enables are only sb (one byte), sh (low or high half), sw (all) or none.
	a_be_legal: assert property (disable iff (RESET) be inside {4'b0000, 4'b0001, 4'b0010, 4'b0100, 4'b1000, 4'b0011, 4'b1100, 4'b1111})
		else begin sva_fail++; $error("[sva] illegal byte enables %b", be); end

	// 3. Stores are aligned: sw to a multiple of 4, sh to a multiple of 2 (the core has no misaligned support).
	a_sw_aligned: assert property (disable iff (RESET) be == 4'b1111 |-> addr[1:0] == 2'b00)
		else begin sva_fail++; $error("[sva] misaligned sw at %h", addr); end
	a_sh_aligned: assert property (disable iff (RESET) (be == 4'b0011 || be == 4'b1100) |-> addr[0] == 1'b0)
		else begin sva_fail++; $error("[sva] misaligned sh at %h", addr); end

	// 4. While MCycle is busy, the instruction in E is held (same PC_E) and M gets a bubble.
	//    This is the exact path VDOTK's FIFO-empty stall will copy.
	a_busy_holds_e: assert property (disable iff (RESET) busy && $past(busy) |-> $stable(pc_e))
		else begin sva_fail++; $error("[sva] E changed while MCycle busy"); end
	a_busy_bubbles_m: assert property (disable iff (RESET) busy |-> flushm)
		else begin sva_fail++; $error("[sva] no bubble into M while busy"); end

	// 5. A computed (non-load) register write is never X. Loads are skipped: restoring a
	//    callee-saved register that was never written (X at power-up in sim) is legal.
	a_result_known: assert property (disable iff (RESET) regwr_w && rd_w != 0 && !load_w |-> !$isunknown(result_w))
		else begin sva_fail++; $error("[sva] X written to x%0d", rd_w); end
`endif


`ifndef GATE
	// ---------------- custom-op coverage counters (subsystem level) ----------------
	// Per committed custom-0 op (SETS / ACCQ are variants of SETP / ACCRD), K in use at each MAC,
	// stall cycles by cause, and custom ops that were forwarded a value from M or W.
	// 0 SETK 1 SETP 2 VDOTR 3 ACCRD 4 VDOTK 5 WPOP 6 BSTORE 7 WSTART 8 SETS 9 ACCQ
	longint unsigned cov_op [10];
	longint unsigned cov_k [9];
	longint unsigned cov_stall_fifo, cov_stall_accq;
	string cov_name [10] = '{"SETK","SETP","VDOTR","ACCRD","VDOTK","WPOP","BSTORE","WSTART","SETS","ACCQ"};
	wire       cov_commit = dut.RV1.LLM1.valid && dut.RV1.LLM1.advance;
	wire [2:0] cov_f3 = dut.RV1.LLM1.f3;
	wire [6:0] cov_f7 = dut.RV1.LLM1.f7;
	always @(posedge CLK) if (!RESET) begin
		if (cov_commit) begin
			if (cov_f3 == 3'd1 && cov_f7[5]) cov_op[8]++;
			else if (cov_f3 == 3'd3 && cov_f7[6]) cov_op[9]++;
			else cov_op[cov_f3]++;
			if (cov_f3 == 3'd2 || cov_f3 == 3'd4) cov_k[dut.RV1.LLM1.k_act]++;
		end
		if (dut.RV1.LLM1.stall) begin
			if (dut.RV1.LLM1.is_accq) cov_stall_accq++; else cov_stall_fifo++;
		end
	end
	task automatic cov_report();
		string s = "[cov] custom ops:";
		for (int i = 0; i < 10; i++) s = {s, $sformatf(" %s=%0d", cov_name[i], cov_op[i])};
		$display("%s", s);
		s = "[cov] MACs by K:";
		for (int i = 1; i <= 8; i++) s = {s, $sformatf(" K%0d=%0d", i, cov_k[i])};
		$display("%s", s);
		$display("[cov] stall cycles: ACCQ %0d, weight FIFO empty %0d", cov_stall_accq, cov_stall_fifo);
	endtask
`endif

	// ---------------- run control ----------------
	always @(posedge CLK) begin
		if (!RESET) begin
			cycles <= cycles + 1;
			if (UART_TX_valid) begin
				if (UART_TX == 8'h04) begin
`ifndef GATE
					$display("[tb] SVA failures: %0d", sva_fail + dut.RV1.LLM1.sva_fail);   // core rules + LLM unit rules
					cov_report();
`endif
					$display("[tb] EXIT code %0d (0x%08h) after %0d cycles", $signed(SEVENSEGHEX), SEVENSEGHEX, cycles);
					$finish;
				end
				else
					$write("%c", UART_TX);
			end
			if (cycles >= max_cycles) begin
				$display("");
`ifndef GATE
				$display("[tb] TIMEOUT after %0d cycles, PC = 0x%08h, SVA failures: %0d", cycles, dut.PC, sva_fail + dut.RV1.LLM1.sva_fail);
				cov_report();
`else
				$display("[tb] TIMEOUT after %0d cycles, PC = 0x%08h", cycles, dut.PC);
`endif
				$finish;
			end
		end
	end
endmodule
