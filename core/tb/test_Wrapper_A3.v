`timescale 1ns / 1ps
// Testbench for Demo_A3.asm (CG3207 Asst 4). Module name test_Wrapper; set it as the simulation top.
// Part 1: the self-check (stages 1-7 from Asst 2, stage 8 = mul/div/divu) must end with LED = 0xFF.
//         On a failure the program shows BAD0000N on the 7-seg and N on the LEDs.
// Part 2: the DIP calculator. hi = DIP[15:8], lo = DIP[7:0];
//         7-seg = { hi*lo [15:0], (hi divu lo) [7:0], (signed hi div signed lo) [7:0] }, lower half 0 if lo = 0.

module test_Wrapper #(
    parameter N_LEDs_OUT = 8,
    parameter N_DIPs = 16,
    parameter N_PBs = 3
)();

reg [N_DIPs-1:0] DIP = 0;
reg [N_PBs-1:0] PB = 0;
wire [N_LEDs_OUT-1:0] LED_OUT;
wire [6:0] LED_PC;
wire [31:0] SEVENSEGHEX;
wire [7:0] UART_TX;
reg UART_TX_ready = 0;
wire UART_TX_valid;
reg [7:0] UART_RX = 0;
reg UART_RX_valid = 0;
wire UART_RX_ack;
wire OLED_Write;
wire [6:0] OLED_Col;
wire [5:0] OLED_Row;
wire [23:0] OLED_Data;
reg [31:0] ACCEL_Data = 0;
wire ACCEL_DReady;
reg RESET = 0;
reg CLK = 0;

integer pass_count = 0;
integer fail_count = 0;
integer i;
reg [15:0] dip_list [0:7];
reg [7:0] hi, lo;
reg signed [7:0] shi, slo;
reg [15:0] prod;
reg [7:0] qu, qs;
reg [31:0] expected;

Wrapper dut(DIP, PB, LED_OUT, LED_PC, SEVENSEGHEX, UART_TX, UART_TX_ready, UART_TX_valid, UART_RX, UART_RX_valid, UART_RX_ack, OLED_Write, OLED_Col, OLED_Row, OLED_Data, ACCEL_Data, ACCEL_DReady, RESET, CLK);

initial begin
dip_list[0] = 16'h0C05; //  12 x 5,   12 / 5   (both positive)
dip_list[1] = 16'hFF02; // 255 x 2: unsigned 255 / 2 = 127, signed -1 / 2 = 0
dip_list[2] = 16'h8003; // 128 / 3 = 42 unsigned, -128 / 3 = -42 signed
dip_list[3] = 16'h07FD; //  7 / 253 = 0 unsigned, 7 / -3 = -2 signed
dip_list[4] = 16'h80FF; // -128 / -1 = 128 signed
dip_list[5] = 16'h2A00; // lo = 0: divisions skipped (beq taken)
dip_list[6] = 16'hFFFF; // 255 x 255 = 65025, 255 / 255 = 1, -1 / -1 = 1
dip_list[7] = 16'h0101;

RESET = 1; #10;
RESET = 0;

// part 1: wait for the self-check to finish (LED becomes non-zero)
wait (LED_OUT != 0 || $time > 200000);
#1;
if (LED_OUT === 8'hFF) begin
    $display("[%0t] PASS self-check: all 8 stages passed (LED = FF)", $time);
    pass_count = pass_count + 1;
end else if (SEVENSEGHEX[31:12] === 20'hBAD00) begin
    $display("[%0t] FAIL self-check: stage %0d failed (7-seg = %h)", $time, SEVENSEGHEX[3:0], SEVENSEGHEX);
    fail_count = fail_count + 1;
end else begin
    $display("[%0t] FAIL self-check: LED = %h, 7-seg = %h", $time, LED_OUT, SEVENSEGHEX);
    fail_count = fail_count + 1;
end

// part 2: DIP calculator
for (i = 0; i < 8; i = i + 1) begin
    DIP = dip_list[i];
    #8000; // nop-padded loop is about 4x slower
    hi = DIP[15:8];
    lo = DIP[7:0];
    shi = hi;
    slo = lo;
    prod = hi * lo;
    if (lo == 0) expected = {prod, 16'h0000};
    else begin
        qu = hi / lo;
        qs = shi / slo;  // Verilog signed division also rounds toward zero
        expected = {prod, qu, qs};
    end
    if (SEVENSEGHEX === expected) begin
        $display("[%0t] PASS DIP = %h, 7-seg = %h", $time, DIP, SEVENSEGHEX);
        pass_count = pass_count + 1;
    end
    else begin
        $display("[%0t] FAIL DIP = %h, 7-seg = %h, expected %h", $time, DIP, SEVENSEGHEX, expected);
        fail_count = fail_count + 1;
    end
end

if (LED_OUT !== 8'hFF) begin
    $display("[%0t] FAIL LED changed to %h", $time, LED_OUT);
    fail_count = fail_count + 1;
end

$display("Passed: %0d, Failed: %0d", pass_count, fail_count);
if (fail_count == 0) $display("*** ALL TESTS PASSED! ***");
else $display("*** %0d TESTS FAILED ***", fail_count);
$finish;
end

always begin
    #5 CLK = ~CLK;
end

endmodule
