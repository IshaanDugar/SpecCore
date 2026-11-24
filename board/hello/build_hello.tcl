# Vivado batch build of the board "hello" app: proves the flow laptop build -> xmutil loadapp -> register access.
# Contents (not an architecture choice): Zynq UltraScale+ PS, M_AXI_HPM0_FPD at 100 MHz ->
#   8 KB AXI BRAM (read/write test)  +  AXI GPIO, all inputs, tied to the constant ID 0x5C0E0001.
# Run through tools/board_build.py (memory/time guard). Output dir = first tcl argument.
set out [file normalize [lindex $argv 0]]
file mkdir $out
create_project hello $out/proj -part xck26-sfvc784-2LV-c -force
set_param general.maxThreads 2
create_bd_design hello

# HPM0 width 128: the KV260 boot firmware sets the PS-side width register (AFI_FS) to 128 bit and
# `xmutil loadapp` only loads the PL, so a 32-bit build saw only every 4th word (16 Nov, measured).
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] $ps

set bram [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl bram_ctrl]
set_property -dict [list CONFIG.SINGLE_PORT_BRAM {1} CONFIG.DATA_WIDTH {32}] $bram
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio id_gpio]
set_property -dict [list CONFIG.C_ALL_INPUTS {1} CONFIG.C_GPIO_WIDTH {32}] $gpio
set idc [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant id_const]
set_property -dict [list CONFIG.CONST_WIDTH {32} CONFIG.CONST_VAL {0x5C0E0001}] $idc
connect_bd_net [get_bd_pins id_const/dout] [get_bd_pins id_gpio/gpio_io_i]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master /ps/M_AXI_HPM0_FPD Clk_master Auto Clk_slave Auto Clk_xbar Auto Slave /bram_ctrl/S_AXI intc_ip Auto ddr_seg Auto} [get_bd_intf_pins bram_ctrl/S_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master /ps/M_AXI_HPM0_FPD Clk_master Auto Clk_slave Auto Clk_xbar Auto Slave /id_gpio/S_AXI intc_ip Auto ddr_seg Auto} [get_bd_intf_pins id_gpio/S_AXI]
apply_bd_automation -rule xilinx.com:bd_rule:bram_cntlr -config {BRAM "Auto"} [get_bd_intf_pins bram_ctrl/BRAM_PORTA]

assign_bd_address
set amap [open $out/address_map.txt w]
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    puts $amap "[get_property NAME $s] [format 0x%08x [get_property OFFSET $s]] [format 0x%x [get_property RANGE $s]]"
}
close $amap
validate_bd_design
save_bd_design
# Global synthesis of the block-design IPs: no separate out-of-context IP runs (each is its own ~1 GB
# Vivado process; with -jobs 2 they hit the 4 GB guard cap on 16 Nov). One run at a time.
set_property synth_checkpoint_mode None [get_files hello.bd]

make_wrapper -files [get_files hello.bd] -top
add_files -norecurse [glob $out/proj/hello.gen/sources_1/bd/hello/hdl/hello_wrapper.v]
set_property top hello_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Synthesis + implementation in THIS process (no launch_runs child: parent + child Vivado together
# hit the 4 GB guard cap twice on 16 Nov).
generate_target all [get_files hello.bd]
synth_design -top hello_wrapper -part xck26-sfvc784-2LV-c
opt_design
place_design
route_design
write_bitstream -force $out/hello.bit
report_utilization -file $out/utilization.txt
report_timing_summary -file $out/timing.txt
puts "HELLO BUILD DONE"
