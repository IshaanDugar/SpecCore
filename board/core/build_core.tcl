# Vivado batch build of the board "core" app (D16 option D): the SpecCore RV core + LLM unit on the KV260.
#   PS M_AXI_HPM0_FPD (128 bit, see build_hello.tcl) -> SmartConnect -> sc_board AXI-Lite (2 MB @ 0xA000_0000)
#   sc_board m_axi_ws + m_axi_ld (read-only AXI4, 128 bit) -> AXI Interconnect -> S_AXI_HP0_FPD (DDR)
#   PL clock 0 = 50 MHz for everything; the core runs on a gated copy (BUFGCE inside sc_board).
# Run through tools/board_build.py (memory/time guard). Output dir = first tcl argument.
set out [file normalize [lindex $argv 0]]
set root [file normalize [file dirname [info script]]/../..]
file mkdir $out
create_project core $out/proj -part xck26-sfvc784-2LV-c -force
set_param general.maxThreads 2
foreach f [glob $root/core/rtl/*.v] { add_files -norecurse $f }
add_files -norecurse [glob $root/ext/rtl/*.sv]
add_files -norecurse $root/board/core/sc_board.v
set_property include_dirs [list $root/ext/rtl] [current_fileset]
update_compile_order -fileset sources_1
create_bd_design core

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {50}] $ps

set sc [create_bd_cell -type module -reference sc_board scb]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master /ps/M_AXI_HPM0_FPD Clk_master Auto Clk_slave Auto Clk_xbar Auto Slave /scb/s_axi intc_ip Auto ddr_seg Auto} [get_bd_intf_pins scb/s_axi]

# two read masters -> HP0 through a plain AXI Interconnect wired by hand (as in build_dma.tcl)
set ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect hp_ic]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $ic
connect_bd_intf_net [get_bd_intf_pins scb/m_axi_ws] [get_bd_intf_pins hp_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins scb/m_axi_ld] [get_bd_intf_pins hp_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins hp_ic/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
set clknet [get_bd_nets -of_objects [get_bd_pins ps/pl_clk0]]
set rstc [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}]
set prst [get_bd_nets -of_objects [get_bd_pins $rstc/peripheral_aresetn]]
set irst [get_bd_pins $rstc/interconnect_aresetn]
foreach p {scb/clk hp_ic/ACLK hp_ic/S00_ACLK hp_ic/S01_ACLK hp_ic/M00_ACLK ps/saxihp0_fpd_aclk} {
    set pin [get_bd_pins $p]
    if {[llength [get_bd_nets -quiet -of_objects $pin]] == 0} { connect_bd_net -net $clknet $pin }
}
foreach p {scb/aresetn hp_ic/S00_ARESETN hp_ic/S01_ARESETN hp_ic/M00_ARESETN} {
    set pin [get_bd_pins $p]
    if {[llength [get_bd_nets -quiet -of_objects $pin]] == 0} { connect_bd_net -net $prst $pin }
}
if {[llength [get_bd_nets -quiet -of_objects $irst]] == 0} { connect_bd_net $irst [get_bd_pins hp_ic/ARESETN] } else {
    connect_bd_net -net [get_bd_nets -of_objects $irst] [get_bd_pins hp_ic/ARESETN] }

assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ *scb*}]
set_property offset 0xA0000000 $seg
set_property range 2M $seg
set amap [open $out/address_map.txt w]
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    puts $amap "[get_property NAME $s] [format 0x%08x [get_property OFFSET $s]] [format 0x%x [get_property RANGE $s]]"
}
close $amap
set dmap [open $out/master_map.txt w]
foreach sp [get_bd_addr_spaces scb/*] {
    foreach s [get_bd_addr_segs -quiet -of_objects $sp] {
        set off [get_property OFFSET $s]
        if {$off eq ""} { puts $dmap "[get_property NAME $sp] [get_property NAME $s] excluded"; continue }
        puts $dmap "[get_property NAME $sp] [get_property NAME $s] [format 0x%x $off] [format 0x%x [get_property RANGE $s]]"
    }
}
close $dmap
validate_bd_design
save_bd_design
set_property synth_checkpoint_mode None [get_files core.bd]

make_wrapper -files [get_files core.bd] -top
add_files -norecurse [glob $out/proj/core.gen/sources_1/bd/core/hdl/core_wrapper.v]
set_property top core_wrapper [current_fileset]
update_compile_order -fileset sources_1

generate_target all [get_files core.bd]
synth_design -top core_wrapper -part xck26-sfvc784-2LV-c -include_dirs $root/ext/rtl
opt_design
place_design
route_design
write_bitstream -force $out/core.bit
report_utilization -file $out/utilization.txt
report_utilization -hierarchical -hierarchical_depth 3 -file $out/util_hier.txt
report_timing_summary -max_paths 5 -file $out/timing.txt
report_clock_utilization -file $out/clocks.txt
puts "CORE BUILD DONE"
