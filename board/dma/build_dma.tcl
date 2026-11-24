# Vivado batch build of the board "dma" app (D16 option B): measures the real DDR -> PL -> DDR bandwidth
# that would feed the weight stream. Contents: Zynq UltraScale+ PS; AXI DMA in simple (register) mode,
# 128-bit memory-mapped and stream width, MM2S stream looped straight into S2MM; DMA masters reach DDR
# through S_AXI_HP0_FPD (128 bit); registers through M_AXI_HPM0_FPD (128 bit, see build_hello.tcl);
# AXI GPIO = constant ID 0x5C0E0002. PL clock 100 MHz (same as hello; 128 bit x 100 MHz = 1.6 GB/s per
# direction is the PL-side ceiling of this test).
set out [file normalize [lindex $argv 0]]
file mkdir $out
create_project dma $out/proj -part xck26-sfvc784-2LV-c -force
set_param general.maxThreads 2
create_bd_design dma

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] $ps

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma dma]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {26} CONFIG.c_addr_width {64} \
    CONFIG.c_m_axi_mm2s_data_width {128} CONFIG.c_m_axis_mm2s_tdata_width {128} CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_m_axi_s2mm_data_width {128} CONFIG.c_s2mm_burst_size {64}] $dma
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins dma/S_AXIS_S2MM]

set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio id_gpio]
set_property -dict [list CONFIG.C_ALL_INPUTS {1} CONFIG.C_GPIO_WIDTH {32}] $gpio
set idc [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant id_const]
set_property -dict [list CONFIG.CONST_WIDTH {32} CONFIG.CONST_VAL {0x5C0E0002}] $idc
connect_bd_net [get_bd_pins id_const/dout] [get_bd_pins id_gpio/gpio_io_i]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master /ps/M_AXI_HPM0_FPD Clk_master Auto Clk_slave Auto Clk_xbar Auto Slave /dma/S_AXI_LITE intc_ip Auto ddr_seg Auto} [get_bd_intf_pins dma/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master /ps/M_AXI_HPM0_FPD Clk_master Auto Clk_slave Auto Clk_xbar Auto Slave /id_gpio/S_AXI intc_ip Auto ddr_seg Auto} [get_bd_intf_pins id_gpio/S_AXI]
# DMA masters -> HP0 through a plain AXI Interconnect, wired by hand (the automation's second
# SmartConnect came out of global synthesis as an empty black box, 22 Nov build #2).
set ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect hp_ic]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $ic
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins hp_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins hp_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins hp_ic/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
set clknet [get_bd_nets -of_objects [get_bd_pins ps/pl_clk0]]
set rstc [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}]
set prst [get_bd_nets -of_objects [get_bd_pins $rstc/peripheral_aresetn]]
set irst [get_bd_pins $rstc/interconnect_aresetn]
foreach p {dma/m_axi_mm2s_aclk dma/m_axi_s2mm_aclk hp_ic/ACLK hp_ic/S00_ACLK hp_ic/S01_ACLK hp_ic/M00_ACLK ps/saxihp0_fpd_aclk} {
    set pin [get_bd_pins $p]
    if {[llength [get_bd_nets -quiet -of_objects $pin]] == 0} { connect_bd_net -net $clknet $pin }
}
foreach p {hp_ic/S00_ARESETN hp_ic/S01_ARESETN hp_ic/M00_ARESETN} {
    set pin [get_bd_pins $p]
    if {[llength [get_bd_nets -quiet -of_objects $pin]] == 0} { connect_bd_net -net $prst $pin }
}
if {[llength [get_bd_nets -quiet -of_objects $irst]] == 0} { connect_bd_net $irst [get_bd_pins hp_ic/ARESETN] } else {
    connect_bd_net -net [get_bd_nets -of_objects $irst] [get_bd_pins hp_ic/ARESETN] }

assign_bd_address
# Buffers come from the CMA heap in low DDR (dma_bw.c): this PS configuration (no board preset) exposes only
# HP0_DDR_LOW and HP0_LPS_OCM on HP0, so there is no high-DDR window to map (build #5 check, 22 Nov).
# fixed register addresses (must match speccore_dma.dts)
set_property offset 0xA0000000 [get_bd_addr_segs ps/Data/SEG_dma_Reg]
set_property range 64K [get_bd_addr_segs ps/Data/SEG_dma_Reg]
set_property offset 0xA0010000 [get_bd_addr_segs ps/Data/SEG_id_gpio_Reg]
set_property range 64K [get_bd_addr_segs ps/Data/SEG_id_gpio_Reg]
set amap [open $out/address_map.txt w]
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    puts $amap "[get_property NAME $s] [format 0x%08x [get_property OFFSET $s]] [format 0x%x [get_property RANGE $s]]"
}
close $amap
set dmap [open $out/dma_map.txt w]
foreach sp [get_bd_addr_spaces dma/*] {
    foreach s [get_bd_addr_segs -quiet -of_objects $sp] {
        set off [get_property OFFSET $s]
        if {$off eq ""} { puts $dmap "[get_property NAME $sp] [get_property NAME $s] excluded"; continue }
        puts $dmap "[get_property NAME $sp] [get_property NAME $s] [format 0x%x $off] [format 0x%x [get_property RANGE $s]]"
    }
}
close $dmap
validate_bd_design
save_bd_design
set_property synth_checkpoint_mode None [get_files dma.bd]

make_wrapper -files [get_files dma.bd] -top
add_files -norecurse [glob $out/proj/dma.gen/sources_1/bd/dma/hdl/dma_wrapper.v]
set_property top dma_wrapper [current_fileset]
update_compile_order -fileset sources_1

generate_target all [get_files dma.bd]
synth_design -top dma_wrapper -part xck26-sfvc784-2LV-c
opt_design
place_design
route_design
write_bitstream -force $out/dma.bit
report_utilization -file $out/utilization.txt
report_timing_summary -file $out/timing.txt
puts "DMA BUILD DONE"
