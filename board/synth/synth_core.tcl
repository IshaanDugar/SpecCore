# Out-of-context implementation of the SpecCore CPU (RV + LLM unit) for the KV260's chip (xck26).
# Purpose: MEASURE achievable clock and resources (no board design, no memories: the Wrapper's sim
# memories are left out). Same RTL files the simulator uses. Clock target = first tcl arg (ns).
# Run through tools/synth_core.py (memory/time guard).
set out [file normalize [lindex $argv 0]]
set period [lindex $argv 1]
set root [file normalize [lindex $argv 2]]
set netlist_only [expr {[lindex $argv 3] eq "netlist-only"}]
file mkdir $out
set_param general.maxThreads 2
foreach f [glob $root/core/rtl/*.v] { read_verilog $f }
read_verilog -sv [glob $root/ext/rtl/*.sv]
set_property include_dirs [list $root/ext/rtl] [current_fileset]
synth_design -top RV -part xck26-sfvc784-2LV-c -mode out_of_context -include_dirs $root/ext/rtl
create_clock -name clk -period $period [get_ports CLK]
report_utilization -file $out/util_synth.txt
if {$netlist_only} {
    write_verilog -mode funcsim -force $out/RV_funcsim.v
    puts "CORE NETLIST DONE $out/RV_funcsim.v"
    exit 0
}
opt_design
place_design
route_design
report_utilization -file $out/util_routed.txt
report_utilization -hierarchical -hierarchical_depth 2 -file $out/util_hier.txt
report_timing_summary -max_paths 5 -file $out/timing_routed.txt
report_timing -max_paths 3 -nworst 1 -path_type full -file $out/worst_paths.txt
# sign-off reports (FPGA stand-in for ASIC sign-off; DFT not applicable to a bitstream)
report_methodology -file $out/methodology.txt
report_cdc -details -file $out/cdc.txt
report_drc -file $out/drc.txt
report_clock_interaction -file $out/clock_interaction.txt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "CORE IMPL DONE period $period ns WNS $wns ns"
