# Gowin command-line build for the SPTorch Tang9k UART responder.
# Run from the repository root with:
#   & 'C:\Gowin\Gowin_V1.9.12.02_SP2_x64\IDE\bin\gw_sh.exe' hardware/tang9k/uart_responder/build.tcl

cd [file dirname [info script]]
open_project tang9k_uart_responder.gprj
set_option -top_module tang9k_uart_responder
set_option -output_base_name tang9k_uart_responder
set_option -verilog_std v2001
run all
exit
