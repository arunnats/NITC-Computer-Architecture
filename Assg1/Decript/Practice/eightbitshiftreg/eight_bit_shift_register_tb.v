module eightBitShiftRegisterTestBench;
    reg clk;
    reg reset;
    reg data_in;
    wire [7:0] data_out;

    eightBitShiftRegister tb (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .data_out(data_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    // Apply test vectors and dump waveforms
    initial begin
        // Setup VCD file for waveform viewing
        $dumpfile("eightBitShiftRegister.vcd");
        $dumpvars(0, eightBitShiftRegisterTestBench);

        reset = 1;
        #20 data_in = 0; 
        #20 data_in = 1;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 data_in = 1;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 data_in = 1;
        #20 reset = 0;
        #20 reset = 1;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 data_in = 1;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 0;
        #20 data_in = 1;
        #20 $finish;  // End the simulation
    end
endmodule
