module half_subtractor_testbench;

    reg a;
    reg b;
    wire diff;
    wire borrow;

    // Instantiate the half_subtractor module
    half_subtractor testbench (
        .a(a),
        .b(b),
        .diff(diff),
        .borrow(borrow)
    );
    
    integer i, j;
    
    initial begin
        // Setup VCD file for waveform viewing
        $dumpfile("half_subtractor.vcd");
        $dumpvars(0, half_subtractor_testbench);

        $display("a\tb\tdiff\tborrow");
        $monitor("%b\t%b\t%b\t%b", a, b, diff, borrow);
        
        // Test all combinations of a and b
        for (i = 0; i < 2; i = i + 1) begin
            for (j = 0; j < 2; j = j + 1) begin
                a = i;
                b = j;
                #10;
            end
        end

        $finish;
    end

endmodule

