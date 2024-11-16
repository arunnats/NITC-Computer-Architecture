module eight_bit_adder_testbench;
   reg [7:0] a;
   reg [7:0] b;
   wire [7:0] sum;
   wire finalcarry;

   eight_bit_adder testbench (
       .a(a),
       .b(b),
       .sum(sum),
       .finalcarry(finalcarry)
   );
    
   integer i, j;

   initial begin
       // Setup VCD file for waveform viewing
       $dumpfile("eight_bit_adder.vcd");
       $dumpvars(0, eight_bit_adder_testbench);

       for (i = 0; i < 256; i = i + 1) begin
           for (j = 0; j < 256; j = j + 1) begin
               a = i;
               b = j;
               #10;
           end
       end
       $finish;
   end
endmodule
