module eight_bit_adder(
    input [7:0] a,
    input [7:0] b,
    output [7:0] sum,
    output finalcarry
);
  wire carry_wire;
  
  four_bit_adder add1(
    .a({a[0], a[1], a[2], a[3]}),
    .b({b[0], b[1], b[2], b[3]}),
    .cin(1'b0),
    .sum({sum[0], sum[1], sum[2], sum[3]}),
    .finalcarry(carry_wire)
  );
  
  four_bit_adder add2(
    .a({a[4], a[5], a[6], a[7]}),
    .b({b[4], b[5], b[6], b[7]}),
    .cin(carry_wire),
    .sum({sum[4], sum[5], sum[6], sum[7]}),
    .finalcarry(finalcarry)
  );
endmodule

module four_bit_adder(
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [3:0] sum,
    output finalcarry
);
    wire [3:0] and1, and2, and3, and4;
    wire xor0, xor1, xor2, xor3, or1, or2, or3;
 	wire [3:0]cout;
      
  	  adder add1 (.a(a[0]), .b(b[0]), .cin(cin), .sum(sum[0]), .cout(cout[0]));
      adder add2 (.a(a[1]), .b(b[1]), .cin(cout[0]), .sum(sum[1]), .cout(cout[1]));
      adder add3 (.a(a[2]), .b(b[2]), .cin(cout[1]), .sum(sum[2]), .cout(cout[2]));
      adder add4 (.a(a[3]), .b(b[3]), .cin(cout[2]), .sum(sum[3]), .cout(finalcarry));

endmodule


module adder(
	input a,
	b,
	cin,
	output sum,
	cout
	);
    
   wire [3:0] wire1;
      xor xor_f1_1 (sum,a,b,cin);
    and and_f1_0 (wire1[0],a,b);
    and and_f1_1 (wire1[1],a,cin);
    and and_f1_2 (wire1[2],cin,b);
    or or_f1_0 (cout,wire1[0],wire1[1],wire1[2]);

endmodule