module half_subtractor(
  input a,
  input b,
  output diff,
  output borrow
);
  wire not_a;
  xor xor_0(diff,a,b);
  not not_0(not_a,a);
  and and_0(borrow,not_a,b);
  
endmodule
