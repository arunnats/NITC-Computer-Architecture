module sl1 (
    input [15:0] a,      // 16-bit input
    output [15:0] y      // 16-bit output
);
    assign y = {a[14:0], 1'b0};  // Shift left by 1: Append one 0 to the least significant bit
endmodule
