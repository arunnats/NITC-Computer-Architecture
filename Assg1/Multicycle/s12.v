module sl2 (
    input [15:0] a,      // 16-bit input
    output [15:0] y      // 16-bit output
);
    assign y = {a[13:0], 2'b00};  // Shift left by 2: Append two 0s to the least significant bits
endmodule
