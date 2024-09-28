module signext(
    input [7:0] a,       // 8-bit input
    output [15:0] y      // 16-bit output
);
    assign y = {
        {8{a[7]}},       // Repeat the most significant bit of 'a' 8 times
        a               // Original 8 bits of 'a'
    };
endmodule 
