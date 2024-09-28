module signext(
    input [8:0] a, 
    output [15:0] y
);
    assign y = {
        {32{a[8]}}, // Repeat the most significant bit of 'a' 16 times to fill the upper 16 bits of 'y'
        a            // Assign the original 16 bits of 'a' to the lower 16 bits of 'y'
    };
endmodule 