module signext(
    input [15:0] a, 
    output [31:0] y
);
    assign y = {
        {16{a[15]}}, // Repeat the most significant bit of 'a' 16 times to fill the upper 16 bits of 'y'
        a            // Assign the original 16 bits of 'a' to the lower 16 bits of 'y'
    };
endmodule 