module flopr #(parameter WIDTH = 16) (  // Default width set to 16
    input clk, reset,
    input [WIDTH-1:0] d,
    output reg [WIDTH-1:0] q
);
    always @(posedge clk, posedge reset)
        if (reset) 
            q <= 0;
        else 
            q <= d;
endmodule 
