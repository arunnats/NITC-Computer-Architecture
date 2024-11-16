module mux2 #(parameter WIDTH = 16) (  // Default width set to 16
    input [WIDTH-1:0] d0, d1,
    input s,
    output reg [WIDTH-1:0] y
);
    always @( * )
        case(s)
            1'b0: y <= d0;
            1'b1: y <= d1;
        endcase
endmodule
