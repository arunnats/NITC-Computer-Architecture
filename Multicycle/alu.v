module alu( input [31:0] A, B,
    input [2:0] F,
    output reg [31:0] Y,
    output Zero
);
    always @ ( * )
        case (F[2:0])
            3'b000: Y <= A & B; // AND
            3'b001: Y <= A | B; // OR
            3'b010: Y <= A + B; // ADD
            3'b011: Y <= 0; // not a valid opcode(?)
            3'b110: Y <= A - B; // SUB
            3'b111: Y <= A < B ? 1:0; //SLT
            default: Y <= 0; //default to 0, probably wont happen
        endcase
    assign Zero = (Y == 32'b0);
endmodule 