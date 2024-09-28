module aludec(
    input [1:0] cz, // Carry and Zero flags
    input [1:0] op, // 4-bit opcode (since your instruction is 16 bits)
    output reg [2:0] alucontrol
);

    always @(*) begin
        case(op)
            2'b00: begin
                case(cz)
                    2'b00: alucontrol <= 3'b010; // ADD
                    2'b10: alucontrol <= 3'b011; // ADC (Add with Carry)
                    default: alucontrol <= 3'bxxx; // Undefined
                endcase
            end
            2'b10: begin
                case(cz)
                    2'b00: alucontrol <= 3'b000; // NDU
                    2'b01: alucontrol <= 3'b001; // NDZ (AND with Zero condition)
                    default: alucontrol <= 3'bxxx; // Undefined
                endcase
            end
            // Handle other opcodes like LW, SW, BEQ, JAL
            default: alucontrol <= 3'bxxx; // Undefined
        endcase
    end
endmodule
