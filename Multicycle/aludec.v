module aludec(
    input [1:0] cz,
    input [1:0] aluop,
    output reg [2:0] alucontrol
);

    always @( * )
        case(aluop)
            2'b00: alucontrol <= 3'b010; // add
            2'b01: alucontrol <= 3'b010; // sub

            // RTYPE instruction use the 2-bit cz field of instruction to specify ALU operation
            2'b10: case(cz)
                2'b10: alucontrol <= 3'b010; // ADC
                2'b01: alucontrol <= 3'b110; // NDZ
                default: alucontrol <= 2'bxx; // ???
            endcase
    default: alucontrol <= 3'bxxx; // ???
    endcase
endmodule 