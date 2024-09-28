module maindec(
    input clk, reset,
    input [3:0] op, // 4-bit opcode instead of 6-bit
    output pcwrite, memwrite, irwrite, regwrite,
    output alusrca, branch, iord, memtoreg, regdst,
    output [1:0] alusrcb,
    output [1:0] pcsrc,
    output [1:0] aluop
);

    // FSM States
    parameter FETCH = 5'b00000; // State 0
    parameter DECODE = 5'b00001; // State 1
    parameter MEMADR = 5'b00010; // State 2
    parameter MEMRD = 5'b00011; // State 3
    parameter MEMWB = 5'b00100; // State 4
    parameter MEMWR = 5'b00101; // State 5
    parameter EXECUTE = 5'b00110; // State 6
    parameter ALUWRITEBACK = 5'b00111; // State 7
    parameter BRANCH = 5'b01000; // State 8
    parameter ADDIEXECUTE = 5'b01001; // State 9
    parameter ADDIWRITEBACK = 5'b01010; // state a
    parameter JUMP = 5'b01011; // State b

    // MIPS Instruction Opcodes
    parameter ADD = 4'b0000;
    parameter ADC = 4'b0001;
    parameter NDU = 4'b0010;
    parameter NDZ = 4'b0011;
    parameter LW  = 4'b0100;
    parameter SW  = 4'b0101;
    parameter BEQ = 4'b0110;
    parameter JAL = 4'b0111;
    reg [4:0] state, nextstate;
    reg [16:0] controls; 

    // state register
    always @(posedge clk or posedge reset)
        if(reset) 
            state <= FETCH;
        else 
            state <= nextstate;
                
    // next state logic
    always @( * )
    case(state)
        FETCH: nextstate <= DECODE;

        DECODE: case(op)
            LW: nextstate <= MEMADR;
            SW: nextstate <= MEMADR;
            ADD: nextstate <= EXECUTE;
            ADC: nextstate <= EXECUTE;
            NDU: nextstate <= EXECUTE;
            NDZ: nextstate <= EXECUTE;
            BEQ: nextstate <= BRANCH;
            JAL: nextstate <= JUMP;
            default: nextstate <= FETCH;
        endcase

        MEMADR: case(op)
            LW: nextstate <= MEMRD;
            SW: nextstate <= MEMWR;
            default: nextstate <= FETCH;
        endcase

        MEMRD: nextstate <= MEMWB;
        MEMWB: nextstate <= FETCH;
        MEMWR: nextstate <= FETCH;
        EXECUTE: nextstate <= ALUWRITEBACK;
        ALUWRITEBACK: nextstate <= FETCH;
        BRANCH: nextstate <= FETCH;
        JUMP: nextstate <= FETCH;

        default: nextstate <= FETCH;
    endcase


    // output logic
    assign {pcwrite, memwrite, irwrite, regwrite, alusrca, branch, iord, memtoreg, regdst, alusrcb, pcsrc, aluop} = controls; 
    always @( * )
    case(state)
        FETCH: controls <= 19'b1010_00000_0100_00;   // PCWrite, IRWrite, ALUSrcB = 01 (increment PC)
        DECODE: controls <= 19'b0000_00000_1100_00;  // ALUSrcB = 11 (sign-extend immediate)
        MEMADR: controls <= 19'b0000_10000_1000_00;  // ALUSrcA = 1, ALUSrcB = 10 (base address + offset)
        MEMRD: controls <= 19'b0000_00100_0000_00;   // Read memory
        MEMWB: controls <= 19'b0001_00010_0000_00;   // Write back to register file
        MEMWR: controls <= 19'b0100_00100_0000_00;   // Write to memory
        EXECUTE: controls <= 19'b0000_10000_0000_10; // ALU operation (R-type)
        ALUWRITEBACK: controls <= 19'b0001_00001_0000_00; // Write ALU result to register
        BRANCH: controls <= 19'b0000_11000_0001_01;  // Branch if equal
        JUMP: controls <= 19'b1000_00000_0010_00;    // Jump
        default: controls <= 19'b0000_xxxxx_xxxx_xx; // Should never happen
    endcase

endmodule 