module datapath(input clk, reset,
    input pcen, irwrite, regwrite,
    input alusrca, iord, memtoreg, regdst,
    input [1:0] alusrcb,
    input [1:0] pcsrc,
    input [2:0] alucontrol,
    output [5:0] op, 
    output [1:0] cz,
    output zero,
    output [15:0] adr, writedata,
    input [15:0] readdata
); 
    // Internal wires 
    wire [4:0] writereg;
    wire [15:0] pcnext, pc;
    wire [15:0] instr, data, srca, srcb;
    wire [15:0] a;
    wire [15:0] aluresult, aluout;
    wire [15:0] signimm; // the sign-extended immediate
    wire [15:0] signimmsh; // the sign-extended immediate shifted left by 2
    wire [15:0] wd3, rd1, rd2;

    // Extracting the op code and cz fields
    assign op = instr[15:12];
    assign cz = instr[1:0];

    // The datapath
    flopenr #(16) pcreg(clk, reset, pcen, pcnext, pc);
    mux2 #(16) adrmux(pc, aluout, iord, adr);
    flopenr #(16) instrreg(clk, reset, irwrite, readdata, instr);
    flopr #(16) datareg(clk, reset, readdata, data);
    mux2 #(5) regdstmux(instr[7:6], instr[5:3], regdst, writereg);
    mux2 #(16) wdmux(aluout, data, memtoreg, wd3);
    regfile rf(clk, regwrite, instr[11:9], instr[8:6],
    writereg, wd3, rd1, rd2);
    signext se(instr[8:0], signimm);
    sl2 immsh(signimm, signimmsh);
    flopr #(16) areg(clk, reset, rd1, a);
    flopr #(16) breg(clk, reset, rd2, writedata);
    mux2 #(16) srcamux(pc, a, alusrca, srca);
    mux4 #(16) srcbmux(writedata, 32'b100, signimm, signimmsh,
    alusrcb, srcb);
    alu alu(srca, srcb, alucontrol, aluresult, zero);
    flopr #(16) alureg(clk, reset, aluresult, aluout);
    mux3 #(16) pcmux(aluresult, aluout,
    {pc[31:28], instr[25:0], 2'b00}, pcsrc, pcnext);
endmodule