module mips(input clk, reset,
    output [15:0] adr, writedata,
    output memwrite,
    input [15:0] readdata
);

    wire zero, pcen, irwrite, regwrite,
    alusrca, iord, memtoreg, regdst;
    wire [1:0] alusrcb;
    wire [1:0] pcsrc;
    wire [2:0] alucontrol;
    wire [3:0] op, 
    wire [1:0] cz; 

    // The control unit receives the current instruction from the datapath and tells the
    // datapath how to execute that instruction.
    controller c(clk, reset, op, cz, zero,
    pcen, memwrite, irwrite, regwrite,
    alusrca, iord, memtoreg, regdst,
    alusrcb, pcsrc, alucontrol);
        
    // The datapath operates on words of data. It
    // contains structures such as memories, registers, ALUs, and multiplexers.
    // MIPS is a 32-bit architecture, so we will use a 32-bit datapath.
    datapath dp(clk, reset,
    pcen, irwrite, regwrite,
    alusrca, iord, memtoreg, regdst,
    alusrcb, pcsrc, alucontrol,
    op, cz, zero,
    adr, writedata, readdata);
    
endmodule 