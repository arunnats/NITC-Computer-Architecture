module topmulti(input clk, reset,
    output [15:0] writedata, adr,
    output memwrite
);
    wire [15:0] readdata;

    mips mips(clk, reset, adr, writedata, memwrite, readdata);
    mem mem(clk, memwrite, adr, writedata, readdata);
endmodule 