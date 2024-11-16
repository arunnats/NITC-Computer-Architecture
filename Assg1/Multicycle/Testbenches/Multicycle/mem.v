module mem(
    input clk, we,
    input [15:0] a, wd,
    output [15:0] rd
);

    reg [15:0] RAM[63:0];  // 16-bit wide memory with 64 locations

    initial begin  
        $readmesh("./memfile.dat", RAM);
    end
    
    assign rd = RAM[a[15:1]];  

    always @(posedge clk)
        if (we)
            RAM[a[15:1]] <= wd;  
endmodule
