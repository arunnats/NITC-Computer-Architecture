module mem(
    input clk, we,
    input [15:0] a, wd,
    output [15:0] rd
);

    reg [15:0] RAM[63:0];  // 16-bit wide memory with 64 locations

    initial begin  
        RAM[0]   <=   16'h2002;  
        RAM[1]   <=   16'h0005;  
        RAM[2]   <=   16'h2003;  
        RAM[3]   <=   16'h000c;  
        RAM[4]   <=   16'h2067;  
        RAM[5]   <=   16'hfff7;  
        RAM[6]   <=   16'h00e2;  
        RAM[7]   <=   16'h2025;  
        RAM[8]   <=   16'h0064;  
        RAM[9]   <=   16'h2824;  
        RAM[10]  <=   16'h00a4;  
        RAM[11]  <=   16'h2820;  
        RAM[12]  <=   16'h10a7;  
        RAM[13]  <=   16'h000a;  
        RAM[14]  <=   16'h0064;  
        RAM[15]  <=   16'h202a;  
        RAM[16]  <=   16'h1080;  
        RAM[17]  <=   16'h0001;  
        RAM[18]  <=   16'h2005;  
        RAM[19]  <=   16'h0000;  
    end
    
    assign rd = RAM[a[15:1]];  

    always @(posedge clk)
        if (we)
            RAM[a[15:1]] <= wd;  
endmodule
