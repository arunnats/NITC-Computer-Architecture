module controller_tb;  
    reg clk, reset, zero;  
    reg [5:0] op, funct;  
    
    wire   iord, memwrite, irwrite, regdst, memtoreg, regwrite, alusrca, pcen;  
    wire [1:0] alusrcb, pcsrc;  
    wire [2:0] alucontrol;  
    
    reg [31:0] instr;  
    
    always @ ( * )  
    begin  
        op <= instr [31:26];  
        funct <= instr [5:0];  
    end  
    
    controller uut(clk, reset, op, funct, zero, pcen, memwrite, irwrite, regwrite, alusrca, iord, memtoreg, regdst, alusrcb, pcsrc, alucontrol);  
    
    always #1 clk = ~clk;  
    
    initial begin  
        $dumpfile("controller_tb.vcd");  
        $dumpvars(0, controller_tb);

        clk = 0;  
        reset = 1;  
        zero = 0;  
        
        #2  reset = 1; instr = 0;  
        #2  reset = 0; instr = 32'h20020005; //addi  
        #2  reset = 0; instr = 32'h20020005; //addi  
        
        #2  reset = 0; instr = 32'h20020005; //addi  
        #2 reset = 0; instr = 32'h20020005; //addi  
        #2 reset = 0; instr = 32'h2003000c; //addi  
        #2 reset = 0; instr = 32'h2003000c; //addi  
        #2 reset = 0; instr = 32'h2003000c; //addi  
        #2 reset = 0; instr = 32'h00e22025; //or  
        #2 reset = 0; instr = 32'h00e22025; //or  
        #2 reset = 0; instr = 32'h00e22025; //or  
        #2 reset = 0; instr = 32'h00e22025; //or  
        #2 reset = 0; instr = 32'h00642824; //and  
        #2 reset = 0; instr = 32'h00642824; //and  
        #2 reset = 0; instr = 32'h00642824; //and  
        #2 reset = 0; instr = 32'h00642824; //and  
        #2 reset = 0; instr = 32'h00a42820; //add  
        #2 reset = 0; instr = 32'h00a42820; //add  
        #2 reset = 0; instr = 32'h00a42820; //add  
        #2 reset = 0; instr = 32'h00a42820; //add  
        #2 reset = 0; instr = 32'h00853820; //add  
        #2 reset = 0; instr = 32'h00853820; //add  
        #2 reset = 0; instr = 32'h00853820; //add  
        #2 reset = 0; instr = 32'h00853820; //add  
        #2 reset = 0; instr = 32'h00e23822; //sub  
        #2 reset = 0; instr = 32'h00e23822; //sub  
        #2 reset = 0; instr = 32'h00e23822; //sub  
        #2 reset = 0; instr = 32'h00e23822; //sub  
        #2 reset = 0; instr = 32'hac670044; //sw  
        #2 reset = 0; instr = 32'hac670044; //sw  
        #2 reset = 0; instr = 32'hac670044; //sw  
        #2 reset = 0; instr = 32'hac670044; //sw  
        #2 reset = 0; instr = 32'h8c020050; //lw  
        #2 reset = 0; instr = 32'h8c020050; //lw  
        #2 reset = 0; instr = 32'h8c020050; //lw  
        #2 reset = 0; instr = 32'h8c020050; //lw  
        #2 reset = 0; instr = 32'h8c020050; //lw  
        #2 $finish;  
    end  
endmodule 