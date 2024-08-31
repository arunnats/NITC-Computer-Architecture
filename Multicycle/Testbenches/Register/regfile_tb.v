module regfile_tb();  
    reg clk, we3;  
    reg [4:0] a1, a2,a3;  
    reg [31:0] wd3;  
    wire [31:0] rd1, rd2;  
    
    regfile UUT (clk, we3,a1,a2, a3,wd3, rd1, rd2);  
    
    initial begin  
        $dumpfile("regfile_tb.vcd");  
        $dumpvars(0, regfile_tb);

        wd3 =32'b0;  
        a1 = 5'b0;  
        a2 = 5'b0;  
        a3 = 5'b0;  
        we3 = 1'b0;  
        clk = 1'b0;  
        
        #100  
        we3=1'b1;  
        
        #20  
        wd3=32'habcd_efab;  
        a1=5'h0;  
        a2=5'h0;  
        a3=5'h1;  
        
        #20  
        wd3=32'h0123_4567;  
        a1=5'h1;  
        a2=5'h0;  
        a3=5'h2;  
        
        #20  
        wd3=32'hcccc_cccc;  
        a1=5'h2;  
        a2=5'h1;  
        a3=5'h3;  
        
        #20  
        wd3=32'h3333_4567;  
        a1=5'h2;  
        
        a2=5'h3;  
        a3=5'h1;  
        $finish;
    end  
    
    always begin  
        #10;  
        clk =~clk;  
    end  
endmodule 