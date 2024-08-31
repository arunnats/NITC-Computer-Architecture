module alu_tb;  
    reg [31:0] A, B;  
    reg [2:0] F;  
    wire Zero;  
    wire [31:0] Y;  
    
    alu uut (A, B, F, Y, Zero);  
    
    initial 
    begin  
        $dumpfile("alu_tb.vcd");  
        $dumpvars(0, alu_tb);
        
        A = 0;  
        B = 0;  
        F = 0;  
        
        #8 A = 2; B = 3;  
        #8 A = 5; B = 7;  
        #8 A = 13;  B = 10;  
        #8 A = 25;  B = 25;  
        #8 A = 35;  B = 56;  
        #8 A = 75;  B = 100;  
        #8 A = 200; B = 155;  
        #8 A = 360; B = 400;  
        #8 A = 1023;  B = 780;  
        #8 A = 1570;  B = 2047;  
        $finish;
    end  
    
    always 
    begin  
        #1 F = F + 1;  
    end  
 
endmodule  
