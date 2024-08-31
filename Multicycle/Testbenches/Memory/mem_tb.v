module mem_tb;  
    reg clk, we;  
    reg [31:0] a, wd;  
    wire [31:0] rd; 
    mem uut(clk, we, a, wd, rd); 
    
    always begin 
        clk <= 1; 
        #5; 
        clk <= 0; 
        #5; 
    end  
    
    initial begin  
        we <= 0;  
        wd <= 0;  
    end 

    initial begin  
        $dumpfile("mem_tb.vcd"); 
        $dumpvars(0, mem_tb);  

        a <= 10; //read memory from address 10  
        #10 a <= 25; //read memory from address 25  
        #10 a <= 25;  
        we <= 1;  
        wd <= 32'habcd1234; //write abcd1234 to address 25 
        #10 we <= 0;  
        a <= 5; //read memory from address 5  
        #10 a <= 25; //read memory from address 25  
        #10 $finish;  
    end  
endmodule 