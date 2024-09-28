module mem_tb;  
    reg clk, we;  
    reg [15:0] a, wd;  
    wire [15:0] rd; 
    
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

        // Read memory from address 10
        a <= 10;  
        #10;  
        
        // Read memory from address 25
        a <= 25;  
        #10;  

        // Write 16-bit value to address 25
        we <= 1;  
        wd <= 16'habcd;  // Write 16-bit value 
        #10;  
        
        we <= 0;  
        
        // Read memory from address 5
        a <= 5;  
        #10;  
        
        // Read memory from address 25
        a <= 25;  
        #10;  
        
        $finish;  
    end  
endmodule
