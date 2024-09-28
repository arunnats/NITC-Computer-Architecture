module controller_tb;
    reg clk, reset, zero;
    reg [15:0] instr; // Now a 16-bit instruction
    reg [3:0] op;     // Opcode (4-bit)
    reg [1:0] cz;     // CZ flags (2-bit)

    wire iord, memwrite, irwrite, regdst, memtoreg, regwrite, alusrca, pcen;
    wire [1:0] alusrcb, pcsrc;
    wire [2:0] alucontrol;

    always @(*) begin
        op <= instr[15:12]; // Extract the opcode
        cz <= instr[1:0];   // Extract CZ flags
    end 
    
    controller uut(clk, reset, op, cz, zero, pcen, memwrite, irwrite, regwrite, alusrca, iord, memtoreg, regdst, alusrcb, pcsrc, alucontrol);  
    
    always #1 clk = ~clk;  
    
    initial begin  
        $dumpfile("controller_tb.vcd");  
        $dumpvars(0, controller_tb);

        clk = 0;  
        reset = 1;  
        zero = 0;  
        
        #2  reset = 1; instr = 0;  
        #2 reset = 0; instr = 16'b0000_001_001_000_00; // ADD
        #2 reset = 0; instr = 16'b0000_001_001_000_10; // ADC
        #2 reset = 0; instr = 16'b0010_001_001_000_00; // NDU
        #2 reset = 0; instr = 16'b0010_001_001_000_01; // NDZ
        #2 $finish;  
    end  
endmodule 