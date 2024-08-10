module eightBitShiftRegister (
    input wire clk,
    input wire reset,
    input wire data_in,
    output [7:0] data_out
);
	
	 reg[7:0] shift_reg;
    always @(posedge clk or negedge reset) begin
        if (~reset) begin
            shift_reg <= 8'bx;
        end else begin
            shift_reg[0] <= data_in;
				shift_reg[7:1] <= shift_reg[6:0];
        end
    end
	 
	 
	 assign data_out = shift_reg;

endmodule