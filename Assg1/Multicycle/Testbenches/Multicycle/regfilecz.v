module regfilecv(cz,inC,inZ,en,out);

	input       inC, inZ, en;
	input [1:0] cz;
	output      out;
	reg C,Z;

	assign out = ~((~cz[1]&~cz[0]) | (cz[0]&~cz[1]&Z) | (~cz[0]&cz[1]&C));

	always @*
		begin
			if(en == 0)
				begin
					Z <= inZ;
					C <= inC;				
				end
		end
endmodule