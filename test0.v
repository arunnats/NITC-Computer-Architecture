//top module
module test0(	input clk, reset,
					output [15:0] writedata, adr,
					output memwrite);

	wire [15:0] readdata;
	
	//instantiate processor and memory
	mips mips(clk, reset, adr, writedata, memwrite, readdata);
	mem mem(clk, memwrite,adr, writedata, readdata);
				
endmodule

//mips module
module  mips(input clk, reset,
					output [15:0] adr, writedata,
					output memwrite,
					input [15:0] readdata);
					
	wire zero, pcen, irwrite, regwrite, alusrca, iord, memtoreg, regdst;
	wire [1:0] alusrcb;
	wire [1:0] pcsrc;
	wire [1:0] alucontrol;
	wire [3:0] op;
	controller c(clk, reset, op, zero, pcen, memwrite, irwrite, regwrite, alusrca, iord, memtoreg, regdst, alusrcb, pcsrc, alucontrol);

	//NEEDS TO BE CHECKED
	datapath dp(clk, reset,
					pcen, irwrite, regwrite,
					alusrca, iord, memtoreg, regdst,
					alusrcb, pcsrc, alucontrol,
					op, zero,
					adr, writedata, readdata);
	
endmodule

module mem(input clk, we, input [15:0] a, wd, output [15:0] rd);

	reg[15:0] RAM[63:0];
	initial
		begin
			$readmemh ("./memfile.dat",RAM);
		end
	assign rd = RAM[a[15:1]]; //word aligned
	
	always @(posedge clk)
		if (we) RAM[a[15:1]] <=wd;
endmodule


//CONTROLLER

module controller(input clk, reset,
						input [3:0] op,
						input zero,
						output pcen, memwrite, irwrite, regwrite,
						output alusrca, iord, memtoreg, regdst,
						output [1:0] alusrcb,
						output [1:0] pcsrc,
						output [1:0] alucontrol);
	wire [1:0] aluop;
	wire branch, pcwrite;
	//main decoder and alu decoder
	maindec md(clk, reset, op,
				pcwrite, memwrite, irwrite, regwrite,
				alusrca, branch, iord, memtoreg, regdst,
				alusrcb, pcsrc, aluop);
	aludec ad(op, aluop, alucontrol);
	assign pcen = pcwrite | ( branch & zero );
endmodule

//main decoder
module maindec(input clk, reset,
					input [3:0] op,
					output pcwrite, memwrite, irwrite, regwrite,
					output alusrca, branch, iord, memtoreg, regdst,
					output [1:0] alusrcb,
					output [1:0] pcsrc,
					output [1:0] aluop);
	//FSM
	parameter FETCH =5'b00000; 
	parameter DECODE=5'b00001;
	parameter MEMADR=5'b00010; 
	parameter MEMRD=5'b00011;
	parameter MEMWB=5'b00100;
	parameter MEMWR=5'b00101;
	parameter EXECUTE=5'b00110;
	parameter ALUWRITEBACK=5'b00111;
	parameter BRANCH=5'b1000;
	parameter JUMPAL=5'b1001;
	
	//opcodes
	parameter LW = 4'b1010;
	parameter SW = 4'b1001;
	parameter ADD = 4'b0000;
	parameter BEQ = 4'b1011;
	parameter NDU = 4'b0010;
	parameter JAL = 4'b1101;
	
	reg [4:0] state, nextstate;
	reg [14:0] controls;
	
	//state reg
	always @(posedge clk or posedge reset)
		if(reset) state <= FETCH;
		else state <= nextstate;
	//next state logic
	
	always @(*)
		case(state)
			FETCH: nextstate <=DECODE;
			DECODE: case(op)
				LW: nextstate <=MEMADR;
				SW: nextstate <=MEMADR;
				ADD: nextstate <=EXECUTE;
				NDU: nextstate <=EXECUTE;
				BEQ: nextstate <= BRANCH;
				JAL: nextstate <=JUMPAL;
				default: nextstate <=FETCH;
				endcase
			MEMADR: case(op)
				LW: nextstate <=MEMRD;
				SW: nextstate <=MEMWR;
				default: nextstate <=FETCH;
				endcase
			MEMRD: nextstate <=MEMWB;
			MEMWB: nextstate <=FETCH;
			MEMWR: nextstate <=FETCH;
			EXECUTE: nextstate <=ALUWRITEBACK;
			ALUWRITEBACK: nextstate <=FETCH;
			BRANCH: nextstate <=FETCH;
			JUMPAL: nextstate <=FETCH;
			default: nextstate <=FETCH;
		endcase
	//output control signals logic
	assign {pcwrite, memwrite, irwrite, regwrite,
				alusrca, branch, iord, memtoreg, regdst, 
				alusrcb, pcsrc, aluop } = controls;
	
	always @(*)
		case(state)
			FETCH:   controls   <=   15'b1010_00000_0100_00;  
			DECODE:   controls   <=   15'b0000_00000_1100_00;  
			MEMADR:   controls   <=   15'b0000_10000_1000_00;  
			MEMRD:   controls   <=   15'b0000_00100_0000_00;  
			MEMWB:   controls   <=   15'b0001_00010_0000_00;  
			MEMWR:   controls   <=   15'b0100_00100_0000_00;  
			EXECUTE:   controls   <=   15'b0000_10000_0000_10;  
			ALUWRITEBACK:   controls   <=   15'b0001_00001_0000_00;  
			BRANCH:   controls   <=   15'b0000_11000_0001_01;  
			JUMPAL:   controls   <=   15'b1000_00000_0010_00;   
			default:   controls   <=   15'b0000_xxxxx_xxxx_xx; 
		endcase
endmodule


//alu decoder - done

module aludec( input [3:0] op,
					input [1:0] aluop,
					output reg [1:0] alucontrol);
	always@(*)
		case(aluop)
			2'b00: alucontrol <=2'b10; //add
			2'b01: alucontrol <=2'b11; //sub
			2'b10: 	case(op)
							4'b0000: alucontrol<=2'b10; //ADD
							4'b0010: alucontrol<=2'b00;//NAND
							default: alucontrol <=2'bxx; //default
						endcase
			default: alucontrol <=2'bxx;
		endcase
endmodule

//DATAPATH
//choose between branchs and jumps immediate, we can use pcsrc for this
module datapath(	input clk, reset,
						input pcen, irwrite, regwrite,
						input alusrca, iord, memtoreg, regdst,
						input [1:0] alusrcb,
						input [1:0] pcsrc,
						input [1:0] alucontrol,
						output [3:0] op,
						output zero,
						output [15:0] adr, writedata,
						input [15:0] readdata);
	wire [2:0] writereg;
	wire [15:0] pcnext, pc;
	wire [15:0] instr, data, srca, srcb;
	wire [15:0] a;
	wire [15:0] aluresult, aluout;
	wire [15:0] signimm,signimmbr,signimmjal; //sign extended imm
	wire [15:0] signimmsh; //sign extended immediate left shifted by 1 or 0?
	wire [15:0] wd3, rd1, rd2;
	//op field to controller
	assign op = instr[15:12];
	//datapath
		//set up the new value replacements for 
			//pc
			//instr
			//read data
		flopenr #(16) prcreg(clk, reset, pcen, pcnext, pc);//pc = pcnext
		mux2 #(16) admux(pc,aluout,iord,adr);//choose between pc and alout for address based on i/o or read data memory
		flopenr #(32) instrreg(clk,reset,irwrite,readdata,instr); //instr = readdata
		flopr #(16) datareg(clk, reset, readdata,data); //data = readdata
		
		//register file
		mux2 #(3) regdstmux(instr[8:6],instr[5:3], regdst, writereg); //choose between rt and rd for the destination register
		mux2 #(16) wdmux(aluout, data, memtoreg, wd3);// choose the write-back data between aluout and the readdata(sw)
		regfile rf(clk, regwrite, instr[11:9],instr[8:6], writereg, wd3, rd1, rd2); //instantiate the registerfile
		
		//signextend for branch and jal

		signext #(6) brse(instr[5:0],signimmbr);
		signext #(9) jlse(instr[8:0], signimmjal);
		mux2 #(16) slmux(signimmbr,signimmjal, pcsrc[1],signimm); 
		sl1 immsh(signimm, signimmsh);
		
		flopr #(16) areg(clk, reset, rd1, a);//a = rd1, i.e the first read port output
		flopr #(16) breg(clk, reset, rd2, writedata);//writedata = rd2, i.e the second read port output
		mux2 #(16) scramux(pc, a, alusrca, srca);//srca is between pc(when pc +2 is done ) and a
		mux4 #(16) srcbmux(writedata,16'b10, signimm,signimmsh, alusrcb, srcb);// srcb is between writedata, 2, signimm, signimmsh
		
		//alu
		alu alu(srca, srcb,alucontrol, aluresult, zero);
		flopr #(16) alureg(clk, reset, aluresult, aluout);// aluout = aluresult
		
		//finally deciding next pc
		
		mux3 #(16) pcmux(aluresult, aluout,aluout, pcsrc, pcnext);
	
endmodule

			
//OTHER MODULES - SUPPLEMENTARY
module regfile(input clk, 
					input we3,
					input [2:0] ra1, ra2, wa3,
					input [15:0] wd3,
					output [15:0] rd1, rd2);
			
	reg [15:0] rf[0:7];
	always @(posedge clk)
			if(we3) rf[wa3] <=wd3;
	assign rd1 = (ra1 != 0) ? rf[ra1]: 0;
	assign rd2 = (ra2 !=0 ) ? rf[ra2]: 0;
		
endmodule

module sl1(input [15:0]a , output [15:0] y);
	assign y = {a[14:0],1'b0}; 
endmodule

module flopenr #(parameter WIDTH = 8)
			(input clk, reset,
				input en,
				input [WIDTH-1:0] d,
				output reg [WIDTH-1:0] q);
	always @(posedge clk, posedge reset)
		if(reset) q<=0;
		else if(en) q <=d;
endmodule

module flopr #(parameter WIDTH = 8)
			(input clk, reset,
			 input [WIDTH-1:0]  d,
			 output reg[WIDTH-1:0] q);
	always @(posedge clk, posedge reset)
		if(reset) q<=0;
		else q<=d;
endmodule

module signext #(parameter WIDTH = 6)
					(input [WIDTH-1:0]a, output [15:0]y);
	assign y = {{(15-WIDTH){a[WIDTH-1]}}, a};
endmodule					
		
module mux2 #(parameter WIDTH=8)
				(input [WIDTH-1:0] d0,d1,
				input s,
				output [WIDTH-1:0] y);
				assign y = s? d1: d0;
endmodule

module mux3 #(parameter WIDTH = 8)
				(input [WIDTH-1:0] d0, d1, d2,
				input [1:0] s,
				output [WIDTH-1:0] y);
				assign #1 y = s[1]?d2:(s[0]? d1:d0);
endmodule

module mux4 #(parameter WIDTH = 8)
				(input [WIDTH-1:0] d0, d1, d2, d3,
				input [1:0] s,
				output reg[WIDTH-1:0] y);
	always @(*)
		case(s)
			2'b00: y <=d0;
			2'b01: y <=d1;
			2'b10: y<= d2;
			2'b11: y <=d3;
		endcase
endmodule
//ALU
//inputs: A,B, opcode
//output: Y, zero
//note that only add, nand, load word and store word and branch have their operations here. Every  other arithmetic operation is done outside
module alu( 	input [15:0] A,B,
					input [1:0] F,
					output reg[15:0] Y,
					output zero);

	always@(*)
		case(F[1:0])
			2'b00: Y <= ~(A&B); //NAND
			2'b10: Y <= A+B; //ADD
			2'b11: Y <= A-B; //SUB
			default: Y <=16'bx; //default
		endcase
	assign zero = (Y==32'b0);
endmodule