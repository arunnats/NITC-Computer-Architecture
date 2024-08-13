module mem(
    input clk, we,
    input [31:0] a, wd,
    output [31:0] rd
);

    reg [31:0] RAM[63:0];

    initial
        begin
            // gotta load the instructions here
        end
    assign rd = RAM[a[31:2]]; // word aligned

    always @(posedge clk)
        if (we)
        RAM[a[31:2]] <= wd;
endmodule 