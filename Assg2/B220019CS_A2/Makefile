# Makefile for CUDA program

# Compiler and flags
NVCC = nvcc
CFLAGS = -O3

# Program name
PROGRAM = B220019CS_A2.cu
OUTPUT = output

# Input, output, and timing files
INPUTS = indata10.txt indata50.txt indata80.txt indata100.txt indata200.txt indata500.txt
OUTPUTS = output10.txt output50.txt output80.txt output100.txt output200.txt output500.txt
TIMINGS = timing10.txt timing50.txt timing80.txt timing100.txt timing200.txt timing500.txt

# Default target
all: build

# Build the program
build:
	$(NVCC) $(CFLAGS) $(PROGRAM) -o $(OUTPUT)

# Run the program with each input, output, and timing file
run: build
	@echo "Running program with indata10.txt..."
	./$(OUTPUT) indata10.txt output10.txt timing10.txt
	@echo "Running program with indata50.txt..."
	./$(OUTPUT) indata50.txt output50.txt timing50.txt
	@echo "Running program with indata80.txt..."
	./$(OUTPUT) indata80.txt output80.txt timing80.txt
	@echo "Running program with indata100.txt..."
	./$(OUTPUT) indata100.txt output100.txt timing100.txt
	@echo "Running program with indata200.txt..."
	./$(OUTPUT) indata200.txt output200.txt timing200.txt
	@echo "Running program with indata500.txt..."
	./$(OUTPUT) indata500.txt output500.txt timing500.txt

# Clean the generated files
clean:
	rm -f $(OUTPUT) $(OUTPUTS) $(TIMINGS)
