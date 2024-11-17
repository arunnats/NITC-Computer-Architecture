from m5.objects import *

# Define system
system = System()
system.clk_domain = SrcClockDomain(clock='3GHz')
system.mem_mode = 'timing'  # Timing mode

# Core parameters
system.cpu = DerivO3CPU()
system.cpu.fetchWidth = 2
system.cpu.decodeWidth = 2
system.cpu.issueWidth = 2
system.cpu.commitWidth = 2
system.cpu.numIQEntries = 64
system.cpu.numROBEntries = 192
system.cpu.LQEntries = 32
system.cpu.SQEntries = 32

# L1 Cache
system.cpu.icache = L1ICache(size='32kB', assoc=4, latency=3)
system.cpu.dcache = L1DCache(size='32kB', assoc=4, latency=3)

# L2 Cache (LLC)
system.l2cache = L2Cache(size='256kB', assoc=16, latency=9)
