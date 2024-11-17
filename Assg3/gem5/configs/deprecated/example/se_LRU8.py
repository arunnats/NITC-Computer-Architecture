from m5.objects import *
from m5.util import *

# Basic system configuration
system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '3GHz'
system.clk_domain.voltage_domain = VoltageDomain()

# Memory configuration
system.mem_mode = 'timing'  # Out-of-order CPUs typically use timing mode
system.mem_ranges = [AddrRange('4GB')]  # 4GB of physical memory
system.membus = SystemXBar()

# DDR3 memory controller configuration
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# CPU configuration
system.cpu = DerivO3CPU()
system.cpu.fetchWidth = 2
system.cpu.decodeWidth = 2
system.cpu.issueWidth = 2
system.cpu.commitWidth = 2
system.cpu.inOrder = False
system.cpu.numROBEntries = 192
system.cpu.numIQEntries = 64
system.cpu.numLSQEntries = 32

# Branch predictor configuration (TAGE SC-L)
system.cpu.branchPred = TAGE_SC_L_8KB()
system.cpu.branchPred.BTBEntries = 4096  # 4K BTB
system.cpu.branchPred.RASSize = 32  # RAS size 32

# L1 Instruction Cache configuration
system.cpu.icache = L1ICache(size='32kB', assoc=4, hit_latency=3, mshrs=32)
system.cpu.icache.replacement_policy = LRURP()

# L1 Data Cache configuration
system.cpu.dcache = L1DCache(size='32kB', assoc=4, hit_latency=3, mshrs=32)
system.cpu.dcache.replacement_policy = LRURP()

# L2 Cache configuration (LLC)
system.l2cache = L2Cache(size='256kB', assoc=16, hit_latency=9, mshrs=32)
system.l2cache.prefetcher = BestOffsetPrefetcher()  # Best Offset Prefetcher (BOP)
system.l2cache.replacement_policy = LRURP()

# Connect L1 caches to the CPU ports
system.cpu.icache_port = system.cpu.icache.cpu_side
system.cpu.dcache_port = system.cpu.dcache.cpu_side

# Connect the L1 caches to L2 cache
system.cpu.icache.mem_side = system.l2cache.cpu_side
system.cpu.dcache.mem_side = system.l2cache.cpu_side

# Connect L2 cache to the memory bus
system.l2cache.mem_side = system.membus.cpu_side_ports

# Interrupt controller
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_master = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_slave = system.membus.mem_side_ports

# System port setup
system.system_port = system.membus.cpu_side_ports

# Create a process and set it to the CPU workload
process = Process()
process.cmd = ['tests/test-progs/hello/bin/x86/linux/hello']  # Example workload
system.cpu.workload = process
system.cpu.createThreads()

# Root system and instantiate
root = Root(full_system=False, system=system)
m5.instantiate()

print("Starting simulation...")
exit_event = m5.simulate()
print('Exiting @ tick {} because {}'.format(m5.curTick(), exit_event.getCause()))