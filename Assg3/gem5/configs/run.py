
# system.cpu.switchTo(AtomicSimpleCPU())

# # Warm-up the CPU
# print("Warming up for {} instructions".format(max_insts_warmup))
# m5.stats.reset()
# exit_event = m5.simulate(max_insts_warmup)
# print("Exiting warm-up @ tick {} because {}".format(m5.curTick(), exit_event.getCause()))

# # Switch the CPU to detailed mode
# system.cpu.switchTo(X86O3CPU())

# # Detailed simulation
# print("Running detailed simulation for {} instructions".format(max_insts_detailed))
# m5.stats.reset()
# exit_event = m5.simulate(max_insts_detailed)
# print("Exiting detailed simulation @ tick {} because {}".format(m5.curTick(), exit_event.getCause()))
# # import the m5 (gem5) library created when gem5 is built

import m5

# import all of the SimObjects
from m5.objects import *
# from gem5.runtime import get_runtime_isa
from m5.objects.BranchPredictor import *

# import the SimpleOpts module
from common import SimpleOpts

# Default to running 'hello', use the compiled ISA to find the binary
# grab the specific path to the binary
thispath = os.path.dirname(os.path.realpath(__file__))
default_binary = os.path.join(
    # "/home/arunnats/cpu2006/benchspec/CPU2006/462.libquantum/exe/libquantum_base.gcc43-64bit",
    "/home/arunnats/cpu2006/benchspec/CPU2006/482.sphinx3/exe/sphinx_livepretend_base.gcc43-64bit",
    # "/home/arunnats/cpu2006/benchspec/CPU2006/429.mcf/exe/mcf_base.gcc43-64bit"
    
)

class L1Cache(Cache):
    assoc = 4
    tag_latency = 3
    data_latency = 3
    response_latency = 3
    mshrs = 32
    tgts_per_mshr = 20
    replacement_policy = LRURP()

    def connectBus(self, bus):
        """Connect this cache to a memory-side bus"""
        self.mem_side = bus.cpu_side_ports

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU-side port
        This must be defined in a subclass"""
        raise NotImplementedError


class L1ICache(L1Cache):
    size = '1kB'
    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU icache port"""
        self.cpu_side = cpu.icache_port

class L1DCache(L1Cache):
    size = '1kB'
    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU dcache port"""
        self.cpu_side = cpu.dcache_port

class L2Cache(Cache):
    size = '4kB'
    assoc = 16
    tag_latency = 9
    data_latency = 9
    response_latency = 9
    mshrs = 32
    tgts_per_mshr = 12
    # replacement_policy = LRURP()
    # replacement_policy = MRURP()
    # replacement_policy = FIFORP()
    # replacement_policy = SecondChanceRP()
    replacement_policy = LRU8RP()
    prefetcher = BOPPrefetcher()


    def connectCPUSideBus(self, bus):
        self.cpu_side = bus.mem_side_ports

    def connectMemSideBus(self, bus):
        self.mem_side = bus.cpu_side_ports

#! Set the maximum number of instructions for warm-up
max_insts_warmup = 50_000_000

#! Set the maximum number of instructions for detailed simulation
max_insts_detailed = 50_000_000

# Set the CPU to run in atomic mode for warm-up


# create the system we are going to simulate
system = System()

# Set the clock frequency of the system (and all of its children)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "3GHz"
system.clk_domain.voltage_domain = VoltageDomain()

# Set up the system
system.mem_mode = "timing"  # Use timing accesses
system.mem_ranges = [AddrRange("4GB")]  # Create an address range


system.cpu = X86O3CPU()

# Create a branch predictor
system.cpu.branchPred = TAGE_SC_L_64KB()

# Set the fetch, decode, issue, and commit width
system.cpu.fetchWidth = 2
system.cpu.decodeWidth = 2
system.cpu.issueWidth = 2
system.cpu.commitWidth = 2

# ?Set the IQ, LSQ, and ROB sizes
system.cpu.numIQEntries = 64
system.cpu.LQEntries = 32
system.cpu.SQEntries = 32
system.cpu.numROBEntries = 192

# Set the BTB and RAS sizes
# system.cpu.branchPred.BTBEntries = 4096
# system.cpu.branchPred.RASSize = 32

# Create a simple CPU

# Create an L1 instruction and data cache
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

# Connect the instruction and data caches to the CPU
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# Create a memory bus, a coherent crossbar, in this case
system.l2bus = L2XBar()

# Hook the CPU ports up to the l2bus
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

# Create an L2 cache and connect it to the l2bus
system.l2cache = L2Cache()
system.l2cache.connectCPUSideBus(system.l2bus)

# Create a memory bus
system.membus = SystemXBar()

# Connect the L2 cache to the membus
system.l2cache.connectMemSideBus(system.membus)

# create the interrupt controller for the CPU
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Connect the system up to the membus
system.system_port = system.membus.cpu_side_ports

# Create a DDR3 memory controller
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.workload = SEWorkload.init_compatible(default_binary)

# Create a process for a simple "Hello World" application
process = Process()
# Set the command
# cmd is a list which begins with the executable (like argv)
process.cmd = [default_binary]
# Set the cpu to use the process as its workload and create thread contexts
system.cpu.workload = process
system.cpu.createThreads()

# set up the root SimObject and start the simulation
root = Root(full_system=False, system=system)
# instantiate all of the objects we've created above

print("instantiating")
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print("Exiting @ tick %i because %s" % (m5.curTick(), exit_event.getCause()))