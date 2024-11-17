from m5.params import *
from m5.objects import BaseReplacementPolicy

class LRU8RP(BaseReplacementPolicy):
    type = 'LRU8RP'
    cxx_header = "mem/cache/replacement_policies/lru8_rp.hh"
