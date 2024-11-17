#ifndef __MEM_CACHE_REPLACEMENT_POLICIES_LRU8_RP_HH__
#define __MEM_CACHE_REPLACEMENT_POLICIES_LRU8_RP_HH__

#include "mem/cache/replacement_policies/base.hh"

namespace gem5
{

struct LRU8RPParams;

namespace replacement_policy
{

class LRU8 : public Base
{
  protected:
    /** LRU8-specific implementation of replacement data. */
    struct LRU8ReplData : ReplacementData
    {
        /** Tick on which the entry was last touched. */
        Tick lastTouchTick;

        /** Default constructor. Invalidate data. */
        LRU8ReplData() : lastTouchTick(0) {}
    };

  public:
    typedef LRU8RPParams Params;
    LRU8(const Params &p);
    ~LRU8() = default;

    void invalidate(const std::shared_ptr<ReplacementData>& replacement_data) override;
    void touch(const std::shared_ptr<ReplacementData>& replacement_data) const override;
    void reset(const std::shared_ptr<ReplacementData>& replacement_data) const override;
    ReplaceableEntry* getVictim(const ReplacementCandidates& candidates) const override;
    std::shared_ptr<ReplacementData> instantiateEntry() override;
};

} // namespace replacement_policy
} // namespace gem5

#endif // __MEM_CACHE_REPLACEMENT_POLICIES_LRU8_RP_HH__
