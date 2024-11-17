#include "mem/cache/replacement_policies/lru8_rp.hh"

#include <cassert>
#include <memory>
#include <vector>
#include <algorithm>

#include "params/LRU8RP.hh"
#include "sim/cur_tick.hh"

namespace gem5
{

    namespace replacement_policy
    {

        LRU8::LRU8(const Params &p)
            : Base(p)
        {
        }

        void
        LRU8::invalidate(const std::shared_ptr<ReplacementData> &replacement_data)
        {
            std::static_pointer_cast<LRU8ReplData>(
                replacement_data)
                ->lastTouchTick = Tick(0);
        }

        void
        LRU8::touch(const std::shared_ptr<ReplacementData> &replacement_data) const
        {
            std::static_pointer_cast<LRU8ReplData>(
                replacement_data)
                ->lastTouchTick = curTick();
        }

        void
        LRU8::reset(const std::shared_ptr<ReplacementData> &replacement_data) const
        {
            std::static_pointer_cast<LRU8ReplData>(
                replacement_data)
                ->lastTouchTick = curTick();
        }

        ReplaceableEntry *
        LRU8::getVictim(const ReplacementCandidates &candidates) const
        {
            assert(candidates.size() > 0);

            // Select the least-recently-used 8 entries
            std::vector<ReplaceableEntry *> sorted_candidates;
            for (const auto &candidate : candidates)
            {
                sorted_candidates.push_back(candidate);
            }

            // Sort candidates by last touch time
            std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                      [](ReplaceableEntry *a, ReplaceableEntry *b)
                      {
                          return std::static_pointer_cast<LRU8ReplData>(
                                     a->replacementData)
                                     ->lastTouchTick <
                                 std::static_pointer_cast<LRU8ReplData>(
                                     b->replacementData)
                                     ->lastTouchTick;
                      });

            // Select the least-recently-used entry from the first 8 entries
            ReplaceableEntry *victim = sorted_candidates[0];
            for (size_t i = 1; i < std::min(sorted_candidates.size(), size_t(8)); ++i)
            {
                if (std::static_pointer_cast<LRU8ReplData>(
                        sorted_candidates[i]->replacementData)
                        ->lastTouchTick <
                    std::static_pointer_cast<LRU8ReplData>(
                        victim->replacementData)
                        ->lastTouchTick)
                {
                    victim = sorted_candidates[i];
                }
            }

            return victim;
        }

        std::shared_ptr<ReplacementData>
        LRU8::instantiateEntry()
        {
            return std::make_shared<LRU8ReplData>();
        }

    } // namespace replacement_policy
} // namespace gem5
