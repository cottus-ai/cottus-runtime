#pragma once
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>

namespace cottus {

class LogitsProcessor {
public:
    virtual void apply(const std::vector<int32_t>& input_ids, std::vector<float>& logits) = 0;
    virtual ~LogitsProcessor() = default;
};
class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
public:
    RepetitionPenaltyLogitsProcessor(float penalty) : penalty_(penalty) {}

    void apply(const std::vector<int32_t>& input_ids, std::vector<float>& logits) override {
        if (penalty_ <= 1.0f) return;
        for (int32_t token : input_ids) {
            if (token >= 0 && token < static_cast<int32_t>(logits.size())) {
                float& val = logits[token];
                val = (val < 0.0f) ? (val * penalty_) : (val / penalty_);
            }
        }
    }

private:
    float penalty_;
};
class NoRepeatNGramLogitsProcessor : public LogitsProcessor {
public:
    NoRepeatNGramLogitsProcessor(int ngram_size) : ngram_size_(ngram_size) {}

    void apply(const std::vector<int32_t>& input_ids, std::vector<float>& logits) override
    {
        if (ngram_size_ <= 0 || input_ids.size() < static_cast<size_t>(ngram_size_ - 1))
        {
            return;
        }
        int prefix_len = ngram_size_ - 1;
        auto prefix_start = input_ids.end() - prefix_len;
        for (size_t i = 0; i <= input_ids.size() - ngram_size_; ++i)
        {
            bool match = true;
            for (int k = 0; k < prefix_len; ++k)
            {
                if (input_ids[i + k] != *(prefix_start + k))
                {
                    match = false;
                    break;
                }
            }
            
            if (match)
            {
                int32_t banned_token = input_ids[i + prefix_len];
                if (banned_token >= 0 && banned_token < static_cast<int32_t>(logits.size()))
                {
                    logits[banned_token] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

private:
    int ngram_size_;
};

// Applies a penalty based on token frequency in the context.
// Invariant: "Vocabulary Diversity"
class FrequencyPenaltyLogitsProcessor : public LogitsProcessor {
public:
    FrequencyPenaltyLogitsProcessor(float frequency_penalty) : penalty_(frequency_penalty) {}

    void apply(const std::vector<int32_t>& input_ids, std::vector<float>& logits) override {
        if (penalty_ == 0.0f) return;

        // Count frequencies - O(N) scan
        // For production, we should maintain a running count map in the Engine.
        // For now, we scan.
        std::unordered_map<int32_t, int> counts;
        for (int32_t token : input_ids) {
            counts[token]++;
        }

        for (const auto& [token, count] : counts) {
            if (token >= 0 && token < static_cast<int32_t>(logits.size())) {
                logits[token] -= (penalty_ * count);
            }
        }
    }

private:
    float penalty_;
};

} // namespace cottus
