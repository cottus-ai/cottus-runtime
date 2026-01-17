#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "page_table.h"

namespace cottus {

enum class SequenceStatus {
    WAITING,
    RUNNING,
    SWAPPED,
    FINISHED
};

struct Sequence {
    int32_t seq_id;
    SequenceStatus status = SequenceStatus::WAITING;
    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> output_ids;
    int32_t current_pos = 0;
    int32_t stop_reason = 0;
    PageTable block_table;

    Sequence(int32_t id, const std::vector<int32_t>& prompt, int32_t block_size)
        : seq_id(id), prompt_ids(prompt), block_table(block_size) {}

    int32_t get_len() const {
        return static_cast<int32_t>(prompt_ids.size() + output_ids.size());
    }

    int32_t get_prompt_len() const {
        return static_cast<int32_t>(prompt_ids.size());
    }

    int32_t get_output_len() const {
        return static_cast<int32_t>(output_ids.size());
    }

    std::vector<int32_t> get_token_ids() const {
        std::vector<int32_t> ids = prompt_ids;
        ids.insert(ids.end(), output_ids.begin(), output_ids.end());
        return ids;
    }
};

struct SequenceGroup {
    int32_t request_id;
    std::vector<std::unique_ptr<Sequence>> seqs;
    int32_t max_tokens;
    float temperature;
    int32_t top_k;
    bool is_prefill = true;

    SequenceGroup(int32_t req_id, const std::vector<int32_t>& prompt, 
                  int32_t max_toks, int32_t block_size, float temp = 0.0f, int32_t k = 50)
        : request_id(req_id), max_tokens(max_toks), temperature(temp), top_k(k) {
        seqs.push_back(std::make_unique<Sequence>(req_id, prompt, block_size));
    }

    Sequence* get_seq() {
        return seqs.empty() ? nullptr : seqs[0].get();
    }

    const Sequence* get_seq() const {
        return seqs.empty() ? nullptr : seqs[0].get();
    }

    bool is_finished() const {
        for (const auto& seq : seqs) {
            if (seq->status != SequenceStatus::FINISHED) return false;
        }
        return true;
    }

    SequenceStatus get_status() const {
        if (seqs.empty()) return SequenceStatus::FINISHED;
        return seqs[0]->status;
    }
};

}
