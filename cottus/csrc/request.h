#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include "logits_processor.h"

namespace cottus {

struct Request {
    enum class State {
        WAITING,
        RUNNING,
        FINISHED
    };
    int id;
    State state = State::WAITING;
    int max_tokens;
    bool ignore_eos;
    float temperature;
    int top_k;
    std::vector<std::unique_ptr<LogitsProcessor>> logits_processors;

    std::vector<int32_t> prompt_ids;
    std::vector<int32_t> generated_ids;
    int current_pos = 0;
    int stop_reason = 0;
    Request(int id_, const std::vector<int32_t>& prompt_, int max_tokens_) 
        : id(id_), prompt_ids(prompt_), max_tokens(max_tokens_) {}
    
    Request(Request&&) = default;
    Request& operator=(Request&&) = default;
    
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;

    std::vector<int32_t> get_context() const {
        std::vector<int32_t> ctx = prompt_ids;
        ctx.insert(ctx.end(), generated_ids.begin(), generated_ids.end());
        return ctx;
    }
};

} // namespace cottus
