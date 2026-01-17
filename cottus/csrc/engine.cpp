#include "engine.h"
#include "scheduler.h"
#include "generic_transformer.h"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace cottus {

Engine::Engine(const EngineConfig& config, const std::unordered_map<std::string, uintptr_t>& weightPtrs)
    : config_(config)
{
    if (config.blockSize <= 0) throw std::invalid_argument("blockSize must be positive");
    if (config.maxSeqLen <= 0) throw std::invalid_argument("maxSeqLen must be positive");

    SchedulerConfig schedConfig;
    schedConfig.maxBatchSize = 64;
    schedConfig.blockSize = config.blockSize;
    scheduler_ = std::make_unique<Scheduler>(schedConfig);

    transformer_ = std::make_unique<GenericTransformer>(config, weightPtrs);

    int32_t totalBlocks = scheduler_->getBlockAllocator().totalBlocks();
    int32_t elementsPerLayerKV = config.blockSize * config.numKvHeads * config.headDim;
    int32_t elementsPerBlock = 2 * elementsPerLayerKV * config.numLayers;
    kvCache_.resize(totalBlocks * elementsPerBlock, 0);

    if (config.device == "cuda") {
        size_t sizeBytes = kvCache_.size() * sizeof(uint16_t);
        cudaError_t err = cudaMalloc(&d_kvCache_, sizeBytes);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate GPU KV cache");
        err = cudaMemset(d_kvCache_, 0, sizeBytes);
        if (err != cudaSuccess) {
            cudaFree(d_kvCache_);
            throw std::runtime_error("Failed to initialize GPU KV cache");
        }
    }
}

Engine::~Engine() {
    if (d_kvCache_) cudaFree(d_kvCache_);
}

std::vector<int32_t> Engine::forward(const std::vector<int32_t>& inputIds) {
    return generate(inputIds, 0);
}

void Engine::addRequest(std::unique_ptr<Request> req) {
    if (!req) return;

    auto sg = std::make_unique<SequenceGroup>(
        req->id,
        req->prompt_ids,
        req->max_tokens,
        config_.blockSize,
        req->temperature,
        req->top_k
    );
    scheduler_->addSequenceGroup(std::move(sg));
}

void Engine::step() {
    if (!scheduler_->hasPendingWork()) return;

    SchedulerOutputs outputs = scheduler_->schedule();

    for (SequenceGroup* sg : outputs.swapped_out) {
        scheduler_->executeSwapOut(*sg, d_kvCache_ ? d_kvCache_ : kvCache_.data());
    }

    for (SequenceGroup* sg : outputs.swapped_in) {
        scheduler_->executeSwapIn(*sg, d_kvCache_ ? d_kvCache_ : kvCache_.data());
    }

    for (SequenceGroup* sg : outputs.scheduled_seq_groups) {
        if (!sg) continue;
        Sequence* seq = sg->get_seq();
        if (!seq || seq->status == SequenceStatus::FINISHED) continue;

        if (sg->is_prefill) {
            while (seq->current_pos < static_cast<int32_t>(seq->prompt_ids.size())) {
                int32_t inputToken = seq->prompt_ids[seq->current_pos];

                if (seq->current_pos % config_.blockSize == 0) {
                    int32_t blockId = scheduler_->getBlockAllocator().allocateBlock();
                    seq->block_table.appendBlock(blockId);
                }

                std::vector<float> logits = transformer_->forwardToken(
                    inputToken,
                    seq->current_pos,
                    seq->block_table,
                    reinterpret_cast<uintptr_t>(kvCache_.data()),
                    d_kvCache_,
                    config_.device
                );
                seq->current_pos++;

                if (seq->current_pos >= static_cast<int32_t>(seq->prompt_ids.size())) {
                    int32_t nextToken = static_cast<int32_t>(
                        std::max_element(logits.begin(), logits.end()) - logits.begin()
                    );
                    seq->output_ids.push_back(nextToken);

                    if (nextToken == config_.eosTokenId || nextToken == 2) {
                        seq->status = SequenceStatus::FINISHED;
                        seq->stop_reason = 2;
                    }
                    if (static_cast<int32_t>(seq->output_ids.size()) >= sg->max_tokens) {
                        seq->status = SequenceStatus::FINISHED;
                        seq->stop_reason = 1;
                    }
                }
            }
            sg->is_prefill = false;
        } else {
            int32_t inputToken = seq->output_ids.back();

            if (seq->current_pos % config_.blockSize == 0) {
                int32_t blockId = scheduler_->getBlockAllocator().allocateBlock();
                seq->block_table.appendBlock(blockId);
            }

            std::vector<float> logits = transformer_->forwardToken(
                inputToken,
                seq->current_pos,
                seq->block_table,
                reinterpret_cast<uintptr_t>(kvCache_.data()),
                d_kvCache_,
                config_.device
            );
            seq->current_pos++;

            int32_t nextToken;
            if (sg->temperature <= 0.0f) {
                nextToken = static_cast<int32_t>(
                    std::max_element(logits.begin(), logits.end()) - logits.begin()
                );
            } else {
                std::vector<std::pair<float, int32_t>> indexed(logits.size());
                for (size_t i = 0; i < logits.size(); ++i) {
                    indexed[i] = {logits[i], static_cast<int32_t>(i)};
                }
                std::partial_sort(indexed.begin(), indexed.begin() + sg->top_k, indexed.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                float maxLogit = indexed[0].first;
                std::vector<float> probs(sg->top_k);
                float sumExp = 0.0f;
                for (int32_t i = 0; i < sg->top_k; ++i) {
                    probs[i] = std::exp((indexed[i].first - maxLogit) / sg->temperature);
                    sumExp += probs[i];
                }
                static std::mt19937 rng(std::random_device{}());
                std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
                nextToken = indexed[dist(rng)].second;
            }

            seq->output_ids.push_back(nextToken);

            if (nextToken == config_.eosTokenId || nextToken == 2) {
                seq->status = SequenceStatus::FINISHED;
                seq->stop_reason = 2;
            }
            if (static_cast<int32_t>(seq->output_ids.size()) >= sg->max_tokens) {
                seq->status = SequenceStatus::FINISHED;
                seq->stop_reason = 1;
            }
        }
    }
}

std::vector<int32_t> Engine::generate(const std::vector<int32_t>& inputIds, int32_t maxNewTokens) {
    if (inputIds.empty()) throw std::invalid_argument("inputIds empty");
    if (maxNewTokens <= 0) return {};

    int reqId = nextRequestId_++;
    auto req = std::make_unique<Request>(reqId, inputIds, maxNewTokens);
    req->temperature = config_.temperature;
    req->top_k = config_.topK;

    addRequest(std::move(req));

    std::vector<int32_t> result;
    while (scheduler_->hasPendingWork()) {
        step();

        auto finished = scheduler_->getFinishedGroups();
        for (auto& sg : finished) {
            if (sg->request_id == reqId) {
                Sequence* seq = sg->get_seq();
                if (seq) {
                    result = seq->output_ids;
                }
                break;
            }
        }
        if (!result.empty()) break;
    }

    return result;
}

void Engine::reset() {
    SchedulerConfig schedConfig;
    schedConfig.maxBatchSize = 64;
    schedConfig.blockSize = config_.blockSize;
    scheduler_ = std::make_unique<Scheduler>(schedConfig);
}

int32_t Engine::getFreeBlockCount() const {
    return scheduler_->getBlockAllocator().numFreeBlocks();
}

std::vector<std::unique_ptr<Request>> Engine::pullFinishedRequests() {
    std::vector<std::unique_ptr<Request>> result;
    auto finished = scheduler_->getFinishedGroups();
    for (auto& sg : finished) {
        Sequence* seq = sg->get_seq();
        if (!seq) continue;

        auto req = std::make_unique<Request>(sg->request_id, seq->prompt_ids, sg->max_tokens);
        req->generated_ids = seq->output_ids;
        req->state = Request::State::FINISHED;
        req->stop_reason = seq->stop_reason;
        result.push_back(std::move(req));
    }
    return result;
}

bool Engine::hasActiveRequests() const {
    return scheduler_->hasPendingWork();
}

}
