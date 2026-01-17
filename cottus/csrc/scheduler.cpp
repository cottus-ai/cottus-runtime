#include "scheduler.h"
#include <algorithm>

namespace cottus {

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config) {
    gpu_allocator_ = std::make_unique<BlockAllocator>(config_.gpuBlocks, config_.blockSize);
    
    int32_t blockSizeBytes = config_.blockSizeBytes;
    if (blockSizeBytes <= 0) {
        blockSizeBytes = config_.blockSize * 64 * 2 * sizeof(uint16_t);
    }
    swap_space_ = std::make_unique<SwapSpace>(config_.cpuSwapBlocks, blockSizeBytes);
}

void Scheduler::addSequenceGroup(std::unique_ptr<SequenceGroup> seq_group) {
    if (!seq_group) return;
    waiting_.push_back(std::move(seq_group));
}

bool Scheduler::canAllocate(const SequenceGroup& seq_group) const {
    int32_t required_blocks = 1;
    if (seq_group.is_prefill) {
        Sequence* seq = seq_group.seqs[0].get();
        required_blocks = (seq->get_prompt_len() + config_.blockSize - 1) / config_.blockSize;
    }
    return gpu_allocator_->numFreeBlocks() >= required_blocks;
}

void Scheduler::allocateBlocks(SequenceGroup& seq_group) {
    Sequence* seq = seq_group.get_seq();
    if (!seq) return;

    int32_t current_blocks = seq->block_table.getBlockCount();
    int32_t current_len = seq->current_pos;
    int32_t required_blocks = (current_len + config_.blockSize) / config_.blockSize;

    while (current_blocks < required_blocks) {
        int32_t block_id = gpu_allocator_->allocateBlock();
        seq->block_table.appendBlock(block_id);
        current_blocks++;
    }
}

void Scheduler::freeBlocks(SequenceGroup& seq_group) {
    for (auto& seq : seq_group.seqs) {
        const auto& blocks = seq->block_table.getBlockIds();
        for (int32_t block_id : blocks) {
            gpu_allocator_->freeBlock(block_id);
        }
        seq->block_table = PageTable(config_.blockSize);
    }
}

void Scheduler::freeSwapBlocks(int32_t request_id) {
    auto it = swap_mappings_.find(request_id);
    if (it != swap_mappings_.end()) {
        for (auto& pair : it->second.gpu_to_cpu) {
            swap_space_->freeCpuBlock(pair.second);
        }
        swap_mappings_.erase(it);
    }
}

void Scheduler::executeSwapOut(SequenceGroup& sg, void* gpuKvCache) {
    Sequence* seq = sg.get_seq();
    if (!seq) return;

    SwapMapping mapping;
    const auto& gpu_blocks = seq->block_table.getBlockIds();
    
    int32_t blockSizeBytes = config_.blockSizeBytes;
    if (blockSizeBytes <= 0) {
        blockSizeBytes = config_.blockSize * 64 * 2 * sizeof(uint16_t);
    }

    for (int32_t gpu_block : gpu_blocks) {
        int32_t cpu_block = swap_space_->allocateCpuBlock();
        swap_space_->swapOut(gpu_block, cpu_block, gpuKvCache, blockSizeBytes);
        mapping.gpu_to_cpu[gpu_block] = cpu_block;
        gpu_allocator_->freeBlock(gpu_block);
    }

    swap_mappings_[sg.request_id] = mapping;
    seq->block_table = PageTable(config_.blockSize);

    for (auto& s : sg.seqs) {
        if (s->status == SequenceStatus::RUNNING) {
            s->status = SequenceStatus::SWAPPED;
        }
    }
}

void Scheduler::executeSwapIn(SequenceGroup& sg, void* gpuKvCache) {
    auto it = swap_mappings_.find(sg.request_id);
    if (it == swap_mappings_.end()) return;

    Sequence* seq = sg.get_seq();
    if (!seq) return;

    int32_t blockSizeBytes = config_.blockSizeBytes;
    if (blockSizeBytes <= 0) {
        blockSizeBytes = config_.blockSize * 64 * 2 * sizeof(uint16_t);
    }

    std::vector<std::pair<int32_t, int32_t>> sorted_mappings;
    for (auto& pair : it->second.gpu_to_cpu) {
        sorted_mappings.push_back(pair);
    }
    std::sort(sorted_mappings.begin(), sorted_mappings.end());

    for (auto& pair : sorted_mappings) {
        int32_t new_gpu_block = gpu_allocator_->allocateBlock();
        swap_space_->swapIn(pair.second, new_gpu_block, gpuKvCache, blockSizeBytes);
        seq->block_table.appendBlock(new_gpu_block);
        swap_space_->freeCpuBlock(pair.second);
    }

    swap_mappings_.erase(it);

    for (auto& s : sg.seqs) {
        if (s->status == SequenceStatus::SWAPPED) {
            s->status = SequenceStatus::RUNNING;
        }
    }
}

SchedulerOutputs Scheduler::schedule() {
    SchedulerOutputs outputs;

    for (auto it = running_.begin(); it != running_.end(); ) {
        SequenceGroup* sg = it->get();
        if (sg->is_finished()) {
            freeBlocks(*sg);
            freeSwapBlocks(sg->request_id);
            finished_.push_back(std::move(*it));
            it = running_.erase(it);
        } else {
            outputs.scheduled_seq_groups.push_back(sg);
            outputs.num_decode_groups++;
            ++it;
        }
    }

    while (!swapped_.empty() && canAllocate(*swapped_.front())) {
        auto sg = std::move(swapped_.front());
        swapped_.pop_front();
        outputs.swapped_in.push_back(sg.get());
        running_.push_back(std::move(sg));
    }

    while (!waiting_.empty()) {
        SequenceGroup* sg = waiting_.front().get();

        if (!canAllocate(*sg)) {
            if (!running_.empty()) {
                auto victim = std::move(running_.back());
                running_.pop_back();
                outputs.swapped_out.push_back(victim.get());
                swapped_.push_back(std::move(victim));
                continue;
            } else {
                break;
            }
        }

        Sequence* seq = sg->get_seq();
        if (seq) {
            seq->status = SequenceStatus::RUNNING;
        }
        sg->is_prefill = true;

        allocateBlocks(*sg);
        outputs.scheduled_seq_groups.push_back(sg);
        outputs.num_prefill_groups++;

        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();

        if (static_cast<int32_t>(outputs.scheduled_seq_groups.size()) >= config_.maxBatchSize) {
            break;
        }
    }

    return outputs;
}

std::vector<std::unique_ptr<SequenceGroup>> Scheduler::getFinishedGroups() {
    std::vector<std::unique_ptr<SequenceGroup>> result;
    result.swap(finished_);
    return result;
}

bool Scheduler::hasPendingWork() const {
    return !waiting_.empty() || !running_.empty() || !swapped_.empty();
}

int32_t Scheduler::getNumUnfinishedSeqs() const {
    int32_t count = 0;
    for (const auto& sg : waiting_) count += static_cast<int32_t>(sg->seqs.size());
    for (const auto& sg : running_) count += static_cast<int32_t>(sg->seqs.size());
    for (const auto& sg : swapped_) count += static_cast<int32_t>(sg->seqs.size());
    return count;
}

}
