
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"
#include "request.h"
#include "scheduler.h"
#include "sequence.h"

namespace py = pybind11;
using namespace cottus;

PYBIND11_MODULE(_cottus_C, m) {
    m.doc() = "Cottus Runtime v0.1 C++ Backend";

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &EngineConfig::vocabSize)
        .def_readwrite("hidden_dim", &EngineConfig::hiddenDim)
        .def_readwrite("num_layers", &EngineConfig::numLayers)
        .def_readwrite("num_heads", &EngineConfig::numHeads)
        .def_readwrite("num_kv_heads", &EngineConfig::numKvHeads)
        .def_readwrite("head_dim", &EngineConfig::headDim)
        .def_readwrite("intermediate_dim", &EngineConfig::intermediateDim)
        .def_readwrite("max_seq_len", &EngineConfig::maxSeqLen)
        .def_readwrite("block_size", &EngineConfig::blockSize)
        .def_readwrite("rope_theta", &EngineConfig::ropeTheta)
        .def_readwrite("norm_epsilon", &EngineConfig::normEpsilon)
        .def_readwrite("device", &EngineConfig::device)
        .def_readwrite("dtype", &EngineConfig::dtype)
        .def_readwrite("eos_token_id", &EngineConfig::eosTokenId)
        .def_readwrite("eos_token_ids", &EngineConfig::eosTokenIds)
        .def_readwrite("temperature", &EngineConfig::temperature)
        .def_readwrite("top_k", &EngineConfig::topK)
        .def_readwrite("repetition_penalty", &EngineConfig::repetitionPenalty)
        .def_readwrite("frequency_penalty", &EngineConfig::frequencyPenalty)
        .def_readwrite("no_repeat_ngram_size", &EngineConfig::noRepeatNGramSize)
        .def_readwrite("stop_token_sequences", &EngineConfig::stopTokenSequences);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const EngineConfig&, const std::unordered_map<std::string, uintptr_t>&>(),
             py::arg("config"), py::arg("weight_ptrs"))
        .def("forward", &Engine::forward, py::arg("input_ids"))
        .def("generate", &Engine::generate, py::arg("input_ids"), py::arg("max_new_tokens"))
        .def("step", &Engine::step)
        .def("has_active_requests", &Engine::hasActiveRequests)
        .def("reset", &Engine::reset)
        .def("get_free_block_count", &Engine::getFreeBlockCount);
    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<const SchedulerConfig&>(), py::arg("config"))
        .def("add_sequence_group", [](Scheduler& self, int32_t req_id, 
             const std::vector<int32_t>& prompt, int32_t max_tokens, int32_t block_size,
             float temp, int32_t top_k)
             {
            self.addSequenceGroup(std::make_unique<SequenceGroup>(
                req_id, prompt, max_tokens, block_size, temp, top_k));
        }, py::arg("request_id"), py::arg("prompt_ids"), py::arg("max_tokens"),
           py::arg("block_size") = 16, py::arg("temperature") = 0.0f, py::arg("top_k") = 50)
        .def("schedule", &Scheduler::schedule)
        .def("has_pending_work", &Scheduler::hasPendingWork)
        .def("get_num_unfinished_seqs", &Scheduler::getNumUnfinishedSeqs);
}
