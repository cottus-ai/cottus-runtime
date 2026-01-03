#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.h"
#include "generic_transformer.h"

namespace py = pybind11;

using namespace cottus;

PYBIND11_MODULE(_cottus_C, m) {
    m.doc() = "Cottus Runtime v0.1 C++ Backend";
    
    // Expose EngineConfig
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
        .def_readwrite("dtype", &EngineConfig::dtype);
    
    // Expose Engine
    py::class_<Engine>(m, "Engine")
        .def(py::init<const EngineConfig&, const std::unordered_map<std::string, uintptr_t>&>(),
             py::arg("config"),
             py::arg("weight_ptrs"))
        .def("generate", &Engine::generate,
             py::arg("input_ids"),
             py::arg("max_new_tokens"),
             "Generate tokens using greedy decoding")
        .def("reset", &Engine::reset)
        .def("get_free_block_count", &Engine::getFreeBlockCount);
}
