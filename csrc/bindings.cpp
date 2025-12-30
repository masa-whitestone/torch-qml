#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "state_vector.h"
#include "adjoint_diff.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "torchqml C++ backend";
    
    // バージョン情報
    m.attr("__version__") = "0.2.0";
    m.attr("has_cuda") = true;
    
    // StateVector クラス
    py::class_<torchqml::StateVector>(m, "StateVector")
        .def(py::init<int, bool>(),
             py::arg("n_qubits"),
             py::arg("use_double") = false)
        .def("reset", &torchqml::StateVector::reset)
        .def("apply_gate", &torchqml::StateVector::apply_gate,
             py::arg("matrix"),
             py::arg("targets"),
             py::arg("controls") = std::vector<int>{},
             py::arg("adjoint") = false)
        .def("apply_h", &torchqml::StateVector::apply_h)
        .def("apply_x", &torchqml::StateVector::apply_x)
        .def("apply_y", &torchqml::StateVector::apply_y)
        .def("apply_z", &torchqml::StateVector::apply_z)
        .def("apply_s", &torchqml::StateVector::apply_s, py::arg("target"), py::arg("adjoint") = false)
        .def("apply_t", &torchqml::StateVector::apply_t, py::arg("target"), py::arg("adjoint") = false)
        .def("apply_rx", &torchqml::StateVector::apply_rx)
        .def("apply_ry", &torchqml::StateVector::apply_ry)
        .def("apply_rz", &torchqml::StateVector::apply_rz)
        .def("apply_cnot", &torchqml::StateVector::apply_cnot)
        .def("apply_cz", &torchqml::StateVector::apply_cz)
        .def("apply_swap", &torchqml::StateVector::apply_swap)
        .def("expectation_z", &torchqml::StateVector::expectation_z)
        .def("expectation_pauli", &torchqml::StateVector::expectation_pauli)
        .def("get_state", &torchqml::StateVector::get_state)
        .def("set_state", &torchqml::StateVector::set_state)
        .def_property_readonly("n_qubits", &torchqml::StateVector::n_qubits)
        .def_property_readonly("dim", &torchqml::StateVector::dim);
    
    // Adjoint Differentiation
    m.def("adjoint_differentiate", &torchqml::adjoint_differentiate,
          py::arg("params"),
          py::arg("operations"),
          py::arg("observable"),
          py::arg("n_qubits"),
          "Compute expectation values and gradients using adjoint method");
    
    // Forward only (勾配不要時)
    m.def("forward_only", &torchqml::forward_only,
          py::arg("params"),
          py::arg("operations"),
          py::arg("observable"),
          py::arg("n_qubits"));
    
    // GateOp 構造体
    py::class_<torchqml::GateOp>(m, "GateOp")
        .def(py::init<>())
        .def_readwrite("gate_type", &torchqml::GateOp::gate_type)
        .def_readwrite("targets", &torchqml::GateOp::targets)
        .def_readwrite("controls", &torchqml::GateOp::controls)
        .def_readwrite("param_index", &torchqml::GateOp::param_index);
    
    // GateType enum
    py::enum_<torchqml::GateType>(m, "GateType")
        .value("H", torchqml::GateType::H)
        .value("X", torchqml::GateType::X)
        .value("Y", torchqml::GateType::Y)
        .value("Z", torchqml::GateType::Z)
        .value("S", torchqml::GateType::S)
        .value("T", torchqml::GateType::T)
        .value("RX", torchqml::GateType::RX)
        .value("RY", torchqml::GateType::RY)
        .value("RZ", torchqml::GateType::RZ)
        .value("CNOT", torchqml::GateType::CNOT)
        .value("CZ", torchqml::GateType::CZ)
        .value("SWAP", torchqml::GateType::SWAP)
        .export_values();
        
    // PauliTerm
    py::class_<torchqml::PauliTerm>(m, "PauliTerm")
        .def(py::init<>())
        .def_readwrite("coefficient", &torchqml::PauliTerm::coefficient)
        .def_readwrite("paulis", &torchqml::PauliTerm::paulis);
}
