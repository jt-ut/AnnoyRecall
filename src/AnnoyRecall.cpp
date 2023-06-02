#include "../include/AnnoyRecall.hpp"
#include "../include/pybind11/pybind11.h"
#include "../include/pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(AnnoyRecall, m)
{
  // define all classes
  py::class_<VQRecall>(m, "VQRecall")
    // Constructor 
    .def(py::init<int, int, int>(), pybind11::arg("d"), pybind11::arg("nBMU")=int(2), pybind11::arg("nAnnoyTrees")=int(50)) 
    // Methods 
    .def("calc_BMU", &VQRecall::calc_BMU)
    //.def("calc_Recall", &VQRecall::calc_Recall)
    //.def("calc_RecallLabels", &VQRecall::calc_RecallLabels)
    .def("Recall", &VQRecall::Recall, pybind11::arg("X"), pybind11::arg("W"), pybind11::arg("XL")=XL_empty)
    // Attributes, set during construction 
    .def_readonly("d", &VQRecall::d)
    .def_readonly("nBMU", &VQRecall::nBMU)
    .def_readonly("nAnnoyTrees", &VQRecall::nAnnoyTrees)
    // Attributes, calc'd during method calls 
    .def_readonly("N", &VQRecall::N)
    .def_readonly("M", &VQRecall::M)
    .def_readonly("BMU", &VQRecall::BMU)
    .def_readonly("QE", &VQRecall::QE)
    .def_readonly("RF", &VQRecall::RF)
    .def_readonly("RF_Size", &VQRecall::RF_Size)
    .def_readonly("CADJi", &VQRecall::CADJi)
    .def_readonly("CADJj", &VQRecall::CADJj)
    .def_readonly("CADJ", &VQRecall::CADJ)
    .def_readonly("RFL_Dist", &VQRecall::RFL_Dist)
    .def_readonly("RFL", &VQRecall::RFL)
    .def_readonly("RFL_Purity", &VQRecall::RFL_Purity)
    .def_readonly("RFL_Purity_UOA", &VQRecall::RFL_Purity_UOA)
    .def_readonly("RFL_Purity_WOA", &VQRecall::RFL_Purity_WOA);
}

