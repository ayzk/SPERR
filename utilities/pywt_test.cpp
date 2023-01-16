#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "sperr_helper.h"
namespace py = pybind11;
int main(int argc, char* argv[])
{
  auto startT = std::chrono::high_resolution_clock::now();
  std::string HOME = "/Users/kzhao/";
  py::scoped_interpreter guard{};

  // append source dir to sys.path, and python interpreter would find your custom python file
  py::module_::import("sys").attr("path").attr("append")(HOME + "/code/sperr/utilities/");
  auto pycode = py::module_::import("pywt_wrapper");

  auto endT = std::chrono::high_resolution_clock::now();
  std::cout << "py prepare env time: " << (std::chrono::duration<double>(endT - startT)).count() << std::endl;

  auto ori_data = sperr::read_whole_file<float>(HOME + "/data/hurricane-100x500x500/Uf48.bin.dat");


  // wavelet
  startT = std::chrono::high_resolution_clock::now();
  py::array_t<float> ori_data_py({100, 500, 500}, ori_data.data());
  py::array_t<float> dwt_data = pycode.attr("dwt")(ori_data_py, "sym13");
  auto dwt_structure = pycode.attr("dwt_structure")().cast<std::string>();
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time for wavelet: " << (std::chrono::duration<double>(endT - startT)).count()
            << std::endl;

  // reverse wavelet
  startT = std::chrono::high_resolution_clock::now();
  py::array_t<float> idwt_data = pycode.attr("idwt")(dwt_data, py::bytes(dwt_structure), "sym13",
                                                     std::vector<size_t>({100, 500, 500}));
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time for reverse wavelet: "
            << (std::chrono::duration<double>(endT - startT)).count() << std::endl;

  // compare
  auto stat = sperr::calc_stats(ori_data.data(), idwt_data.data(), ori_data.size(), 1);
  printf("Stat: rmse = %f, linfty = %f, psnr = %fdB, orig_min = %f, orig_max = %f\n", stat[0],
         stat[1], stat[2], stat[3], stat[4]);
}
