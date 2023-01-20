#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include "sperr_helper.h"
namespace py = pybind11;
//py::module_ pywavelet;

template <class T>
void wavelet(T* data, std::vector<size_t>& dims)
{
  auto startT = std::chrono::high_resolution_clock::now();
  auto pywavelet = py::module_::import("pywt_wrapper");
  auto endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time for import python module: " << (std::chrono::duration<double>(endT - startT)).count()
            << std::endl;

  // wavelet
  startT = std::chrono::high_resolution_clock::now();
  py::array_t<float> ori_data_py(dims, data);
  py::array_t<float> dwt_data = pywavelet.attr("dwt")(ori_data_py, "sym13");
  auto dwt_structure = pywavelet.attr("dwt_structure")().cast<std::string>();
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time for wavelet: " << (std::chrono::duration<double>(endT - startT)).count()
            << std::endl;

  // reverse wavelet
  startT = std::chrono::high_resolution_clock::now();
  py::array_t<float> idwt_data = pywavelet.attr("idwt")(dwt_data, py::bytes(dwt_structure), "sym13",
                                                        std::vector<size_t>({100, 500, 500}));
  endT = std::chrono::high_resolution_clock::now();
  std::cout << "Time for reverse wavelet: "
            << (std::chrono::duration<double>(endT - startT)).count() << std::endl;

  // compare
  size_t num = std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  auto stat = sperr::calc_stats(data, idwt_data.data(), num, 1);
  printf("Stat: rmse = %f, linfty = %f, psnr = %fdB, orig_min = %f, orig_max = %f\n", stat[0],
         stat[1], stat[2], stat[3], stat[4]);
}

int main(int argc, char* argv[])
{
  std::string HOME = "/Users/kzhao/";
  py::scoped_interpreter guard{};

  // append source dir to sys.path, and python interpreter would find your custom python file
  py::module_::import("sys").attr("path").attr("append")(HOME + "/code/sperr/utilities/");

  {
    printf("\n\nhurricane-100x500x500/Uf48.bin.dat\n");
    auto data = sperr::read_whole_file<float>(HOME + "/data/hurricane-100x500x500/Uf48.bin.dat");
    std::vector<size_t> dims({100, 500, 500});
    wavelet(data.data(), dims);
  }
  {
    printf("\n\nhurricane-100x500x500/Pf48.bin.dat\n");
    auto data = sperr::read_whole_file<float>(HOME + "/data/hurricane-100x500x500/Pf48.bin.dat");
    std::vector<size_t> dims({100, 500, 500});
    wavelet(data.data(), dims);
  }
  {
    printf("\n\nhurricane-100x500x500/Vf48.bin.dat\n");
    auto data = sperr::read_whole_file<float>(HOME + "/data/hurricane-100x500x500/Vf48.bin.dat");
    std::vector<size_t> dims({100, 500, 500});
    wavelet(data.data(), dims);
  }
  {
    printf("\n\nhurricane-100x500x500/Wf48.bin.dat\n");
    auto data = sperr::read_whole_file<float>(HOME + "/data/hurricane-100x500x500/Wf48.bin.dat");
    std::vector<size_t> dims({100, 500, 500});
    wavelet(data.data(), dims);
  }
}
