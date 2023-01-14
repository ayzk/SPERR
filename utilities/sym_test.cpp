#include "SYM.h"
#include "CDF97.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::cerr << "Usage: ./a.out input_filename dim_0 dim_1 dim_2"
              << "example: ./a.out rtm 449 449 235"
              << "example: ./a.out 2d 10 10 1" << std::endl;
    return 1;
  }

  const char* input = argv[1];
  const size_t dim_0 = std::atoi(argv[2]);
  const size_t dim_1 = std::atoi(argv[3]);
  const size_t dim_2 = std::atoi(argv[4]);

  auto in_buf = sperr::read_whole_file<float>(input);

  sperr::SYM sym;
  sperr::dims_type dims({dim_0, dim_1, dim_2});
  sym.copy_data(in_buf.data(), in_buf.size(), dims);

  sperr::CDF97 cdf;
  cdf.copy_data(in_buf.data(), in_buf.size(), dims);

  {
    const auto startT = std::chrono::high_resolution_clock::now();
    cdf.dwt3d();
    const auto endT = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diffT = endT - startT;
    std::cout << "[CDF97] Time for transforms: " << diffT.count() << std::endl;
  }
  {
    const auto startT = std::chrono::high_resolution_clock::now();
    cdf.idwt3d();
    const auto endT = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diffT = endT - startT;
    std::cout << "[CDF97] Time for reverse transforms: " << diffT.count() << std::endl;
  }
  std::vector<float> output_cdf(cdf.view_data().data(),
                                cdf.view_data().data() + cdf.view_data().size());
  auto stat_cdf = sperr::calc_stats(in_buf.data(), output_cdf.data(), in_buf.size(), 1);
  printf("Stat: rmse = %f, linfty = %f, psnr = %fdB, orig_min = %f, orig_max = %f\n", stat_cdf[0],
         stat_cdf[1], stat_cdf[2], stat_cdf[3], stat_cdf[4]);

  {
    const auto startT = std::chrono::high_resolution_clock::now();
    sym.dwt3d();
    const auto endT = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diffT = endT - startT;
    std::cout << "[SYM13] Time for transforms: " << diffT.count() << std::endl;
  }
  {
    const auto startT = std::chrono::high_resolution_clock::now();
    sym.idwt3d();
    const auto endT = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diffT = endT - startT;
    std::cout << "[SYM13] Time for reverse transforms: " << diffT.count() << std::endl;
  }
  std::vector<float> output_sym(sym.view_data().data(),
                            sym.view_data().data() + sym.view_data().size());
  auto stat = sperr::calc_stats(in_buf.data(), output_sym.data(), in_buf.size(), 1);
  printf("Stat: rmse = %f, linfty = %f, psnr = %fdB, orig_min = %f, orig_max = %f\n", stat[0],
         stat[1], stat[2], stat[3], stat[4]);
}
