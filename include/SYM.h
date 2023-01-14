
#ifndef SYM_H
#define SYM_H

#include "sperr_helper.h"

#include <cmath>
#define TYPE double
namespace sperr {

class SYM {
 public:
  //
  // Input
  //
  // Note that copy_data() and take_data() effectively resets internal states of this class.
  template <typename T>
  auto copy_data(const T* buf, size_t len, dims_type dims) -> RTNType;
  auto take_data(std::vector<double>&& buf, dims_type dims) -> RTNType;

  //
  // Output
  //
  auto view_data() const -> const std::vector<double>&;
  auto release_data() -> std::vector<double>&&;
  auto get_dims() const -> std::array<size_t, 3>;  // In 2D case, the 3rd value equals 1.

  // Action items
  void dwt1d();
  void dwt2d();
  void dwt3d();
  void idwt2d();
  void idwt1d();
  void idwt3d();

 private:
  //
  // Private methods helping DWT.
  //
  // Perform one level of interleaved 3D dwt/idwt on a given volume (m_dims),
  // specifically on its top left (len_xyz) subset.
  void m_dwt3d_one_level(double* vol, std::array<size_t, 3> len_xyz);
  void m_idwt3d_one_level(double* vol, std::array<size_t, 3> len_xyz);

  // Perform one level of 2D dwt/idwt on a given plane (m_dims),
  // specifically on its top left (len_xy) subset.
  void m_dwt2d_one_level(double* plane, std::array<size_t, 2> len_xy);
  void m_idwt2d_one_level(double* plane, std::array<size_t, 2> len_xy);

  // Perform one level of 1D dwt/idwt on a given array (array_len).
  // A buffer space (tmp_buf) should be passed in for
  // this method to work on with length at least 2*array_len.
  void m_dwt1d_one_level(double* array, size_t array_len);
  void m_idwt1d_one_level(double* array, size_t array_len);

  void SYM_DWT(double* input, size_t N, double* output);
  void SYM_IDWT(double* input, size_t N, double* output);

  void DOWNSAMPLING_CONV(double* input, size_t N, const TYPE* filter, size_t F, double* output);
  void UPSAMPLING_CONV(double* input, size_t N, const TYPE* filter, size_t F, double* output);

  //
  // Private data members
  //
  vecd_type m_data_buf;          // Holds the entire input data.
  dims_type m_dims = {0, 0, 0};  // Dimension of the data volume

  // Temporary buffers that are big enough for any (1D column * 2) or any 2D
  // slice. Note: `m_qcc_buf` should be used by m_***_one_level() functions and
  // should not be used by higher-level functions. `m_slice_buf` is only used by
  // wavelet-packet transforms.
  vecd_type m_qcc_buf;
  vecd_type m_slice_buf;

  const std::array<double, 26> sym13_dec_lo = {
      6.820325263075319e-05,  -3.573862364868901e-05, -0.0011360634389281183,
      -0.0001709428585302221, 0.0075262253899681,     0.005296359738725025,
      -0.02021676813338983,   -0.017211642726299048,  0.013862497435849205,
      -0.0597506277179437,    -0.12436246075153011,   0.19770481877117801,
      0.6957391505614964,     0.6445643839011856,     0.11023022302137217,
      -0.14049009311363403,   0.008819757670420546,   0.09292603089913712,
      0.017618296880653084,   -0.020749686325515677,  -0.0014924472742598532,
      0.0056748537601224395,  0.00041326119884196064, -0.0007213643851362283,
      3.690537342319624e-05,  7.042986690694402e-05};

  const std::array<double, 26> sym13_dec_hi = {
      -7.042986690694402e-05, 3.690537342319624e-05,  0.0007213643851362283, 0.00041326119884196064,
      -0.0056748537601224395, -0.0014924472742598532, 0.020749686325515677,  0.017618296880653084,
      -0.09292603089913712,   0.008819757670420546,   0.14049009311363403,   0.11023022302137217,
      -0.6445643839011856,    0.6957391505614964,     -0.19770481877117801,  -0.12436246075153011,
      0.0597506277179437,     0.013862497435849205,   0.017211642726299048,  -0.02021676813338983,
      -0.005296359738725025,  0.0075262253899681,     0.0001709428585302221, -0.0011360634389281183,
      3.573862364868901e-05,  6.820325263075319e-05};

  const std::array<double, 26> sym13_rec_lo = {
      7.042986690694402e-05,  3.690537342319624e-05,  -0.0007213643851362283,
      0.00041326119884196064, 0.0056748537601224395,  -0.0014924472742598532,
      -0.020749686325515677,  0.017618296880653084,   0.09292603089913712,
      0.008819757670420546,   -0.14049009311363403,   0.11023022302137217,
      0.6445643839011856,     0.6957391505614964,     0.19770481877117801,
      -0.12436246075153011,   -0.0597506277179437,    0.013862497435849205,
      -0.017211642726299048,  -0.02021676813338983,   0.005296359738725025,
      0.0075262253899681,     -0.0001709428585302221, -0.0011360634389281183,
      -3.573862364868901e-05, 6.820325263075319e-05};

  const std::array<double, 26> sym13_rec_hi = {
      6.820325263075319e-05,  3.573862364868901e-05,  -0.0011360634389281183, 0.0001709428585302221,
      0.0075262253899681,     -0.005296359738725025,  -0.02021676813338983,   0.017211642726299048,
      0.013862497435849205,   0.0597506277179437,     -0.12436246075153011,   -0.19770481877117801,
      0.6957391505614964,     -0.6445643839011856,    0.11023022302137217,    0.14049009311363403,
      0.008819757670420546,   -0.09292603089913712,   0.017618296880653084,   0.020749686325515677,
      -0.0014924472742598532, -0.0056748537601224395, 0.00041326119884196064, 0.0007213643851362283,
      3.690537342319624e-05,  -7.042986690694402e-05};
};

};  // namespace sperr

#endif
