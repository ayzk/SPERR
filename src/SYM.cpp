#include "SYM.h"

#include <algorithm>
#include <cassert>
#include <numeric>  // std::accumulate()
#include <type_traits>

template <typename T>
auto sperr::SYM::copy_data(const T* data, size_t len, dims_type dims) -> RTNType
{
  static_assert(std::is_floating_point<T>::value, "!! Only floating point values are supported !!");
  if (len != dims[0] * dims[1] * dims[2])
    return RTNType::WrongDims;

  m_data_buf.resize(len);
  std::copy(data, data + len, m_data_buf.begin());

  m_dims = dims;

  auto max_col = std::max(std::max(dims[0], dims[1]), dims[2]);
  if (max_col * 2 > m_qcc_buf.size())
    m_qcc_buf.resize(std::max(m_qcc_buf.size(), max_col) * 2);

  auto max_slice = std::max(std::max(dims[0] * dims[1], dims[0] * dims[2]), dims[1] * dims[2]);
  if (max_slice > m_slice_buf.size())
    m_slice_buf.resize(std::max(m_slice_buf.size() * 2, max_slice));

  return RTNType::Good;
}
template auto sperr::SYM::copy_data(const float*, size_t, dims_type) -> RTNType;
template auto sperr::SYM::copy_data(const double*, size_t, dims_type) -> RTNType;

auto sperr::SYM::take_data(vecd_type&& buf, dims_type dims) -> RTNType
{
  if (buf.size() != dims[0] * dims[1] * dims[2])
    return RTNType::WrongDims;

  m_data_buf = std::move(buf);
  m_dims = dims;

  auto max_col = std::max(std::max(dims[0], dims[1]), dims[2]);
  if (max_col * 2 > m_qcc_buf.size())
    m_qcc_buf.resize(std::max(m_qcc_buf.size(), max_col) * 2);

  auto max_slice = std::max(std::max(dims[0] * dims[1], dims[0] * dims[2]), dims[1] * dims[2]);
  if (max_slice > m_slice_buf.size())
    m_slice_buf.resize(std::max(m_slice_buf.size() * 2, max_slice));

  return RTNType::Good;
}

auto sperr::SYM::view_data() const -> const vecd_type&
{
  return m_data_buf;
}

auto sperr::SYM::release_data() -> vecd_type&&
{
  m_dims = {0, 0, 0};
  return std::move(m_data_buf);
}

auto sperr::SYM::get_dims() const -> std::array<size_t, 3>
{
  return m_dims;
}

void sperr::SYM::dwt1d()
{
  size_t num_xforms = sperr::num_of_xforms(m_dims[0]);
  for (size_t lev = 0; lev < num_xforms; lev++) {
    auto [apx, nnm] = sperr::calc_approx_detail_len(m_data_buf.size(), lev);
    m_dwt1d_one_level(m_data_buf.data(), apx);
  }
}

void sperr::SYM::idwt1d()
{
  size_t num_xforms = sperr::num_of_xforms(m_dims[0]);
  for (size_t lev = num_xforms; lev > 0; lev--) {
    auto [apx, nnm] = sperr::calc_approx_detail_len(m_data_buf.size(), lev - 1);
    m_idwt1d_one_level(m_data_buf.data(), apx);
  }
}

void sperr::SYM::dwt2d()
{
  size_t num_xforms_xy = sperr::num_of_xforms(std::min(m_dims[0], m_dims[1]));
  for (size_t lev = 0; lev < num_xforms_xy; lev++) {
    auto approx_x = sperr::calc_approx_detail_len(m_dims[0], lev);
    auto approx_y = sperr::calc_approx_detail_len(m_dims[1], lev);
    m_dwt2d_one_level(m_data_buf.data(), {approx_x[0], approx_y[0]});
  }
}

void sperr::SYM::idwt2d()
{
  size_t num_xforms_xy = sperr::num_of_xforms(std::min(m_dims[0], m_dims[1]));
  for (size_t lev = 0; lev < num_xforms_xy; lev++) {
    auto approx_x = sperr::calc_approx_detail_len(m_dims[0], lev - 1);
    auto approx_y = sperr::calc_approx_detail_len(m_dims[1], lev - 1);
    m_idwt2d_one_level(m_data_buf.data(), {approx_x[0], approx_y[0]});
  }
}

void sperr::SYM::dwt3d()
{
  // clang-format off
  const auto xforms = std::array<size_t, 3>{sperr::num_of_xforms(m_dims[0]),
                                            sperr::num_of_xforms(m_dims[1]),
                                            sperr::num_of_xforms(m_dims[2])};
  const auto num_xforms = *std::min_element(xforms.cbegin(), xforms.cend());
  // clang-format on

  for (size_t lev = 0; lev < num_xforms; lev++) {
    auto app_x = sperr::calc_approx_detail_len(m_dims[0], lev);
    auto app_y = sperr::calc_approx_detail_len(m_dims[1], lev);
    auto app_z = sperr::calc_approx_detail_len(m_dims[2], lev);
    m_dwt3d_one_level(m_data_buf.data(), {app_x[0], app_y[0], app_z[0]});
  }
}

void sperr::SYM::idwt3d()
{
  // clang-format off
  const auto xforms = std::array<size_t, 3>{sperr::num_of_xforms(m_dims[0]),
                                            sperr::num_of_xforms(m_dims[1]),
                                            sperr::num_of_xforms(m_dims[2])};
  const auto num_xforms = *std::min_element(xforms.cbegin(), xforms.cend());
  // clang-format on

  for (size_t lev = num_xforms; lev > 0; lev--) {
    auto app_x = sperr::calc_approx_detail_len(m_dims[0], lev - 1);
    auto app_y = sperr::calc_approx_detail_len(m_dims[1], lev - 1);
    auto app_z = sperr::calc_approx_detail_len(m_dims[2], lev - 1);
    m_idwt3d_one_level(m_data_buf.data(), {app_x[0], app_y[0], app_z[0]});
  }
}

//
// Private Methods
//

void sperr::SYM::m_dwt1d_one_level(double* array, size_t array_len)
{
  std::copy(array, array + array_len, m_qcc_buf.begin());
  this->SYM_DWT(m_qcc_buf.data(), array_len, array);
}

void sperr::SYM::m_idwt1d_one_level(double* array, size_t array_len)
{
  std::copy(array, array + array_len, m_qcc_buf.begin());
  this->SYM_IDWT(m_qcc_buf.data(), array_len, array);
}

void sperr::SYM::m_dwt2d_one_level(double* plane, std::array<size_t, 2> len_xy)
{
  // Note: here we call low-level functions (Qcc*()) instead of
  // m_dwt1d_one_level() because we want to have only one even/odd test outside of
  // the loop.

  const size_t max_len = std::max(len_xy[0], len_xy[1]);
  const auto beg = m_qcc_buf.data();
  const auto beg2 = beg + max_len;

  for (size_t i = 0; i < len_xy[1]; i++) {
    auto pos = plane + i * m_dims[0];
    std::copy(pos, pos + len_xy[0], beg);
    this->SYM_DWT(m_qcc_buf.data(), len_xy[0], pos);
  }

  // Second, perform DWT along Y for every column
  // Note, I've tested that up to 1024^2 planes it is actually slightly slower
  // to transpose the plane and then perform the transforms. This was consistent
  // on both a MacBook and a RaspberryPi 3. Note2, I've tested transpose again
  // on an X86 linux machine using gcc, clang, and pgi. Again the difference is
  // either indistinguishable, or the current implementation has a slight edge.

  for (size_t x = 0; x < len_xy[0]; x++) {
    for (size_t y = 0; y < len_xy[1]; y++)
      m_qcc_buf[y] = *(plane + y * m_dims[0] + x);
    this->SYM_DWT(m_qcc_buf.data(), len_xy[1], beg2);
    for (size_t y = 0; y < len_xy[1]; y++)
      *(plane + y * m_dims[0] + x) = *(beg2 + y);
  }
}

void sperr::SYM::m_idwt2d_one_level(double* plane, std::array<size_t, 2> len_xy)
{
  const size_t max_len = std::max(len_xy[0], len_xy[1]);
  const auto beg = m_qcc_buf.data();  // First half of the buffer
  const auto beg2 = beg + max_len;    // Second half of the buffer

  for (size_t x = 0; x < len_xy[0]; x++) {
    for (size_t y = 0; y < len_xy[1]; y++)
      m_qcc_buf[y] = *(plane + y * m_dims[0] + x);
    this->SYM_IDWT(m_qcc_buf.data(), len_xy[1], beg2);
    for (size_t y = 0; y < len_xy[1]; y++)
      *(plane + y * m_dims[0] + x) = *(beg2 + y);
  }

  for (size_t i = 0; i < len_xy[1]; i++) {
    auto pos = plane + i * m_dims[0];
    std::copy(pos, pos + len_xy[0], beg);
    this->SYM_IDWT(m_qcc_buf.data(), len_xy[0], pos);
  }
}

void sperr::SYM::m_dwt3d_one_level(double* vol, std::array<size_t, 3> len_xyz)
{
  // First, do one level of transform on all XY planes.
  const size_t plane_size_xy = m_dims[0] * m_dims[1];
  for (size_t z = 0; z < len_xyz[2]; z++) {
    const size_t offset = plane_size_xy * z;
    m_dwt2d_one_level(vol + offset, {len_xyz[0], len_xyz[1]});
  }

  const auto beg = m_qcc_buf.data();   // First half of the buffer
  const auto beg2 = beg + len_xyz[2];  // Second half of the buffer

  // Second, do one level of transform on all Z columns.  Strategy:
  // 1) extract a Z column to buffer space `m_qcc_buf`
  // 2) use appropriate even/odd Qcc*** function to transform it
  // 3) gather coefficients from `m_qcc_buf` to the second half of `m_qcc_buf`
  // 4) put the Z column back to their locations as a Z column.

  for (size_t y = 0; y < len_xyz[1]; y++) {
    for (size_t x = 0; x < len_xyz[0]; x++) {
      const size_t xy_offset = y * m_dims[0] + x;
      // Step 1
      for (size_t z = 0; z < len_xyz[2]; z++)
        m_qcc_buf[z] = m_data_buf[z * plane_size_xy + xy_offset];
      // Step 2
      this->SYM_DWT(m_qcc_buf.data(), len_xyz[2], beg2);

      // Step 4
      for (size_t z = 0; z < len_xyz[2]; z++)
        m_data_buf[z * plane_size_xy + xy_offset] = *(beg2 + z);
    }
  }
}

void sperr::SYM::m_idwt3d_one_level(double* vol, std::array<size_t, 3> len_xyz)
{
  const size_t plane_size_xy = m_dims[0] * m_dims[1];
  const auto beg = m_qcc_buf.data();   // First half of the buffer
  const auto beg2 = beg + len_xyz[2];  // Second half of the buffer

  // First, do one level of inverse transform on all Z columns.  Strategy:
  // 1) extract a Z column to buffer space `m_qcc_buf`
  // 2) scatter coefficients from `m_qcc_buf` to the second half of `m_qcc_buf`
  // 3) use appropriate even/odd Qcc*** function to transform it
  // 4) put the Z column back to their locations as a Z column.

  for (size_t y = 0; y < len_xyz[1]; y++) {
    for (size_t x = 0; x < len_xyz[0]; x++) {
      const size_t xy_offset = y * m_dims[0] + x;
      // Step 1
      for (size_t z = 0; z < len_xyz[2]; z++)
        m_qcc_buf[z] = m_data_buf[z * plane_size_xy + xy_offset];
      // Step 3
      this->SYM_IDWT(m_qcc_buf.data(), len_xyz[2], beg2);
      // Step 4
      for (size_t z = 0; z < len_xyz[2]; z++)
        m_data_buf[z * plane_size_xy + xy_offset] = *(beg2 + z);
    }
  }

  // Second, do one level of inverse transform on all XY planes.
  for (size_t z = 0; z < len_xyz[2]; z++) {
    const size_t offset = plane_size_xy * z;
    m_idwt2d_one_level(vol + offset, {len_xyz[0], len_xyz[1]});
  }
}

void sperr::SYM::SYM_DWT(double* input, size_t N, double* output)
{
  this->DOWNSAMPLING_CONV(input, N, sym13_dec_lo.data(), sym13_dec_lo.size(), output);
  this->DOWNSAMPLING_CONV(input, N, sym13_dec_hi.data(), sym13_dec_lo.size(), output + N / 2);
}

void sperr::SYM::SYM_IDWT(double* input, size_t N, double* output)
{
  std::fill(output, output + N, 0);
  this->DOWNSAMPLING_CONV(input, N / 2, sym13_rec_lo.data(), sym13_rec_lo.size(), output);
  this->DOWNSAMPLING_CONV(input + N / 2, N / 2, sym13_rec_hi.data(), sym13_rec_hi.size(), output);
}

// TODO output should have the same size as input, but current it is not the case.
void sperr::SYM::UPSAMPLING_CONV(double* input,
                                 size_t N,
                                 const TYPE* filter,
                                 size_t F,
                                 double* output)
{
  size_t const start = F / 4;
  size_t i = start;
  size_t const end = N + start - (((F / 2) % 2) ? 0 : 1);
  size_t o = 0;

  if (F % 2) {
    exit(0); /* Filter must have even-length. */
  }

  if ((F / 2) % 2 == 0) {
    // Shift output one element right. This is necessary for perfect reconstruction.

    // i = N-1; even element goes to output[O-1], odd element goes to output[0]
    size_t j = 0;
    while (j <= start - 1) {
      size_t k;
      for (k = 0; k < N && j <= start - 1; ++k, ++j) {
        output[2 * N - 1] += filter[2 * (start - 1 - j)] * input[k];
        output[0] += filter[2 * (start - 1 - j) + 1] * input[k];
      }
    }
    for (; j <= N + start - 1 && j < F / 2; ++j) {
      output[2 * N - 1] += filter[2 * j] * input[N + start - 1 - j];
      output[0] += filter[2 * j + 1] * input[N + start - 1 - j];
    }
    while (j < F / 2) {
      size_t k;
      for (k = 0; k < N && j < F / 2; ++k, ++j) {
        output[2 * N - 1] += filter[2 * j] * input[N - 1 - k];
        output[0] += filter[2 * j + 1] * input[N - 1 - k];
      }
    }

    o += 1;
  }

  for (; i < F / 2 && i < N; ++i, o += 2) {
    size_t j = 0;
    for (; j <= i; ++j) {
      output[o] += filter[2 * j] * input[i - j];
      output[o + 1] += filter[2 * j + 1] * input[i - j];
    }
    while (j < F / 2) {
      size_t k;
      for (k = 0; k < N && j < F / 2; ++k, ++j) {
        output[o] += filter[2 * j] * input[N - 1 - k];
        output[o + 1] += filter[2 * j + 1] * input[N - 1 - k];
      }
    }
  }

  for (; i < N; ++i, o += 2) {
    size_t j;
    for (j = 0; j < F / 2; ++j) {
      output[o] += filter[2 * j] * input[i - j];
      output[o + 1] += filter[2 * j + 1] * input[i - j];
    }
  }

  for (; i < F / 2 && i < end; ++i, o += 2) {
    size_t j = 0;
    while (i - j >= N) {
      size_t k;
      for (k = 0; k < N && i - j >= N; ++k, ++j) {
        output[o] += filter[2 * (i - N - j)] * input[k];
        output[o + 1] += filter[2 * (i - N - j) + 1] * input[k];
      }
    }
    for (; j <= i && j < F / 2; ++j) {
      output[o] += filter[2 * j] * input[i - j];
      output[o + 1] += filter[2 * j + 1] * input[i - j];
    }
    while (j < F / 2) {
      size_t k;
      for (k = 0; k < N && j < F / 2; ++k, ++j) {
        output[o] += filter[2 * j] * input[N - 1 - k];
        output[o + 1] += filter[2 * j + 1] * input[N - 1 - k];
      }
    }
  }

  for (; i < end; ++i, o += 2) {
    size_t j = 0;
    while (i - j >= N) {
      size_t k;
      for (k = 0; k < N && i - j >= N; ++k, ++j) {
        output[o] += filter[2 * (i - N - j)] * input[k];
        output[o + 1] += filter[2 * (i - N - j) + 1] * input[k];
      }
    }
    for (; j <= i && j < F / 2; ++j) {
      output[o] += filter[2 * j] * input[i - j];
      output[o + 1] += filter[2 * j + 1] * input[i - j];
    }
  }
}

// TODO output should have the same size as input, but current it is not the case.
void sperr::SYM::DOWNSAMPLING_CONV(double* input,
                                   size_t N,
                                   const TYPE* filter,
                                   size_t F,
                                   double* output)
{
  size_t step = 2, fstep = 1;
  size_t i = F / 2, o = 0;
  const size_t padding = (step - (N % step)) % step;

  for (; i < F && i < N; i += step, ++o) {
    TYPE sum = 0;
    size_t j;
    size_t k_start = 0;
    for (j = 0; j <= i; j += fstep)
      sum += filter[j] * input[i - j];
    if (fstep > 1)
      k_start = j - (i + 1);
    while (j < F) {
      size_t k;
      for (k = k_start; k < padding && j < F; k += fstep, j += fstep)
        sum += filter[j] * input[N - 1];
      for (k = k_start; k < N && j < F; k += fstep, j += fstep)
        sum += filter[j] * input[N - 1 - k];
    }
    output[o] = sum;
  }

  for (; i < N; i += step, ++o) {
    TYPE sum = 0;
    size_t j;
    for (j = 0; j < F; j += fstep)
      sum += input[i - j] * filter[j];
    output[o] = sum;
  }

  for (; i < F && i < N + F / 2; i += step, ++o) {
    TYPE sum = 0;
    size_t j = 0;
    size_t k_start = 0;
    while (i - j >= N) {
      size_t k;
      // for simplicity, not using fstep here
      for (k = 0; k < padding && i - j >= N; ++k, ++j)
        sum += filter[i - N - j] * input[N - 1];
      for (k = 0; k < N && i - j >= N; ++k, ++j)
        sum += filter[i - N - j] * input[k];
    }
    if (fstep > 1)
      j += (fstep - j % fstep) % fstep;  // move to next non-zero entry
    for (; j <= i; j += fstep)
      sum += filter[j] * input[i - j];
    if (fstep > 1)
      k_start = j - (i + 1);
    while (j < F) {
      size_t k;
      for (k = k_start; k < padding && j < F; k += fstep, j += fstep)
        sum += filter[j] * input[N - 1];
      for (k = k_start; k < N && j < F; k += fstep, j += fstep)
        sum += filter[j] * input[N - 1 - k];
    }
    output[o] = sum;
  }

  for (; i < N + F / 2; i += step, ++o) {
    TYPE sum = 0;
    size_t j = 0;
    while (i - j >= N) {
      // for simplicity, not using fstep here
      size_t k;
      for (k = 0; k < padding && i - j >= N; ++k, ++j)
        sum += filter[i - N - j] * input[N - 1];
      for (k = 0; k < N && i - j >= N; ++k, ++j)
        sum += filter[i - N - j] * input[k];
    }
    if (fstep > 1)
      j += (fstep - j % fstep) % fstep;  // move to next non-zero entry
    for (; j < F; j += fstep)
      sum += filter[j] * input[i - j];
    output[o] = sum;
  }
}