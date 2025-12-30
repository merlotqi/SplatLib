/**
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 */

#include <splat/models/morton-order.h>

#include <cuda_runtime.h>
#include <cfloat>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cmath>

namespace splat {

__device__ inline uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ static void atomicMinFloat(float* addr, float val) {
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
}

__device__ static void atomicMaxFloat(float* addr, float val) {
  int* addr_as_i = (int*)addr;
  int old = *addr_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
}

__global__ void calcAABBAndMorton(const float* px, const float* py, const float* pz, uint32_t* codes, uint32_t* indices,
                                  float* minB, float* maxB, uint32_t n, bool secondPass, float3 minBound,
                                  float3 invRange) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (!secondPass) {
    atomicMinFloat(&minB[0], px[i]);
    atomicMaxFloat(&maxB[0], px[i]);
    atomicMinFloat(&minB[1], py[i]);
    atomicMaxFloat(&maxB[1], py[i]);
    atomicMinFloat(&minB[2], pz[i]);
    atomicMaxFloat(&maxB[2], pz[i]);
  } else {
    uint32_t ix = (uint32_t)fminf(1023.0f, fmaxf(0.0f, (px[i] - minBound.x) * invRange.x * 1023.0f));
    uint32_t iy = (uint32_t)fminf(1023.0f, fmaxf(0.0f, (py[i] - minBound.y) * invRange.y * 1023.0f));
    uint32_t iz = (uint32_t)fminf(1023.0f, fmaxf(0.0f, (pz[i] - minBound.z) * invRange.z * 1023.0f));

    codes[i] = (expandBits(iz) << 2) | (expandBits(iy) << 1) | expandBits(ix);
  }
}

void sortMortonOrder(const DataTable* dataTable, absl::Span<uint32_t> indices) {
  if (!dataTable || indices.empty()) return;

  uint32_t n = static_cast<uint32_t>(indices.size());
  const float* h_px = reinterpret_cast<const float*>(dataTable->getColumnByName("x").rawPointer());
  const float* h_py = reinterpret_cast<const float*>(dataTable->getColumnByName("y").rawPointer());
  const float* h_pz = reinterpret_cast<const float*>(dataTable->getColumnByName("z").rawPointer());

  float *d_x, *d_y, *d_z, *d_minB, *d_maxB;
  uint32_t *d_codes, *d_indices;

  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));
  cudaMalloc(&d_z, n * sizeof(float));
  cudaMalloc(&d_codes, n * sizeof(uint32_t));
  cudaMalloc(&d_indices, n * sizeof(uint32_t));
  cudaMalloc(&d_minB, 3 * sizeof(float));
  cudaMalloc(&d_maxB, 3 * sizeof(float));

  float initMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float initMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
  cudaMemcpy(d_minB, initMin, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxB, initMax, 3 * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, h_pz, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  calcAABBAndMorton<<<blocks, threads>>>(d_x, d_y, d_z, d_codes, d_indices, d_minB, d_maxB, n, false, {}, {});

  float resMin[3], resMax[3];
  cudaMemcpy(resMin, d_minB, 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(resMax, d_maxB, 3 * sizeof(float), cudaMemcpyDeviceToHost);

  float3 minBound = {resMin[0], resMin[1], resMin[2]};
  float3 invRange = {(resMax[0] - resMin[0] > 1e-6f) ? 1.0f / (resMax[0] - resMin[0]) : 0.0f,
                     (resMax[1] - resMin[1] > 1e-6f) ? 1.0f / (resMax[1] - resMin[1]) : 0.0f,
                     (resMax[2] - resMin[2] > 1e-6f) ? 1.0f / (resMax[2] - resMin[2]) : 0.0f};

  calcAABBAndMorton<<<blocks, threads>>>(d_x, d_y, d_z, d_codes, d_indices, d_minB, d_maxB, n, true, minBound,
                                         invRange);

  thrust::device_ptr<uint32_t> t_codes(d_codes);
  thrust::device_ptr<uint32_t> t_indices(d_indices);
  thrust::sort_by_key(t_codes, t_codes + n, t_indices);

  cudaMemcpy(indices.data(), d_indices, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_codes);
  cudaFree(d_indices);
  cudaFree(d_minB);
  cudaFree(d_maxB);
}

}  // namespace splat
