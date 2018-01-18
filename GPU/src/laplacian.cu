#include "include/laplacian.h"
#include "include/device_util.h"
#include "include/host_image.h"
#include <helper_cuda.h>

__constant__ float L[9];

void set_kern() {
  float K[9] = {
    0.0f, -1.0f,  0.0f,
   -1.0f,  4.0f, -1.0f,
    0.0f, -1.0f,  0.0f
  };

  cudaMemcpyToSymbol(L, K, sizeof(float) * 9);
}

__global__ void laplacian_kern(size_t w, size_t h, float *src, float *dst) {
  laplacian_gpu(w, h, src, dst);
}

__device__ void laplacian_gpu(size_t w, size_t h, float *src, float *dst) {
  __shared__ float surround[LAP_BLOCK_X + 2][LAP_BLOCK_Y + 2];

  const int x_pos = threadIdx.x + (blockIdx.x * blockDim.x);
  const int y_pos = threadIdx.y + (blockIdx.y * blockDim.y);

  const int pos = x_pos + y_pos * w;


  // Upper left:
  if ((x_pos - 1) > 0 && (y_pos - 1) > 0) {
    surround[threadIdx.x][threadIdx.y] = src[pos - w - 1];
  }
  else {
    surround[threadIdx.x][threadIdx.y] = 0.0f;
  }

  // Upper right:
  if ((x_pos + 1) < w && (y_pos - 1) > 0) {
    surround[threadIdx.x + 2][threadIdx.y] = src[pos - w + 1];
  }
  else {
    surround[threadIdx.x + 2][threadIdx.y] = 0.0f;
  }

  // Lower left:
  if ((x_pos - 1) > 0 && (y_pos + 1) < h) {
    surround[threadIdx.x][threadIdx.y + 2] = src[pos + w - 1];
  }
  else {
    surround[threadIdx.x][threadIdx.y + 2] = 0.0f;
  }

  // Lower right:
  if ((x_pos + 1) < w && (y_pos + 1) < h) {
    surround[threadIdx.x + 2][threadIdx.y + 2] = src[pos + w + 1];
  }
  else {
    surround[threadIdx.x + 2][threadIdx.y + 2] = 0.0f;
  }

  __syncthreads();
 
  dst[pos] = surround[threadIdx.x][threadIdx.y] * L[0];
  dst[pos] += surround[threadIdx.x + 1][threadIdx.y] * L[1];
  dst[pos] += surround[threadIdx.x + 2][threadIdx.y] * L[2];
  dst[pos] += surround[threadIdx.x][threadIdx.y + 1] * L[3];
  dst[pos] += surround[threadIdx.x + 1][threadIdx.y + 1] * L[4];
  dst[pos] += surround[threadIdx.x + 2][threadIdx.y + 1] * L[5];
  dst[pos] += surround[threadIdx.x][threadIdx.y + 2] * L[6];
  dst[pos] += surround[threadIdx.x + 1][threadIdx.y + 2] * L[7];
  dst[pos] += surround[threadIdx.x + 2][threadIdx.y + 2] * L[8];
}

void laplacian(size_t w, size_t h, float *src, float *dst) {
  size_t block_cnt_w = w / LAP_BLOCK_X;
  size_t block_cnt_h = h / LAP_BLOCK_Y;
  dim3 block_cnt(block_cnt_w, block_cnt_h);
  dim3 block_size(LAP_BLOCK_X, LAP_BLOCK_Y);

  laplacian_kern<<<block_cnt, block_size>>>(w, h, src, dst);
  
  HostImage *image = new_host_image(w, h);
  image_from_gpu(image, dst, dst, dst);
}
