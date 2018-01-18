#include <helper_cuda.h>

#include "include/host_image.h"
#include "include/device_util.h"
#include "include/laplacian.h"

void image_to_gpu(
  HostImage *img,
  float **d_rdata,
  float **d_gdata,
  float **d_bdata
) {
  size_t data_size = sizeof(float) * img->w * img->h;
  
  checkCudaErrors(cudaMalloc(d_rdata, data_size));
  checkCudaErrors(cudaMalloc(d_gdata, data_size));
  checkCudaErrors(cudaMalloc(d_bdata, data_size));

  checkCudaErrors(cudaMemcpy(*d_rdata, img->r_data, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_gdata, img->g_data, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*d_bdata, img->b_data, data_size, cudaMemcpyHostToDevice));
}

void image_from_gpu(
  HostImage *img,
  float *d_rdata,
  float *d_gdata,
  float *d_bdata
) {
  size_t data_size = sizeof(float) * img->w * img->h;

  checkCudaErrors(cudaMemcpy(img->r_data, d_rdata, data_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(img->g_data, d_gdata, data_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(img->b_data, d_bdata, data_size, cudaMemcpyDeviceToHost));
}

__device__ void masked_image(
  int w,
  int h,
  float *mask,
  float *chan_data,
  float *result,
  float mask_val,
  float spec_val
) { 
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;
  result[index] = (fabs(mask[index] - mask_val) > 0.0001) ? chan_data[index] : spec_val;
}

__device__ void paste_masked(
  int w,
  int h,
  float *bg_data,
  float *fg_data,
  float *result,
  float spec_val
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  result[index] = fg_data[index] == spec_val ? bg_data[index] : fg_data[index];
}

__device__ void mat_mul(
  int w,
  int h,
  float *a,
  float *b,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = a[index] * b[index];
}

__device__ void mat_add(
  int w,
  int h,
  float *a,
  float *b,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = a[index] + b[index];
}

__device__ void mat_sub(
  int w,
  int h,
  float *a,
  float *b,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = a[index] - b[index];
}

__device__ void mat_add_scalar(
  int w,
  int h,
  float *m,
  float s,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = m[index] + s;
}

__device__ void mat_mul_scalar(
  int w,
  int h,
  float *m,
  float s,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = m[index] * s;
}

__global__ void mat_sum_all(
  int w,
  int h,
  float *m
) {
  __shared__ float temp[LAP_BLOCK_X][LAP_BLOCK_Y];
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;
  temp[threadIdx.x][threadIdx.y] = m[index];

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    atomicExch(&m[0], 0);
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float sum = 0.0f;
    for(int i = 0; i < blockDim.x; ++i) {
      for(int j = 0; j < blockDim.y; ++j) {
        sum += temp[i][j];      
      }
    }
    atomicAdd(&m[0], sum);
  }
}

__device__ void mat_clip(
  int w,
  int h,
  float *m,
  float min_v,
  float max_v,
  float *r
) {
  const int xIdx = threadIdx.x + blockIdx.x * blockDim.x;
  const int yIdx = threadIdx.y + blockIdx.y * blockDim.y;
  const int index = yIdx * w + xIdx;

  r[index] = m[index] > max_v ? max_v : m[index];
  r[index] = r[index] < min_v ? min_v : r[index];
}


