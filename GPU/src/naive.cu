#include "include/naive.h"
#include "include/host_image.h"
#include "include/device_util.h"
#include <helper_cuda.h>

const size_t BLOCK_DIM = 32;

__global__ void apply_mask_to_image(int w, int h, float *fg, float *mask, float *r) {
  masked_image(w, h, mask, fg, r, 0.0f, 2.0f);
}

__global__ void naive_paste(int w, int h, float *bg, float *fg, float *r) {
  paste_masked(w, h, bg, fg, r, 2.0f);
}

HostImage* run_naive_method(HostImage *bg, HostImage *fg, HostImage *mask) {
  findCudaDevice(0, NULL);

  size_t block_cnt = (bg->w * bg->h) / BLOCK_DIM;
  size_t block_rem = (bg->w * bg->h) % BLOCK_DIM;
  size_t new_w = bg->w + block_rem;
  size_t new_h = bg->h + block_rem;

  HostImage *aligned_bg = resize_host_image(
    bg,
    new_w, 
    new_h,
    0,
    0
  );

  HostImage *aligned_fg = resize_host_image(
    fg,
    new_w, 
    new_h,
    0,
    0
  );

  HostImage *aligned_mask = resize_host_image(
    mask,
    new_w, 
    new_h,
    0,
    0
  );

  float *d_bg_r, *d_bg_g, *d_bg_b;
  float *d_fg_r, *d_fg_g, *d_fg_b;
  float *d_mask_r, *d_mask_g, *d_mask_b;
  float *d_masked_r, *d_masked_g, *d_masked_b;
  float *d_res_r, *d_res_g, *d_res_b;

  checkCudaErrors(cudaMalloc(&d_masked_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_masked_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_masked_b, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_res_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_res_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_res_b, sizeof(float) * new_w * new_h));

  image_to_gpu(
    aligned_bg,
    &d_bg_r,
    &d_bg_g,
    &d_bg_b
  );

  image_to_gpu(
    aligned_fg,
    &d_fg_r,
    &d_fg_g,
    &d_fg_b
  );
  
  image_to_gpu(
    aligned_mask,
    &d_mask_r,
    &d_mask_g,
    &d_mask_b
  );

  apply_mask_to_image<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_fg_r, d_mask_r, d_masked_r);
  apply_mask_to_image<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_fg_g, d_mask_r, d_masked_g);
  apply_mask_to_image<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_fg_b, d_mask_r, d_masked_b);

  naive_paste<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_bg_r, d_masked_r, d_res_r);
  naive_paste<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_bg_g, d_masked_g, d_res_g);
  naive_paste<<<block_cnt, BLOCK_DIM>>>(new_w, new_h, d_bg_b, d_masked_b, d_res_b);

  checkCudaErrors(cudaDeviceSynchronize());

  HostImage *result = new_host_image(new_w, new_h);
  image_from_gpu(result, d_res_r, d_res_g, d_res_b);

  drop_host_image(aligned_bg);
  drop_host_image(aligned_fg);
  drop_host_image(aligned_mask);

  checkCudaErrors(cudaFree(d_masked_r));
  checkCudaErrors(cudaFree(d_masked_g));
  checkCudaErrors(cudaFree(d_masked_b));
  checkCudaErrors(cudaFree(d_fg_r));
  checkCudaErrors(cudaFree(d_fg_g));
  checkCudaErrors(cudaFree(d_fg_b));
  checkCudaErrors(cudaFree(d_bg_r));
  checkCudaErrors(cudaFree(d_bg_g));
  checkCudaErrors(cudaFree(d_bg_b));
  checkCudaErrors(cudaFree(d_res_r));
  checkCudaErrors(cudaFree(d_res_g));
  checkCudaErrors(cudaFree(d_res_b));

  cudaDeviceReset();

  return result;
}

