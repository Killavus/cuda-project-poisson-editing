#include <helper_cuda.h>

#include "include/poisson.h"
#include "include/laplacian.h"
#include "include/device_util.h"

__global__ void apply_mask_to_image(size_t w, size_t h, float *fg, float *mask, float *r, float mask_val) {
  masked_image(w, h, mask, fg, r, mask_val, 0.0f);
}

__global__ void initial_x(size_t w, size_t h, float *maskedbg, float *maskedfg, float *x) {
  mat_add(w, h, maskedbg, maskedfg, x);
}

__global__ void clip_result(size_t w, size_t h, float *result, float *clipped) {
  mat_clip(w, h, result, 0.0f, 1.0f, clipped);
}

__global__ void poisson_kern_p1(
  size_t w,
  size_t h,
  float *mask,
  float *a_utosum,
  float *a_btosum,
  float *b,
  float *r,
  float *x
) {
  laplacian_gpu(w, h, x, r);
  mat_sub(w, h, b, r, r);
  mat_mul(w, h, r, mask, r);

  // upper part of a: 
  // a_u = r * r
  mat_mul(w, h, r, r, a_utosum);

  // lower part of a:
  // a_b = laplacian_operator(r);
  laplacian_gpu(w, h, r, a_btosum);
  // a_b = r * laplacian_operator(r);
  mat_mul(w, h, r, a_btosum, a_btosum);
}

__global__ void poisson_kern_p2(
  size_t w,
  size_t h,
  float *a_utosum_r,
  float *a_btosum_r,
  float *a_utosum_g,
  float *a_btosum_g,
  float *a_utosum_b,
  float *a_btosum_b,
  float *r,
  float *x
) {
  float a = (a_utosum_r[0] + a_utosum_g[0] + a_utosum_b[0]) / (a_btosum_r[0] + a_btosum_g[0] + a_btosum_b[0]);

  mat_mul_scalar(w, h, r, a, r);
  mat_add(w, h, x, r, x);
}

HostImage* run_poisson(HostImage *bg, HostImage *fg, HostImage *mask, size_t iterations) {
  findCudaDevice(0, NULL);
  set_kern();

  size_t block_rem_w = bg->w % LAP_BLOCK_X;
  size_t block_rem_h = bg->h % LAP_BLOCK_Y;

  size_t new_w = bg->w - block_rem_w + LAP_BLOCK_X;
  size_t new_h = bg->h - block_rem_h + LAP_BLOCK_Y;

  size_t block_cnt_w = new_w / LAP_BLOCK_X;
  size_t block_cnt_h = new_h / LAP_BLOCK_Y;
 
  dim3 block_cnt(block_cnt_w, block_cnt_h);

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
  float *d_maskedfg_r, *d_maskedfg_g, *d_maskedfg_b;
  float *d_res_r, *d_res_g, *d_res_b;

  checkCudaErrors(cudaMalloc(&d_maskedfg_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_maskedfg_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&d_maskedfg_b, sizeof(float) * new_w * new_h));
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
  
  checkCudaErrors(cudaDeviceSynchronize());
  
  dim3 block(LAP_BLOCK_X, LAP_BLOCK_Y);

  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_fg_r, d_mask_r, d_maskedfg_r, 0.0f);
  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_fg_g, d_mask_g, d_maskedfg_g, 0.0f);
  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_fg_b, d_mask_b, d_maskedfg_b, 0.0f);
  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_bg_r, d_mask_r, d_bg_r, 1.0f);
  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_bg_g, d_mask_g, d_bg_g, 1.0f);
  apply_mask_to_image<<<block_cnt, block>>>(new_w, new_h, d_bg_b, d_mask_b, d_bg_b, 1.0f);

  HostImage *debug = new_host_image(new_w, new_h);
  image_from_gpu(debug, d_bg_r, d_bg_g, d_bg_b);
  save_host_image(debug, "debug_bg.png");

  float *b_r, *b_g, *b_b, *r_r, *r_g, *r_b, *x_r, *x_g, *x_b, *au_r, *au_g, *au_b, *ad_r, *ad_g, *ad_b;

  checkCudaErrors(cudaMalloc(&b_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&b_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&b_b, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&r_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&r_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&r_b, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&x_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&x_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&x_b, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&au_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&au_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&au_b, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&ad_r, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&ad_g, sizeof(float) * new_w * new_h));
  checkCudaErrors(cudaMalloc(&ad_b, sizeof(float) * new_w * new_h));
 
  laplacian(new_w, new_h, d_fg_r, b_r);
  laplacian(new_w, new_h, d_fg_g, b_g);
  laplacian(new_w, new_h, d_fg_b, b_b);

  image_from_gpu(debug, b_r, b_g, b_b);
 
  initial_x<<<block_cnt, block>>>(new_w, new_h, d_maskedfg_r, d_bg_r, x_r);
  initial_x<<<block_cnt, block>>>(new_w, new_h, d_maskedfg_g, d_bg_g, x_g);
  initial_x<<<block_cnt, block>>>(new_w, new_h, d_maskedfg_b, d_bg_b, x_b);

  checkCudaErrors(cudaDeviceSynchronize());

  for(int i = 0; i < iterations; ++i) {
    poisson_kern_p1<<<block_cnt, block>>>(
      new_w,
      new_h,
      d_mask_r,
      au_r,
      ad_r,
      b_r,
      r_r,
      x_r
    );
    poisson_kern_p1<<<block_cnt, block>>>(
      new_w,
      new_h,
      d_mask_g,
      au_g,
      ad_g,
      b_g,
      r_g,
      x_g
    );
    poisson_kern_p1<<<block_cnt, block>>>(
      new_w,
      new_h,
      d_mask_b,
      au_b,
      ad_b,
      b_b,
      r_b,
      x_b
    );
    checkCudaErrors(cudaDeviceSynchronize());

    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, au_r);
    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, au_g);
    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, au_b);
    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, ad_r);
    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, ad_g);
    mat_sum_all<<<block_cnt, block>>>(new_w, new_h, ad_b);
    checkCudaErrors(cudaDeviceSynchronize());
    
    poisson_kern_p2<<<block_cnt, block>>>(
      new_w,
      new_h,
      au_r,
      ad_r,
      au_g,
      ad_g,
      au_b,
      ad_b,
      r_r,
      x_r
    );
    poisson_kern_p2<<<block_cnt, block>>>(
      new_w,
      new_h,
      au_r,
      ad_r,
      au_g,
      ad_g,
      au_b,
      ad_b,
      r_g,
      x_g
    );
    poisson_kern_p2<<<block_cnt, block>>>(
      new_w,
      new_h,
      au_r,
      ad_r,
      au_g,
      ad_g,
      au_b,
      ad_b,
      r_b,
      x_b
    );

    checkCudaErrors(cudaDeviceSynchronize());
  }

  clip_result<<<block_cnt, block>>>(new_w, new_h, x_r, x_r);
  clip_result<<<block_cnt, block>>>(new_w, new_h, x_g, x_g);
  clip_result<<<block_cnt, block>>>(new_w, new_h, x_b, x_b);
  checkCudaErrors(cudaDeviceSynchronize());

  HostImage *result = new_host_image(new_w, new_h);
  image_from_gpu(
    result,
    x_r,
    x_g,
    x_b
  );
  checkCudaErrors(cudaDeviceSynchronize());

  drop_host_image(aligned_bg);
  drop_host_image(aligned_fg);
  drop_host_image(aligned_mask);

  checkCudaErrors(cudaFree(x_r));
  checkCudaErrors(cudaFree(x_g));
  checkCudaErrors(cudaFree(x_b));
  checkCudaErrors(cudaFree(r_r));
  checkCudaErrors(cudaFree(r_g));
  checkCudaErrors(cudaFree(r_b));
  checkCudaErrors(cudaFree(au_r));
  checkCudaErrors(cudaFree(au_g));
  checkCudaErrors(cudaFree(au_b));
  checkCudaErrors(cudaFree(ad_r));
  checkCudaErrors(cudaFree(ad_g));
  checkCudaErrors(cudaFree(ad_b));
  checkCudaErrors(cudaFree(d_maskedfg_r));
  checkCudaErrors(cudaFree(d_maskedfg_g));
  checkCudaErrors(cudaFree(d_maskedfg_b));
  checkCudaErrors(cudaFree(d_fg_r));
  checkCudaErrors(cudaFree(d_fg_g));
  checkCudaErrors(cudaFree(d_fg_b));
  checkCudaErrors(cudaFree(d_bg_r));
  checkCudaErrors(cudaFree(d_bg_g));
  checkCudaErrors(cudaFree(d_bg_b));
  
  cudaDeviceReset();
  return result;
}
