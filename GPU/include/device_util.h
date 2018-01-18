#ifndef __DEVICE_UTIL_H__
#define __DEVICE_UTIL_H__
#include "include/host_image.h"

void image_to_gpu(
  HostImage *img,
  float **d_rdata,
  float **d_gdata,
  float **d_bdata
);

void image_from_gpu(
  HostImage *img,
  float *d_rdata,
  float *d_gdata,
  float *d_bdata
);

__device__ void masked_image(
  int w,
  int h,
  float *mask,
  float *chan_data,
  float *result,
  float mask_val,
  float spec_val
);

__device__ void paste_masked(
  int w,
  int h,
  float *bg_data,
  float *fg_data,
  float *result,
  float spec_val
);

__device__ void mat_mul(
  int w,
  int h,
  float *a,
  float *b,
  float *r
);

__device__ void mat_add(
  int w,
  int h,
  float *a,
  float *b,
  float *r
);

__device__ void mat_sub(
  int w,
  int h,
  float *a,
  float *b,
  float *r
);

__device__ void mat_add_scalar(
  int w,
  int h,
  float *m,
  float s,
  float *r
);

__device__ void mat_mul_scalar(
  int w,
  int h,
  float *m,
  float s,
  float *r
);

__global__ void mat_sum_all(
  int w,
  int h,
  float *m
);

__device__ void mat_clip(
  int w,
  int h,
  float *m,
  float min_val,
  float max_val,
  float *r
);

#endif //__DEVICE_UTIL_H__
