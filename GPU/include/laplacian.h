#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__
#define LAP_BLOCK_X 16
#define LAP_BLOCK_Y 8

void set_kern();
void laplacian(size_t w, size_t h, float *src, float *dst);
__device__ void laplacian_gpu(size_t w, size_t h, float *src, float *dst);
#endif //__LAPLACIAN_H__
