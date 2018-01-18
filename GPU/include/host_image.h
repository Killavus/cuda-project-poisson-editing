#ifndef __HOST_IMAGE_H__
#define __HOST_IMAGE_H__
typedef struct hostimage {
  size_t w;
  size_t h;

  float *r_data;
  float *g_data;
  float *b_data;
} HostImage;

HostImage* new_host_image(size_t w, size_t h);
HostImage* load_host_image(const char *filename);
HostImage* resize_host_image(
  HostImage *src,
  size_t new_w,
  size_t new_h,
  size_t move_w,
  size_t move_h
);

void drop_host_image(HostImage *img);
void save_host_image(HostImage *img, const char *filename);
#endif // __HOST_IMAGE_H__
