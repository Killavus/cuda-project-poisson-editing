#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <MagickWand/MagickWand.h>
#include "include/host_image.h"
#define ThrowWandException(wand) \
{ \
  char \
    *description; \
 \
  ExceptionType \
    severity; \
 \
  description=MagickGetException(wand,&severity); \
  (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  description=(char *) MagickRelinquishMemory(description); \
  exit(-1); \
}

HostImage* new_host_image(size_t w, size_t h) {
  HostImage *new_image = (HostImage*) malloc(sizeof(HostImage)); 

  new_image->w = w;
  new_image->h = h;

  size_t color_layer_size = w * h * sizeof(float);

  new_image->r_data = (float*) malloc(color_layer_size);
  new_image->g_data = (float*) malloc(color_layer_size);
  new_image->b_data = (float*) malloc(color_layer_size);

  memset(new_image->r_data, 0.0f, color_layer_size);
  memset(new_image->g_data, 0.0f, color_layer_size);
  memset(new_image->b_data, 0.0f, color_layer_size);

  return new_image;
}

HostImage* load_host_image(const char *filename) {
  MagickWandGenesis();
  char *qdepth = MagickQueryConfigureOption("QuantumDepth");
  unsigned int depthmod = 1<<atoi(qdepth);
  
  MagickBooleanType status;
  MagickWand *wand;
  PixelWand **pixels;
  PixelInfo pixel;
  PixelIterator *iter;

  wand = NewMagickWand();
  status = MagickReadImage(wand, filename);

  if (status == MagickFalse) {
    ThrowWandException(wand);
  }

  iter = NewPixelIterator(wand);

  if (iter == (PixelIterator*) NULL) {
    ThrowWandException(wand);
  }

  size_t w = MagickGetImageWidth(wand);
  size_t h = MagickGetImageHeight(wand);

  HostImage *img = new_host_image(w, h);
  int with_alpha = MagickGetImageAlphaChannel(wand) == MagickTrue ? 1 : 0;

  int i, j;
  for (i = 0; i < h; ++i) {
    pixels = PixelGetNextIteratorRow(iter, &w);

    if (pixels == (PixelWand**) NULL) {
      break;
    }


    for(j = 0; j < w; ++j) {
      PixelGetMagickColor(pixels[j], &pixel);
      img->r_data[i * w + j] = pixel.red / depthmod;
      img->g_data[i * w + j] = pixel.green / depthmod;
      img->b_data[i * w + j] = pixel.blue / depthmod;
    }
  }

  iter = DestroyPixelIterator(iter);
  wand = DestroyMagickWand(wand);
  MagickWandTerminus();

  return img;
}

void save_host_image(HostImage *img, const char *filename) {
  MagickWandGenesis();
  char *qdepth = MagickQueryConfigureOption("QuantumDepth");
  unsigned int depthmod = 1<<atoi(qdepth);
  MagickBooleanType status;
  MagickWand *wand;
  PixelIterator *iter;
  PixelWand **pixels;

  wand = NewMagickWand();
  MagickSetSize(wand, img->w, img->h);
  status = MagickReadImage(wand, "xc:black");

  if (status == MagickFalse) {
    ThrowWandException(wand);
  }

  iter = NewPixelIterator(wand);

  int y, x;
  size_t w_;
  for(y = 0; y < img->h; ++y) {
    pixels = PixelGetNextIteratorRow(iter, &w_);
    for(x = 0; x < img->w; ++x) {
      PixelSetRedQuantum(pixels[x], img->r_data[y * w_ + x] * depthmod);
      PixelSetGreenQuantum(pixels[x], img->g_data[y * w_ + x] * depthmod);
      PixelSetBlueQuantum(pixels[x], img->b_data[y * w_ + x] * depthmod);
    }
    PixelSyncIterator(iter);
  }
  
  MagickWriteImage(wand, filename);

  iter = DestroyPixelIterator(iter);
  wand = DestroyMagickWand(wand);
  MagickWandTerminus();
}

void drop_host_image(HostImage *img) {
  free(img->r_data);
  free(img->g_data);
  free(img->b_data);
  free(img);
}

HostImage* resize_host_image(
  HostImage *src,
  size_t new_w,
  size_t new_h,
  size_t move_w,
  size_t move_h
) {
  HostImage *new_image = new_host_image(new_w, new_h);

  size_t i, j;
  for(i = 0; i < src->h; ++i) {
    if (i + move_h >= new_h) break;

    for(j = 0; j < src->w; ++j) {
      if (j + move_w >= new_w) break; 
      size_t index = (i + move_h) * new_image->w + j + move_w;

      new_image->r_data[index] = src->r_data[i * src->w + j];
      new_image->g_data[index] = src->g_data[i * src->w + j];
      new_image->b_data[index] = src->b_data[i * src->w + j];
    }
  }

  return new_image;
}
