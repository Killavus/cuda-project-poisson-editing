#ifndef __POISSON_H__
#define __POISSON_H__
#include "include/host_image.h"

HostImage* run_poisson(HostImage *bg, HostImage *fg, HostImage *mask, size_t iterations); 
#endif //__POISSON_H__
