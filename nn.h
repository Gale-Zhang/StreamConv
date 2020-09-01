#ifndef NN_H
#define NN_H

#include "global.h"
#include "hls_video.h"

#define dim_kernel 5
#define padding 2
#define stride 1

#define bias_shift 0
#define out_shift 9

#define dim_in 32
#define ch_in 3

#define dim_out 32
#define ch_out 32

int streamConv(hls::stream<int8_channel> &inStreamImage, hls::stream<int8_channel> &outStream, q7_t kernel[dim_kernel * dim_kernel * ch_in * ch_out], q7_t bias[ch_out]);
q31_t sumWindow(hls::Window<dim_kernel,dim_kernel,q31_t> *window);

#endif
