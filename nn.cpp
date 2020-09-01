#include "nn.h"

int streamConv(hls::stream<int8_channel> &inStreamImage, hls::stream<int8_channel> &outStream, q7_t kernel[dim_kernel * dim_kernel * ch_in * ch_out], q7_t bias[ch_out]){
#pragma HLS INTERFACE axis port=inStreamImage
#pragma HLS INTERFACE axis port=outStream
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=bias bundle=BIAS_BUS
#pragma HLS INTERFACE s_axilite port=kernel bundle=KERNEL_BUS
	hls::LineBuffer<dim_kernel, dim_in + padding * 2, q7_t> imageBufferR;
	hls::LineBuffer<dim_kernel, dim_in + padding * 2, q7_t> imageBufferG;
	hls::LineBuffer<dim_kernel, dim_in + padding * 2, q7_t> imageBufferB;

	for(int i = 0; i < dim_kernel; i++){
		for(int j = 0; j < dim_in + padding * 2; j++){
#pragma HLS PIPELINE
			imageBufferR.shift_up(j);
			imageBufferR.insert_top(0,j);
			imageBufferG.shift_up(j);
			imageBufferG.insert_top(0,j);
			imageBufferB.shift_up(j);
			imageBufferB.insert_top(0,j);
		}
	}
	hls::Window<dim_kernel, dim_kernel, q31_t> windowR;
	hls::Window<dim_kernel, dim_kernel, q31_t> windowG;
	hls::Window<dim_kernel, dim_kernel, q31_t> windowB;

	int idxRow = 0;
	int idxCol = 0;
	int idxCh = 0;
	int countWait = 0;
	q7_t convolvedNum = 0;
	q31_t valOut = 0;
	q7_t pixel = 0;
	int8_channel inChannel;
	int8_channel outChannel;

	int outputSize = 36 * 36;
	for(int idx = 0; idx < outputSize; idx++){
//#pragma HLS PIPELINE
		//read data
		if((idxCol >= 2) &&(idxCol <= 33)){
			if(idxRow < dim_in){
				//read R data and save pixel into buffer
				inChannel = inStreamImage.read();
				pixel = inChannel.data;
				imageBufferR.shift_up(idxCol);
				imageBufferR.insert_top(pixel, idxCol);
				//read G data and save pixel into buffer
				pixel = inStreamImage.read().data;
				imageBufferG.shift_up(idxCol);
				imageBufferG.insert_top(pixel, idxCol);
				//read B data and save pixel into buffer
				pixel = inStreamImage.read().data;
				imageBufferB.shift_up(idxCol);
				imageBufferB.insert_top(pixel, idxCol);
			}else{
				//padding 0
				pixel = 0;
				imageBufferR.shift_up(idxCol);
				imageBufferR.insert_top(pixel, idxCol);
				//padding 0
				pixel = 0;
				imageBufferG.shift_up(idxCol);
				imageBufferG.insert_top(pixel, idxCol);
				//padding 0
				pixel = 0;
				imageBufferB.shift_up(idxCol);
				imageBufferB.insert_top(pixel, idxCol);
			}
		}

		//do convolution
		if((idxCol >= (dim_kernel - 1)) && (idxRow >= 2) && (idxRow <= 33)){
			for(idxCh = 0; idxCh < 32; idxCh++){
#pragma HLS PIPELINE
				valOut = NN_ROUND(out_shift) + bias[idxCh]<<bias_shift;
				for(int idxWinRow = 0; idxWinRow < dim_kernel; idxWinRow++)
				{
//#pragma HLS PIPELINE
					for(int idxWinCol = 0; idxWinCol < dim_kernel; idxWinCol++)
					{
						q31_t val = 0;
						val = imageBufferR.getval(idxWinRow, idxWinCol + convolvedNum);
						val *= kernel[idxCh * dim_kernel * dim_kernel * ch_in+ (idxWinRow * dim_kernel + idxWinCol) * ch_in];
						windowR.insert(val, idxWinRow, idxWinCol);

						val = imageBufferG.getval(idxWinRow, idxWinCol + convolvedNum);
						val *= kernel[idxCh * dim_kernel * dim_kernel * ch_in+ (idxWinRow * dim_kernel + idxWinCol) * ch_in + 1];
						windowG.insert(val, idxWinRow, idxWinCol);

						val = imageBufferB.getval(idxWinRow, idxWinCol + convolvedNum);
						val *= kernel[idxCh * dim_kernel * dim_kernel * ch_in+ (idxWinRow * dim_kernel + idxWinCol) * ch_in + 2];
						windowB.insert(val, idxWinRow, idxWinCol);
					}
				}
				valOut += sumWindow(&windowR) + sumWindow(&windowG) + sumWindow(&windowB);
				valOut = (q7_t) SSAT((valOut >> out_shift), 8);
				outChannel.data = valOut;
				outChannel.keep = inChannel.keep;
				outChannel.strb = inChannel.strb;
				outChannel.user = inChannel.user;
				outChannel.id = inChannel.id;
				outChannel.dest = inChannel.dest;
				if((idxCol == 35) && (idxRow == 33) && (idxCh == 31)){
					outChannel.last = 1;
				}else{
					outChannel.last = 0;
				}
				// Send to the stream (Block if the FIFO receiver is full)
				outStream.write(outChannel);
			}
			convolvedNum++;
		}


		idxCol++;
		if(idxCol == 36){
			idxCol = 0;
			idxRow++;
			convolvedNum = 0;
		}
	}
	return 0;
}

q31_t sumWindow(hls::Window<dim_kernel,dim_kernel,q31_t> *window)
{
	q31_t accumulator = 0;

	// Iterate on the window multiplying and accumulating the kernel and sampling window
	for (int idxRow = 0; idxRow < dim_kernel; idxRow++)
	{
//#pragma HLS PIPELINE
		for (int idxCol = 0; idxCol < dim_kernel; idxCol++)
		{
			accumulator += (q31_t)window->getval(idxRow,idxCol);
		}
	}
	return accumulator;
}
