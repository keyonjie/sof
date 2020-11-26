/* SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright(c) 2017 Intel Corporation. All rights reserved.
 *
 * Author: Amery Song <chao.song@intel.com>
 *	   Keyon Jie <yang.jie@linux.intel.com>
 */

#include <sof/common.h>

#define FFT_SIZE_MAX	1024

struct icomplex32 {
	int32_t real;
	int32_t imag;
};

void fft(struct icomplex32 *inb, struct icomplex32 *outb, uint32_t size, bool ifft);
void fft_real(struct comp_buffer *src, struct comp_buffer *dst, uint32_t size);
void ifft_complex(struct comp_buffer *src, struct comp_buffer *dst, uint32_t size);
