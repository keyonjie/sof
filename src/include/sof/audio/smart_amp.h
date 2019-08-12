// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright(c) 2016 Intel Corporation. All rights reserved.
//
// Author: Bartosz Kokoszko <bartoszx.kokoszko@linux.intel.com>

#ifndef __SOF_AUDIO_SMART_AMP_H__
#define __SOF_AUDIO_SMART_AMP_H__

#include <sof/platform.h>

#define SMART_AMP_MAX_STREAM_CHAN   8

enum {
        SMART_AMP_MODE_PASSTHROUGH,
        SMART_AMP_MODE_PROC_FEEDBACK,
        SMART_AMP_MODE_NOCODEC_FEEDBACK,
        SMART_AMP_MODE_AMOUNT
};

struct sof_smart_amp_config {
        uint32_t size;
        int8_t source_ch_map[PLATFORM_MAX_CHANNELS];
        int8_t feedback_ch_map[PLATFORM_MAX_CHANNELS];
};

#endif /* __SOF_AUDIO_SMART_AMP_H__ */
