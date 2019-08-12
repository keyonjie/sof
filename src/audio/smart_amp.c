// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright(c) 2016 Intel Corporation. All rights reserved.
//
// Author: Bartosz Kokoszko <bartoszx.kokoszko@linux.intel.com>

#include <sof/audio/smart_amp.h>
#include <sof/audio/component.h>
#include <sof/trace/trace.h>
#include <sof/drivers/ipc.h>
#include <sof/ut.h>

static const struct comp_driver comp_smart_amp;

struct smart_amp_data {
	struct sof_smart_amp_config config;

	struct comp_buffer *source_buf; /**< stream source buffer */
	struct comp_buffer *feedback_buf; /**< feedback source buffer */
	struct comp_buffer *sink_buf; /**< sink buffer */

	uint32_t in_channels;
	uint32_t out_channels;
};

static struct comp_dev *smart_amp_new(const struct comp_driver *drv,
				      struct sof_ipc_comp *comp)
{
	struct comp_dev *dev;
	struct sof_ipc_comp_process *sa;
	struct sof_ipc_comp_process *ipc_sa =
		(struct sof_ipc_comp_process *)comp;
	struct smart_amp_data *sad;
	struct sof_smart_amp_config *cfg;
	size_t bs;

	comp_cl_info(&comp_smart_amp, "smart_amp_new()");

	if (IPC_IS_SIZE_INVALID(ipc_sa->config)) {
		IPC_SIZE_ERROR_TRACE(TRACE_CLASS_SMART_AMP, ipc_sa->config);
		return NULL;
	}

	dev = rzalloc(SOF_MEM_ZONE_RUNTIME, 0, SOF_MEM_CAPS_RAM,
		      COMP_SIZE(struct sof_ipc_comp_process));
	if (!dev)
		return NULL;
	dev->drv = drv;

	sa = (struct sof_ipc_comp_process *)&dev->comp;

	assert(!memcpy_s(sa, sizeof(*sa), ipc_sa,
	       sizeof(struct sof_ipc_comp_process)));

	sad = rzalloc(SOF_MEM_ZONE_RUNTIME, 0, SOF_MEM_CAPS_RAM, sizeof(*sad));

	if (!sad) {
		rfree(dev);
		return NULL;
	}

	comp_set_drvdata(dev, sad);

	cfg = (struct sof_smart_amp_config *)ipc_sa->data;
	bs = ipc_sa->size;

	if ((bs > 0) && (bs < sizeof(struct sof_smart_amp_config))) {
		comp_err(dev, "smart_amp_new(): failed to apply config");

		if (sad)
			rfree(sad);
		rfree(sad);
		return NULL;
	}

	memcpy_s(&sad->config, sizeof(struct sof_smart_amp_config), cfg, bs);

	dev->state = COMP_STATE_READY;

	return dev;
}

static int smart_amp_set_config(struct comp_dev *dev,
				struct sof_ipc_ctrl_data *cdata)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	struct sof_smart_amp_config *cfg;
	size_t bs;

	comp_info(dev, "smart_amp_set_config()");

	/* Copy new config, find size from header */
	cfg = (struct sof_smart_amp_config *)cdata->data->data;
	bs = cfg->size;

	comp_info(dev, "smart_amp_set_config(), blob size = %u", bs);

	if (bs != sizeof(struct sof_smart_amp_config)) {
		comp_err(dev, "smart_amp_set_config(): invalid blob size");
		return -EINVAL;
	}

	memcpy_s(&sad->config, sizeof(struct sof_smart_amp_config), cfg, sizeof(struct sof_smart_amp_config));

	return 0;
}


/* used to pass standard and bespoke commands (with data) to component */
static int smart_amp_cmd(struct comp_dev *dev, int cmd, void *data,
			 int max_data_size)
{
	struct sof_ipc_ctrl_data *cdata = data;

	comp_info(dev, "smart_amp_cmd()");

	comp_info(dev, "smart_amp_cmd(): cmd: %d", cmd);

	switch (cmd) {
	case COMP_CMD_SET_DATA:
		return smart_amp_set_config(dev, cdata);
	default:
		return -EINVAL;
	}
}

static void smart_amp_free(struct comp_dev *dev)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);

	comp_info(dev, "smart_amp_free()");

	rfree(sad);
	rfree(dev);
}

static int smart_amp_verify_params(struct comp_dev *dev,
				   struct sof_ipc_stream_params *params)
{
	int ret;

	comp_info(dev, "smart_amp_verify_params()");

	ret = comp_verify_params(dev, BUFF_PARAMS_CHANNELS, params);
	if (ret < 0) {
		comp_err(dev, "volume_verify_params() error: comp_verify_params() failed.");
		return ret;
	}

	return 0;
}

static int smart_amp_params(struct comp_dev *dev,
			    struct sof_ipc_stream_params *params)
{
	int err;

	comp_info(dev, "smart_amp_params()");

	err = smart_amp_verify_params(dev, params);
	if (err < 0) {
		comp_err(dev, "smart_amp_params(): pcm params verification failed.");
		return -EINVAL;
	}

	return 0;
}

static int smart_amp_trigger(struct comp_dev *dev, int cmd)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	int ret = 0;

	comp_info(dev, "smart_amp_trigger(), command = %u", cmd);

	ret = comp_set_state(dev, cmd);

	if (ret == COMP_STATUS_STATE_ALREADY_SET)
		ret = PPL_STATUS_PATH_STOP;

	switch (cmd) {
	case COMP_TRIGGER_START:
	case COMP_TRIGGER_RELEASE:
		buffer_zero(sad->feedback_buf);
		break;
	case COMP_TRIGGER_PAUSE:
	case COMP_TRIGGER_STOP:
		break;
	default:
		break;
	}

	return ret;
}

static int smart_amp_process_s16(struct comp_dev *dev,
				 const struct audio_stream *source,
				 const struct audio_stream *sink,
				 uint32_t frames, int8_t *chan_map)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	int16_t *src;
	int16_t *dest;
	uint32_t in_buff_frag = 0;
	uint32_t out_buff_frag = 0;
	int i;
	int j;

	comp_info(dev, "smart_amp_process_s16()");

	for (i = 0; i < frames; i++) {
		for (j = 0 ; j < sad->out_channels; j++) {
			if (chan_map[j] != -1) {
				src = audio_stream_read_frag_s16(source, in_buff_frag + chan_map[j]);	
				dest = audio_stream_write_frag_s16(sink, out_buff_frag);
				*dest = *src;
			}
			out_buff_frag++;
		}
		in_buff_frag += source->channels;
	}
	return 0;
}

static int smart_amp_process_s32(struct comp_dev *dev,
				 const struct audio_stream *source,
				 const struct audio_stream *sink,
				 uint32_t frames, int8_t *chan_map)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	int32_t *src;
	int32_t *dest;
	uint32_t in_buff_frag = 0;
	uint32_t out_buff_frag = 0;
	int i;
	int j;

	comp_info(dev, "smart_amp_process_s32()");

	for (i = 0; i < frames; i++) {
		for (j = 0 ; j < sad->out_channels; j++) {
			if (chan_map[j] != -1) {
				src = audio_stream_read_frag_s32(source, in_buff_frag + chan_map[j]);	
				dest = audio_stream_write_frag_s32(sink, out_buff_frag);
				*dest = *src;
			}
			out_buff_frag++;
		}
		in_buff_frag += source->channels;
	}

	return 0;
}

static int smart_amp_process(struct comp_dev *dev, uint32_t frames,
			     struct comp_buffer *source,
			     struct comp_buffer *sink, int8_t *chan_map)
{
	int ret = 0;

	switch (source->stream.frame_fmt) {
	case SOF_IPC_FRAME_S16_LE:
		ret = smart_amp_process_s16(dev, &source->stream, &sink->stream,
					    frames, chan_map);
		break;
	case SOF_IPC_FRAME_S24_4LE:
	case SOF_IPC_FRAME_S32_LE:
		ret = smart_amp_process_s32(dev, &source->stream, &sink->stream,
					    frames, chan_map);

		break;
	default:
		comp_err(dev, "smart_amp_process() error: not supported frame format");
		return -EINVAL;
	}

	return ret;
}

static int smart_amp_copy(struct comp_dev *dev)
{
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	uint32_t avail_passthrough_frames;
	uint32_t avail_feedback_frames;
	uint32_t avail_frames;
	uint32_t source_bytes;
	uint32_t sink_bytes;
	uint32_t feedback_bytes;
	uint32_t source_flags;
	uint32_t sink_flags;
	uint32_t feedback_flags;
	int ret = 0;

	comp_dbg(dev, "smart_amp_copy()");

	buffer_lock(sad->source_buf, &source_flags);
	buffer_lock(sad->sink_buf, &sink_flags);

	/* available bytes and samples calculation */
	avail_passthrough_frames = audio_stream_avail_frames(&sad->source_buf->stream,
						 &sad->sink_buf->stream);

	buffer_unlock(sad->source_buf, source_flags);
	buffer_unlock(sad->sink_buf, sink_flags);

	avail_frames = avail_passthrough_frames;

	comp_dbg(dev, "smart_amp_copy(): avail_passthrough_frames: %d", avail_passthrough_frames);

	buffer_lock(sad->feedback_buf, &feedback_flags);
	if (sad->feedback_buf->source->state == dev->state) {
		/* feedback */
		avail_feedback_frames = sad->feedback_buf->stream.avail /
			audio_stream_frame_bytes(&sad->feedback_buf->stream);

		avail_frames = MIN(avail_passthrough_frames, avail_feedback_frames);

		feedback_bytes = avail_frames *
			audio_stream_frame_bytes(&sad->feedback_buf->stream);

		buffer_unlock(sad->feedback_buf, feedback_flags);

		comp_dbg(dev, "smart_amp_copy(): processing %d feedback bytes",
		  	feedback_bytes);

		smart_amp_process(dev, avail_frames, sad->feedback_buf, sad->sink_buf,
			  sad->config.feedback_ch_map);

		comp_update_buffer_consume(sad->feedback_buf, feedback_bytes);
	}

	/* bytes calculation */
	buffer_lock(sad->source_buf, &source_flags);
	source_bytes = avail_frames *
		audio_stream_frame_bytes(&sad->source_buf->stream);
	buffer_unlock(sad->source_buf, source_flags);
	
	buffer_lock(sad->sink_buf, &sink_flags);
	sink_bytes = avail_frames *
		audio_stream_frame_bytes(&sad->sink_buf->stream);
	buffer_unlock(sad->sink_buf, sink_flags);

	/* process data */
	smart_amp_process(dev, avail_frames, sad->source_buf, sad->sink_buf,
			  sad->config.source_ch_map);

	/* source/sink buffer pointers update */
	comp_update_buffer_consume(sad->source_buf, source_bytes);
	comp_update_buffer_produce(sad->sink_buf, sink_bytes);

	return ret;
}

static int smart_amp_reset(struct comp_dev *dev)
{
	comp_info(dev, "smart_amp_reset()");

	comp_set_state(dev, COMP_TRIGGER_RESET);

	return 0;
}

static int smart_amp_prepare(struct comp_dev *dev)
{
	struct sof_ipc_comp_process *ipc_sa =
		(struct sof_ipc_comp_process *)&dev->comp;
	struct smart_amp_data *sad = comp_get_drvdata(dev);
	struct comp_buffer *source_buffer;
	struct list_item *blist;
	int ret;

	(void)ipc_sa;

	comp_info(dev, "smart_amp_prepare()");

	ret = comp_set_state(dev, COMP_TRIGGER_PREPARE);
	if (ret < 0)
		return ret;

	if (ret == COMP_STATUS_STATE_ALREADY_SET)
		return PPL_STATUS_PATH_STOP;

	/* searching for stream and feedback source buffers */
	list_for_item(blist, &dev->bsource_list) {
		source_buffer = container_of(blist, struct comp_buffer,
					     sink_list);

		if (source_buffer->source->comp.type == SOF_COMP_DEMUX)
			sad->feedback_buf = source_buffer;
		else
			sad->source_buf = source_buffer;
	}

	sad->sink_buf = list_first_item(&dev->bsink_list, struct comp_buffer,
					source_list);

	sad->in_channels = sad->source_buf->stream.channels;
	sad->out_channels = sad->sink_buf->stream.channels;

	sad->feedback_buf->stream.channels = 8;
	sad->feedback_buf->stream.frame_fmt = 2;

	return 0;
}

static const struct comp_driver comp_smart_amp = {
	.type = SOF_COMP_SMART_AMP,
	.ops = {
		.create = smart_amp_new,
		.free = smart_amp_free,
		.params = smart_amp_params,
		.prepare = smart_amp_prepare,
		.cmd = smart_amp_cmd,
		.trigger = smart_amp_trigger,
		.copy = smart_amp_copy,
		.reset = smart_amp_reset,
	},
};

static SHARED_DATA struct comp_driver_info comp_smart_amp_info = {
	.drv = &comp_smart_amp,
};

static void sys_comp_smart_amp_init(void)
{
	comp_register(platform_shared_get(&comp_smart_amp_info,
					  sizeof(comp_smart_amp_info)));
}

DECLARE_MODULE(sys_comp_smart_amp_init);
