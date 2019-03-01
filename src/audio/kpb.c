/*
 * Copyright (c) 2019, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the Intel Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Marcin Rajwa <marcin.rajwa@linux.intel.com>
 *
 * A key phrase buffer component.
 */

#include <stdint.h>
#include <uapi/ipc/topology.h>
#include <sof/ipc.h>
#include <sof/audio/component.h>
#include <sof/audio/kpb.h>
#include <sof/list.h>
#include <sof/audio/buffer.h>

static int kpb_register_client(struct comp_dev *dev, struct kpb_client *cli);
static int kpb_begin_draining(struct comp_dev *dev, uint8_t client_id);
//static void kpb_buffer_data(struct comp_data *kpb, struct comp_buffer *source);

/**
 * \brief Create a key phrase buffer component.
 * \param[in] comp - generic ipc component pointer.
 *
 * \return: a pointer to newly created KPB component.
 */
static struct comp_dev *kpb_new(struct sof_ipc_comp *comp)
{
	struct comp_dev *dev;
	struct sof_ipc_comp_process *kpb;
	struct sof_ipc_comp_process *ipc_kpb =
		(struct sof_ipc_comp_process *)comp;
	struct comp_data *cd;

	trace_kpb("kpb_new()");

	if (IPC_IS_SIZE_INVALID(ipc_kpb->config)) {
		IPC_SIZE_ERROR_TRACE(TRACE_CLASS_KPB, ipc_kpb->config);
		return NULL;
	}
#if 0
	/* Validate input parameters */
	if (ipc_kpb->channels > KPB_MAX_SUPPORTED_CHANNELS) {
		trace_kpb_error("kpb_new() error: "
		"nr of channels exceeded the limit");
		return NULL;
	}

	if (ipc_kpb->history_depth > KPB_MAX_BUFFER_SIZE) {
		trace_kpb_error("kpb_new() error: "
		"history depth exceeded the limit");
		return NULL;
	}

	if (ipc_kpb->sampling_freq != KPB_SAMPLNG_FREQUENCY) {
		trace_kpb_error("kpb_new() error: "
		"requested sampling frequency not supported");
		return NULL;
	}

	if (ipc_kpb->sampling_width != KPB_SAMPLING_WIDTH) {
		trace_kpb_error("kpb_new() error: "
		"requested sampling width not supported");
		return NULL;
	}
#endif
	dev = rzalloc(RZONE_RUNTIME, SOF_MEM_CAPS_RAM,
		COMP_SIZE(struct sof_ipc_comp_process));
	if (!dev)
		return NULL;

	kpb = (struct sof_ipc_comp_process *)&dev->comp;
	memcpy(kpb, ipc_kpb, sizeof(struct sof_ipc_comp_process));

	cd = rzalloc(RZONE_RUNTIME, SOF_MEM_CAPS_RAM, sizeof(*cd));
	if (!cd) {
		rfree(dev);
		return NULL;
	}
#if 0
	cd->history_depth = ipc_kpb->history_depth;
#endif

	/* by default KPB will copy buffers to the host comp */
	cd->sink_type = SOF_COMP_HOST;

	comp_set_drvdata(dev, cd);
	dev->state = COMP_STATE_READY;

	return dev;
}

static void kpb_init_kd_dev(struct comp_dev *dev)
{
	struct comp_data *kpb = comp_get_drvdata(dev);
	struct list_item *buffer_list = comp_buffer_list(dev, PPL_DIR_DOWNSTREAM);
	struct list_item *clist;
	struct comp_buffer *buffer;
	struct comp_dev *buffer_comp;

	if (kpb->kd_dev)
		return;

	list_for_item(clist, buffer_list) {
		buffer = buffer_from_list(clist, struct comp_buffer, PPL_DIR_DOWNSTREAM);

		buffer_comp = buffer_get_comp(buffer, PPL_DIR_DOWNSTREAM);

		/* don't go further if this component is not connected */
		if (!buffer_comp || buffer_comp->comp.type != SOF_COMP_SELECTOR)
			continue;

		kpb->kd_dev = buffer_comp;
	}
}

/* set component audio stream parameters */
static int kpb_params(struct comp_dev *dev)
{
	struct comp_data *cd = comp_get_drvdata(dev);;
	struct sof_ipc_comp_config *config = COMP_GET_CONFIG(dev);
	struct comp_data *kpb = comp_get_drvdata(dev);
	int err;

	trace_kpb("kpb_params(), config->frame_fmt = %u", config->frame_fmt);

	dev->params.frame_fmt = config->frame_fmt;

	dev->frame_bytes = comp_frame_bytes(dev);

	/* calculate period size based on config */
	cd->period_bytes = dev->frames * dev->frame_bytes;

	trace_kpb("kpb_params(), period_bytes = %d frames %d", cd->period_bytes, dev->frames);

	/* initial kd_dev if needed */
	kpb_init_kd_dev(dev);
	
	if (kpb->sink_type == SOF_COMP_SELECTOR && kpb->kd_dev) {
		/* configure detect pipeline audio params */
//		err = pipeline_params(kpb->kd_dev->pipeline, kpb->kd_dev,
		err = pipeline_params(kpb->kd_dev->pipeline,
				      kpb->kd_dev->pipeline->sched_comp,
				      &dev->params);
		if (err < 0) {
			trace_kpb_error("kpb: pipe %d comp %d params failed %d",
					kpb->kd_dev->pipeline->ipc_pipe.pipeline_id,
					kpb->kd_dev->comp.id, err);
			return err;
		}
	}

	return 0;
}

/**
 * \brief Reclaim memory for a key phrase buffer.
 * \param[in] dev - component device pointer
 *
 * \return none
 */
static void kpb_free(struct comp_dev *dev)
{
	struct comp_data *kd = comp_get_drvdata(dev);

	trace_kpb("kpb_free()");

	rfree(kd);
	rfree(dev);
}

static int kpb_trigger(struct comp_dev *dev, int cmd)
{
	struct comp_data *kpb = comp_get_drvdata(dev);
	int ret = 0;

	trace_kpb("kpb_trigger()");

	ret = comp_set_state(dev, cmd);
	if (ret)
		return ret;

	if (kpb->sink_type == SOF_COMP_SELECTOR && kpb->kd_dev) {
		/* trigger detect pipeline */
		ret = pipeline_trigger(kpb->kd_dev->pipeline,
				       kpb->kd_dev->pipeline->sched_comp,
				       cmd);
		if (ret < 0) {
			trace_kpb_error("kpb: pipe %d comp %d prepare failed %d",
					kpb->kd_dev->pipeline->ipc_pipe.pipeline_id,
					kpb->kd_dev->comp.id, ret);
			return ret;
		}
	}

	return ret;

}

/**
 * \brief Prepare key phrase buffer.
 * \param[in] dev - kpb component device pointer.
 *
 * \return integer representing either
 *	0 -> success
 *	-EINVAL -> failure.
 */
static int kpb_prepare(struct comp_dev *dev)
{
	//struct comp_data *cd = comp_get_drvdata(dev);
	struct comp_data *kpb = comp_get_drvdata(dev);
	int ret = 0;
	//int i;

	trace_kpb("kpb_prepare()");

	ret = comp_set_state(dev, COMP_TRIGGER_PREPARE);
	if (ret)
		return ret;

	if (kpb->sink_type == SOF_COMP_SELECTOR && kpb->kd_dev) {
		/* prepare detect pipeline */
		ret = pipeline_prepare(kpb->kd_dev->pipeline,
				       kpb->kd_dev->pipeline->sched_comp);
		if (ret < 0) {
			trace_kpb_error("kpb: pipe %d comp %d prepare failed %d",
					kpb->kd_dev->pipeline->ipc_pipe.pipeline_id,
					kpb->kd_dev->comp.id, ret);
			return ret;
		}
	}


	return 0;

#if 0

	cd->no_of_clients = 0;

	/* allocate history_buffer/s */
#if KPB_NO_OF_HISTORY_BUFFERS == 2

	cd->his_buf_hp.sta_addr = rballoc(RZONE_RUNTIME, SOF_MEM_CAPS_RAM,
					  KPB_MAX_BUFFER_SIZE - LPSRAM_SIZE);
	cd->his_buf_lp.sta_addr = rballoc(RZONE_RUNTIME, SOF_MEM_CAPS_LP,
					  LPSRAM_SIZE);

	if (!cd->his_buf_hp.sta_addr || !cd->his_buf_lp.sta_addr) {
		trace_kpb_error("Failed to allocate space for "
				"KPB bufefrs");
		return -ENOMEM;
	}

	cd->his_buf_hp.end_addr = cd->his_buf_hp.sta_addr +
	(KPB_MAX_BUFFER_SIZE - LPSRAM_SIZE);

	cd->his_buf_lp.end_addr = cd->his_buf_lp.sta_addr +
	LPSRAM_SIZE;

#elif KPB_NO_OF_HISTORY_BUFFERS == 1

	cd->his_buf_hp.sta_addr = rballoc(RZONE_RUNTIME, SOF_MEM_CAPS_RAM,
					  KPB_MAX_BUFFER_SIZE);

	if (!cd->his_buf_hp.sta_addr) {
		trace_kpb_error("Failed to allocate space for "
				"KPB bufefrs");
	return -ENOMEM;
	}

	cd->his_buf_hp.end_addr = cd->his_buf_hp.sta_addr + KPB_MAX_BUFFER_SIZE;

#else
#error "Wrong number of key phrase buffers configured"
#endif

	/* TODO: zeroes both buffers */

	/* Initialize clients data */
	for (i = 0; i < KPB_MAX_NO_OF_CLIENTS; i++)
		cd->clients[i].state = KPB_CLIENT_UNREGISTERED;

	return ret;
#endif
}

/**
 * \brief Copy real time input stream into sink buffer,
 *	and in the same time buffers that input for
 *	later use by some of clients.
 *
 *\param[in] dev - kpb component device pointer.
 *
 * \return integer representing either
 *	0 - success
 *	-EINVAL - failure.
 */
static int kpb_copy(struct comp_dev *dev)
{
	int ret = 0;
	int update_buffers = 0;
	struct comp_data *kpb = comp_get_drvdata(dev);
	struct comp_buffer *source;
	struct comp_buffer *sink;
	struct list_item *clist;

	tracev_kpb("kpb_copy()");

	/* KPB only ever has one source buffer */
	source = list_first_item(&dev->bsource_list, struct comp_buffer,
				 sink_list);

	/* get the host sink buffer */
        list_for_item(clist, &dev->bsink_list) {
		struct comp_buffer *buffer;

                buffer = container_of(clist, struct comp_buffer, source_list);

		if (buffer->sink->comp.type == kpb->sink_type) {
			sink = buffer;
			goto found;
		}
	}

	trace_kpb_error("kpb_copy() error: cannot find host sink buffer");
	return -EINVAL;

found:
	/* process source data */
	/* check if there are valid pointers */
	if (source && sink) {
		if (!source->r_ptr || !sink->w_ptr)
			return -EINVAL;

		if (sink->free < kpb->period_bytes) {
			trace_kpb("sink free bytes %d", sink->free);
			trace_kpb_error("kpb_copy() error: "
				   "sink component buffer"
				   " has not enough free bytes for copy");
			comp_overrun(dev, sink, kpb->period_bytes, 0);
			return -EIO; /* xrun */
		} 

		if (source->avail < kpb->period_bytes) {
			trace_kpb_error("kpb_copy() error: "
					   "source component buffer"
					   " has not enough data available");
			comp_underrun(dev, source, kpb->period_bytes,
				      0);
			return -EIO; /* xrun */
		}

		/* sink and source are both ready and have space */
		/* TODO: copy sink or source period data here? */
		memcpy(sink->w_ptr, source->r_ptr,
		       kpb->period_bytes);
		/* signal update source & sink data */
		update_buffers = 1;

#if 0
		/* buffer source data internally for future use by clients */
		if (source->avail <= KPB_MAX_BUFFER_SIZE) {
			/* TODO: should we copy what is available or just
			 * a small portion of it?
			 */
			kpb_buffer_data(kpb, source);
		}
#endif
	} else {
		ret = -EIO;
	}

	if (update_buffers) {
		comp_update_buffer_produce(sink, kpb->period_bytes);
		comp_update_buffer_consume(source, kpb->period_bytes);
	}
	return ret;
}

#if 0
/**
 * \brief Buffer real time data stream in
 *	the internal buffer.
 *
 * \param[in] kpb - KPB component data pointer.
 * \param[in] source pointer to the buffer source.
 *
 * \return none
 */
static void kpb_buffer_data(struct comp_data *kpb, struct comp_buffer *source)
{
	trace_kpb("kpb_buffer_data()");
	int size_to_copy = kpb->period_bytes;
	int space_avail;
	struct history_buffer *hb;

#if KPB_NO_OF_HISTORY_BUFFERS == 2

	while (size_to_copy) {
		hb = (kpb->his_buf_lp.state == KPB_BUFFER_FREE) ?
		&kpb->his_buf_lp : &kpb->his_buf_hp;
		space_avail = (int)hb->end_addr - (int)hb->w_ptr;

		if (size_to_copy > space_avail) {
			memcpy(hb->w_ptr, source->r_ptr, space_avail);
			size_to_copy = size_to_copy - space_avail;
			hb->w_ptr = hb->sta_addr;
			hb->state = KPB_BUFFER_FULL;

			if (hb->id == KPB_LP)
				kpb->his_buf_hp.state = KPB_BUFFER_FREE;
			else
				kpb->his_buf_lp.state = KPB_BUFFER_FREE;
		} else  {
			memcpy(hb->w_ptr, source->r_ptr, size_to_copy);
			hb->w_ptr += size_to_copy;
			size_to_copy = 0;
		}
	}
#elif KPB_NO_OF_HISTORY_BUFFERS == 1
	hb = &kpb->his_buf_hp;
	space_avail = (int)hb->end_addr - (int)hb->w_ptr;

	if (size_to_copy > space_avail) {
		/* We need to split copying into two parts
		 * and wrap buffer pointer
		 */
		memcpy(hb->w_ptr, source->r_ptr, space_avail);
		size_to_copy = size_to_copy - space_avail;
		hb->w_ptr = hb->sta_addr;
		memcpy(hb->w_ptr, source->r_ptr, size_to_copy);
		hb->w_ptr += size_to_copy;
		size_to_copy = 0;

	} else {
		memcpy(kpb->w_ptr, source->data_ptr, size_to_copy);
		kpb->w_ptr += size_to_copy;
		size_to_copy = 0;
	}
#else
#error "Wrong number of key phrase buffers configured"
#endif

}
#endif

/**
 * \brief Kpb event handler.
 * \param[in] dev - kpb device component pointer.
 * \param[in] cmd - command to be executed
 * \param[in] data pointer to data related with request
 * \param[in] max_data_size - maxiumum size of data
 *
 *
 * \return integer representing either
 *	0 - success
 *	-EINVAL - failure.
 */
static int kpb_cmd(struct comp_dev *dev, int cmd, void *data,
		   int max_data_size)
{
	//struct sof_ipc_ctrl_data *cdata = data;
	struct comp_data *kpb = comp_get_drvdata(dev);
#if 0
	struct kpb_client *cli = (struct kpb_client *)cdata->data->data;
	uint8_t client_id = cli->id;
#endif
	struct kpb_client *cli = NULL;
	uint8_t client_id = 0;
	int ret = 0;
	uint32_t *cmd_data;

	trace_kpb("kpb_cmd()");


	switch (cmd) {
	case COMP_CMD_SET_VALUE:
		trace_kpb("kpb set value command");
		cmd_data = (uint32_t *)data;
		if (*cmd_data)
			kpb->sink_type = SOF_COMP_SELECTOR;
		else
			kpb->sink_type = SOF_COMP_HOST;
		break;
	case KPB_EVENT_REGISTER_CLIENT:
		ret = kpb_register_client(dev, cli);
		break;
	case KPB_EVENT_UNREGISTER_CLIENT:
		/*TODO*/
		ret = -EINVAL;
		break;
	case KPB_EVENT_BEGIN_DRAINING:
		ret = kpb_begin_draining(dev, client_id);
		break;
	case KPB_EVENT_STOP_DRAINING:
		/*TODO*/
		ret = -EINVAL;
		break;
	default:
		trace_kpb_error("kpb_cmd() error: "
				"unsupported command");
		ret = -EINVAL;
		break;
	}

	return 0;
}

/**
 * \brief Drain internal buffer into client's
 *	sink buffer.
 *
 * \param[in] dev - kpb device component pointer.
 * \param[in] client_id - clients id
 *
 * \return integer representing either
 *	0 - success
 *	-EINVAL - failure.
 */
static int kpb_begin_draining(struct comp_dev *dev, uint8_t client_id)
{
	int ret = 0;
	struct comp_data *kpb = comp_get_drvdata(dev);
	struct comp_buffer *sink;
	int i = client_id;
	int history_depth = kpb->clients[client_id].history_depth;
	struct history_buffer *hb;
	int buffered_so_far;
	void *source_r_ptr;
	int copy_residue;
	void *sink_w_ptr;

	trace_kpb("kpb_begin_draining()");

	struct list_item *sink_list = dev->bsink_list.next;

	do {
		sink = list_item(sink_list->next, struct comp_buffer,
				 sink_list);
	} while (--i);

	if (!sink) {
		trace_kpb_error("kpb_begin_draining() error: "
				"requested draining for unregistered client");
		return -EINVAL;
	}

#if KPB_NO_OF_HISTORY_BUFFERS == 2
	/* choose proper buffer to read from */
	hb = (kpb->his_buf_lp.state == KPB_BUFFER_FREE &&
	     (kpb->his_buf_lp.w_ptr != kpb->his_buf_lp.sta_addr)) ?
	     &kpb->his_buf_lp : &kpb->his_buf_hp;

	if (hb->sta_addr == hb->w_ptr)
		source_r_ptr = hb->end_addr;
	else
		source_r_ptr = hb->w_ptr;

	buffered_so_far = (int)source_r_ptr - (int)hb->sta_addr;

	if (buffered_so_far < history_depth) {
		source_r_ptr = source_r_ptr - buffered_so_far;
		memcpy(sink->w_ptr, source_r_ptr, buffered_so_far);
		sink_w_ptr = sink->w_ptr + buffered_so_far;
		copy_residue = history_depth - buffered_so_far;

		/* change buffer */
		if (hb->id == KPB_LP)
			hb = &kpb->his_buf_hp;
		else
			hb = &kpb->his_buf_lp;

		source_r_ptr = hb->end_addr - copy_residue;
		memcpy(sink_w_ptr, source_r_ptr, copy_residue);

	} else {
		source_r_ptr = source_r_ptr - history_depth;
		memcpy(sink->w_ptr, source_r_ptr, history_depth);
	}

	/* update sink data */
	comp_update_buffer_produce(sink, history_depth);

#elif KPB_NO_OF_HISTORY_BUFFERS == 1

	hb = &kpb->his_buf_hp;

	if (hb->sta_addr == hb->w_ptr)
		source_r_ptr = hb->end_addr;
	else
		source_r_ptr = hb->w_ptr;

	buffered_so_far = (int)source_r_ptr - (int)hb->sta_addr;

	if (buffered_so_far < history_depth) {
		sink_w_ptr = sink->w_ptr;
		source_r_ptr = source_r_ptr - buffered_so_far;
		memcpy(sink_w_ptr, source_r_ptr, buffered_so_far);
		sink_w_ptr = sink_w_ptr + buffered_so_far;
		source_r_ptr = hb->end_addr;
		copy_residue = history_depth - buffered_so_far;
		memcpy(sink_w_ptr, source_r_ptr, history_depth);
	} else {
		sink_w_ptr = sink->w_ptr;
		memcpy(sink_w_ptr, source_r_ptr, history_depth);
	}

	comp_update_buffer_produce(sink, history_depth);

#else
#error "Wrong number of key phrase buffers configured"
#endif

	return ret;
}

/**
 * \brief Register clients in the system.
 *
 * \param[in] dev - kpb device component pointer.
 * \param[in] cli - pointer to KPB client's data.
 *
 * \return integer representing either
 *	0 - success
 *	-EINVAL - failure.
 */
static int kpb_register_client(struct comp_dev *dev, struct kpb_client *cli)
{
	int ret;
	struct comp_data *kpb = comp_get_drvdata(dev);

	trace_kpb("kpb_register_client()");

	if (!cli) {
		trace_kpb_error("kpb_register_client() error: "
				"no client data");
		return -EINVAL;
	}
	/* Do we have a room for a new client? */
	if (kpb->no_of_clients >= KPB_MAX_NO_OF_CLIENTS ||
	    cli->id > KPB_MAX_NO_OF_CLIENTS) {
		trace_kpb_error("kpb_register_client() error: "
				"no free room for client = %u ",
				cli->id);
		ret = -EINVAL;
	} else if (kpb->clients[cli->id].state != KPB_CLIENT_UNREGISTERED) {
		trace_kpb_error("kpb_register_client() error: "
				"client = %u already registered",
				cli->id);
		ret = -EINVAL;
	} else {
		/* Client accepted, let's store his data */
		kpb->clients[cli->id].history_depth = cli->history_depth;
		kpb->clients[cli->id].id  = cli->id;
		kpb->clients[cli->id].state = KPB_CLIENT_BUFFERING;
		kpb->no_of_clients++;
		ret = 0;
	}

	return ret;
}

static void kpb_cache(struct comp_dev *dev, int cmd)
{
	/* TODO: writeback history buffer */
}

static int kpb_reset(struct comp_dev *dev)
{
	struct comp_data *kpb = comp_get_drvdata(dev);
	int ret = 0;

	/* TODO: what data of KPB should we reset here? */

	if (kpb->sink_type == SOF_COMP_SELECTOR && kpb->kd_dev) {
		/* reset detect pipeline */
		ret = pipeline_reset(kpb->kd_dev->pipeline,
				     kpb->kd_dev->pipeline->sched_comp);
		if (ret < 0) {
			trace_kpb_error("kpb: pipe %d comp %d reset failed %d",
					kpb->kd_dev->pipeline->ipc_pipe.pipeline_id,
					kpb->kd_dev->comp.id, ret);
			return ret;
		}
	}

	return ret;
}

struct comp_driver comp_kpb = {
	.type = SOF_COMP_KPB,
	.ops = {
		.new = kpb_new,
		.free = kpb_free,
		.cmd = kpb_cmd,
		.params	= kpb_params,
		.trigger = kpb_trigger,
		.copy = kpb_copy,
		.prepare = kpb_prepare,
		.reset = kpb_reset,
		.cache = kpb_cache,
	},
};

void sys_comp_kpb_init(void)
{
	comp_register(&comp_kpb);
}
