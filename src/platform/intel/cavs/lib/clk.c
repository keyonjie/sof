// SPDX-License-Identifier: BSD-3-Clause
//
// Copyright(c) 2019 Intel Corporation. All rights reserved.
//
// Author: Janusz Jankowski <janusz.jankowski@linux.intel.com>

#include <sof/common.h>
#include <sof/drivers/ssp.h>
#include <sof/lib/clk.h>
#include <sof/lib/uuid.h>
#include <sof/lib/memory.h>
#include <sof/lib/notifier.h>
#include <sof/lib/pm_runtime.h>
#include <sof/sof.h>
#include <sof/spinlock.h>

/* 77de2074-828c-4044-a40b-420b72749e8b */
DECLARE_SOF_UUID("clk", clk_uuid, 0x77de2075, 0x828c, 0x4044,
		 0xa4, 0x0b, 0x42, 0x0b, 0x72, 0x74, 0x9e, 0x8b);

DECLARE_TR_CTX(clk_tr, SOF_UUID(clk_uuid), LOG_LEVEL_INFO);

static SHARED_DATA struct clock_info platform_clocks_info[NUM_CLOCKS];

#if CAVS_VERSION == CAVS_VERSION_1_5
static inline void select_cpu_clock(int freq_idx, bool release_unused)
{
	uint32_t enc = cpu_freq_enc[freq_idx];

	io_reg_update_bits(SHIM_BASE + SHIM_CLKCTL, SHIM_CLKCTL_HDCS, 0);
	io_reg_update_bits(SHIM_BASE + SHIM_CLKCTL,
			   SHIM_CLKCTL_DPCS_MASK(cpu_get_id()),
			   enc);
}
#else
static inline void select_cpu_clock(int freq_idx, bool release_unused)
{
	uint32_t enc = cpu_freq_enc[freq_idx];
	uint32_t status_mask = cpu_freq_status_mask[freq_idx];

#if CONFIG_TIGERLAKE
	if (freq_idx == CPU_HPRO_FREQ_IDX)
		pm_runtime_get(PM_RUNTIME_DSP, PWRD_BY_HPRO | (PLATFORM_CORE_COUNT - 1));
#endif

	/* request clock */
	io_reg_write(SHIM_BASE + SHIM_CLKCTL,
		     io_reg_read(SHIM_BASE + SHIM_CLKCTL) | enc);

	/* wait for requested clock to be on */
	while ((io_reg_read(SHIM_BASE + SHIM_CLKSTS) &
		status_mask) != status_mask)
		idelay(PLATFORM_DEFAULT_DELAY);

	/* switch to requested clock */
	io_reg_update_bits(SHIM_BASE + SHIM_CLKCTL,
			   SHIM_CLKCTL_OSC_SOURCE_MASK, enc);

	if (release_unused) {
		/* release other clocks */
		io_reg_write(SHIM_BASE + SHIM_CLKCTL,
			     (io_reg_read(SHIM_BASE + SHIM_CLKCTL) &
			      ~SHIM_CLKCTL_OSC_REQUEST_MASK) | enc);
#if CONFIG_TIGERLAKE
		if (freq_idx != CPU_HPRO_FREQ_IDX)
			pm_runtime_put(PM_RUNTIME_DSP, PWRD_BY_HPRO | (PLATFORM_CORE_COUNT - 1));
#endif
	}
}
#endif

static int clock_platform_set_cpu_freq(int clock, int freq_idx)
{
	select_cpu_clock(freq_idx, true);
	return 0;
}

/* Clock source to be used when not waiting for an interrupt. */
static SHARED_DATA int active_freq_idx = CPU_DEFAULT_IDX;

static inline int get_cpu_current_freq_idx(void)
{
	struct clock_info *clk_info = clocks_get() + CLK_CPU(cpu_get_id());

	return clk_info->current_freq_idx;
}

static inline void set_cpu_current_freq_idx(int freq_idx)
{
	int i;
	struct clock_info *clk_info = clocks_get();

	for (i = 0; i < PLATFORM_CORE_COUNT; i++)
		clk_info[i].current_freq_idx = freq_idx;

	platform_shared_commit(clk_info,
			       sizeof(*clk_info) * PLATFORM_CORE_COUNT);
}

#if CONFIG_CAVS_USE_LPRO_IN_WAITI
void platform_clock_on_wakeup(void)
{
#if 0
	int freq_idx = *cache_to_uncache(&active_freq_idx);

	if (freq_idx != get_cpu_current_freq_idx()) {
		select_cpu_clock(freq_idx, true);
		set_cpu_current_freq_idx(freq_idx);
	}
#endif
}
#endif

int platform_get_active_freq(void)
{
	return *cache_to_uncache(&active_freq_idx);
}

void platform_clock_waiti_entry(void)
{
	int freq_idx;
	int target_idx;

	/* update the active clock according to the pm runtime state */
	if (*cache_to_uncache(&active_freq_idx) != CPU_LPRO_FREQ_IDX &&
	    !pm_runtime_is_active(PM_RUNTIME_DSP, PLATFORM_MASTER_CORE_ID))
		platform_set_active_clock(CPU_LPRO_FREQ_IDX);
	else if (*cache_to_uncache(&active_freq_idx) != CPU_HPRO_FREQ_IDX &&
		 pm_runtime_is_active(PM_RUNTIME_DSP, PLATFORM_MASTER_CORE_ID))
		platform_set_active_clock(CPU_HPRO_FREQ_IDX);

	freq_idx = get_cpu_current_freq_idx();

#if CONFIG_CAVS_USE_LPRO_IN_WAITI
	target_idx = CPU_LPRO_FREQ_IDX;

	/* store the cpu freq_idx */
	*cache_to_uncache(&active_freq_idx) = freq_idx;
#else
	target_idx = *cache_to_uncache(&active_freq_idx);
#endif

	if (freq_idx != target_idx) {
		/* LPRO requests are fast, but requests for other ROs
		 * can take a lot of time. That's why it's better to
		 * not release active clock just for waiti,
		 * so they can be switched without delay on wake up.
		 */
		select_cpu_clock(target_idx, false);
		set_cpu_current_freq_idx(target_idx);
	}
}

void platform_clock_waiti_exit(void)
{
	int current_idx = get_cpu_current_freq_idx();
	int target_idx = *cache_to_uncache(&active_freq_idx);

	/* restore the active cpu freq_idx */
	if (current_idx != target_idx) {
		/* LPRO requests are fast, but requests for other ROs
		 * can take a lot of time. That's why it's better to
		 * not release active clock just for waiti,
		 * so they can be switched without delay on wake up.
		 */
		select_cpu_clock(target_idx, false);
		set_cpu_current_freq_idx(target_idx);
	}
}

void platform_set_active_clock(int index)
{
	if (*cache_to_uncache(&active_freq_idx) == index)
		return;

	tr_info(&clk_tr, "platform_set_active_clock start at %u",
		platform_timer_get(timer_get()));

	select_cpu_clock(index, true);
	tr_info(&clk_tr, "platform_set_active_clock cpu clock switched done at %u",
		platform_timer_get(timer_get()));
	set_cpu_current_freq_idx(index);

	tr_info(&clk_tr, "platform_set_active_clock current freq updated at %u",
		platform_timer_get(timer_get()));

	*cache_to_uncache(&active_freq_idx) = index;
}

void platform_clock_init(struct sof *sof)
{
	int i;

	sof->clocks =
		cache_to_uncache((struct clock_info *)platform_clocks_info);

	for (i = 0; i < PLATFORM_CORE_COUNT; i++) {
		sof->clocks[i] = (struct clock_info) {
			.freqs_num = NUM_CPU_FREQ,
			.freqs = cpu_freq,
			.default_freq_idx = CPU_DEFAULT_IDX,
			.current_freq_idx = CPU_DEFAULT_IDX,
			.notification_id = NOTIFIER_ID_CPU_FREQ,
			.notification_mask = NOTIFIER_TARGET_CORE_MASK(i),
			.set_freq = clock_platform_set_cpu_freq,
		};

		spinlock_init(&sof->clocks[i].lock);
	}

	sof->clocks[CLK_SSP] = (struct clock_info) {
		.freqs_num = NUM_SSP_FREQ,
		.freqs = ssp_freq,
		.default_freq_idx = SSP_DEFAULT_IDX,
		.current_freq_idx = SSP_DEFAULT_IDX,
		.notification_id = NOTIFIER_ID_SSP_FREQ,
		.notification_mask = NOTIFIER_TARGET_CORE_ALL_MASK,
		.set_freq = NULL,
	};

	spinlock_init(&sof->clocks[CLK_SSP].lock);

	platform_shared_commit(sof->clocks, sizeof(*sof->clocks) * NUM_CLOCKS);
}
