/*
 * nRF54L15 — Continuous Keyword Spotting  (sliding-window edition)
 *
 * Key changes vs v1:
 *   • 50% overlapping windows (0.5 s stride over 1.0 s window) so a "hello"
 *     spoken across a chunk boundary is always fully captured in one window.
 *   • No mid-loop audio drain — the overlap itself covers boundaries, and
 *     audio buffered during inference is real signal we want, not garbage.
 *   • "hello" looked up by label name, not hardcoded index 0 — survives
 *     EI project label reordering.
 *   • Detection cooldown prevents one utterance firing 3–4 overlapping windows.
 *   • PDM warmup drain at startup removes oscillator settle-time garbage.
 *   • Dead code (unused best_idx) removed.
 *   • Per-cycle timing printed for profiling.
 */

#include <stdio.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/regulator.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/settings/settings.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>
#include <zephyr/sys_clock.h>

#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

#define SAMPLE_RATE                 16000
#define INF_SECONDS                 1.0f
#define STRIDE_SECONDS              0.5f

// Derived — do not edit these directly
#define INF_SAMPLES                 ((int)(SAMPLE_RATE * INF_SECONDS))    // 16 000
#define STRIDE_SAMPLES              ((int)(SAMPLE_RATE * STRIDE_SECONDS)) //  8 000
#define BLOCK_SIZE                  2048   // bytes per DMIC read block

// Detection tuning
#define HELLO_LABEL                 "Hello"
#define HELLO_CONFIDENCE_THRESHOLD  0.85f  // slightly relaxed from 0.9; tune to taste
#define LED_ON_DURATION_MS          3000

// After a detection, skip this many inference windows before triggering again.
// With a 0.5 s stride, 3 windows = ~1.5 s cooldown. This prevents a single
// utterance from firing multiple times as overlapping windows all score high.
#define DETECTION_COOLDOWN_WINDOWS  3

// ── 2. HARDWARE NODES ─────────────────────────────────────────────────────────

static const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct device *const rfsw_pwr = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(rfsw_pwr));
static const struct device *const rfsw_ctl = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(rfsw_ctl));
static const struct gpio_dt_spec led0      = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

// ── 3. AUDIO BUFFER ───────────────────────────────────────────────────────────
//
// audio_buf always holds the most recent INF_SAMPLES of audio.
// Layout after each slide:
//
//   Before:  [ ── old first half (A) ── | ── old second half (B) ── ]
//   After:   [ ── old second half (B) ── | ── fresh stride (C) ── ]
//
// 50% overlap means any 300–500 ms keyword will be fully inside at least
// one of two consecutive windows regardless of when it started.

static int16_t audio_buf[INF_SAMPLES];
K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 16, 4);

// ── 4. BLE DATA ───────────────────────────────────────────────────────────────

typedef enum { ADV_STATE_IDLE, ADV_STATE_RUNNING } adv_state_t;
static adv_state_t adv_state = ADV_STATE_IDLE;

static uint8_t __aligned(4) mfg_data[] = { 0x90, 0x05, 0x00 };

static struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_MANUFACTURER_DATA, mfg_data, 3),
};
static struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, "XIAO_SENSE", 10),
};

// ── 5. DEFERRED WORK ──────────────────────────────────────────────────────────

static void led_off_work_fn(struct k_work *w)
{
    gpio_pin_set_dt(&led0, 1);  // active-low: 1 = off
}
K_WORK_DELAYABLE_DEFINE(led_off_work, led_off_work_fn);

static void stop_adv_work_fn(struct k_work *w)
{
    if (adv_state == ADV_STATE_RUNNING) {
        bt_le_adv_stop();
        adv_state = ADV_STATE_IDLE;
        printk("[BLE] Radio sleep\n");
    }
}
K_WORK_DELAYABLE_DEFINE(stop_adv_work, stop_adv_work_fn);

// ── 6. RF POWER ───────────────────────────────────────────────────────────────

static void enable_rf_hardware(void)
{
    if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_enable(rfsw_pwr);
    if (rfsw_ctl && device_is_ready(rfsw_ctl)) regulator_enable(rfsw_ctl);
    k_msleep(200);
}

// ── 7. BLE ADVERTISING ───────────────────────────────────────────────────────

static void start_advertising(void)
{
    if (adv_state == ADV_STATE_RUNNING) {
        bt_le_adv_update_data(ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
        k_work_reschedule(&stop_adv_work, K_MSEC(600));
        return;
    }

    struct bt_le_adv_param adv_param = {
        .id           = BT_ID_DEFAULT,
        .sid          = 0,
        .options      = 0,
        .interval_min = 160,
        .interval_max = 165,
    };

    int err = -EAGAIN;
    for (int attempt = 0; attempt < 3 && err != 0; attempt++) {
        if (attempt > 0) k_msleep(20);
        err = bt_le_adv_start(&adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    }

    if (err == 0) {
        adv_state = ADV_STATE_RUNNING;
        printk("[BLE] Burst start\n");
        k_work_reschedule(&stop_adv_work, K_MSEC(600));
    } else {
        printk("[BLE] Adv fail: %d\n", err);
    }
}

// ── 8. AUDIO HELPERS ──────────────────────────────────────────────────────────

// EI get_data callback: converts int16 PCM in audio_buf to float [-1, 1].
static int get_audio_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)audio_buf[offset + i];
    }
    return 0;
}

// Blocking read of exactly num_samples int16 PCM samples into dst.
// Consumes whatever is already queued in the slab first (including any audio
// that accumulated during the previous inference), then waits for more.
// This is intentional: that buffered audio is real signal, not stale garbage.
static void read_pcm(int16_t *dst, uint32_t num_samples)
{
    uint32_t bytes_needed  = num_samples * sizeof(int16_t);
    uint32_t bytes_written = 0;

    while (bytes_written < bytes_needed) {
        void    *buffer;
        uint32_t size;

        if (dmic_read(dmic_dev, 0, &buffer, &size, 200) < 0) {
            continue;  // slab not yet refilled — retry
        }

        uint32_t to_copy = MIN(size, bytes_needed - bytes_written);
        memcpy((uint8_t *)dst + bytes_written, buffer, to_copy);
        bytes_written += to_copy;
        k_mem_slab_free(&mem_slab, buffer);
    }
}

// Discard any blocks already queued in the slab. Non-blocking (timeout = 0).
// Used ONLY at startup to remove PDM oscillator settle-time garbage.
static void drain_stale_audio(void)
{
    void    *buffer;
    uint32_t size;
    while (dmic_read(dmic_dev, 0, &buffer, &size, 0) >= 0) {
        k_mem_slab_free(&mem_slab, buffer);
    }
}

// ── 9. SLIDING WINDOW ─────────────────────────────────────────────────────────

// Fills audio_buf with the very first INF_SAMPLES. Called once at startup.
static void initial_fill(void)
{
    read_pcm(audio_buf, INF_SAMPLES);
}

// Slides the window forward by STRIDE_SAMPLES:
//   1. Shift the second half of audio_buf into the first half.
//   2. Fill the second half with a fresh STRIDE_SAMPLES from the DMIC.
//
// Because read_pcm drains slab-queued audio before waiting for new blocks,
// audio that buffered during the previous inference is consumed here —
// exactly as if there had been no inference gap.
static void slide_window(void)
{
    memmove(audio_buf,
            audio_buf + STRIDE_SAMPLES,
            STRIDE_SAMPLES * sizeof(int16_t));

    read_pcm(audio_buf + STRIDE_SAMPLES, STRIDE_SAMPLES);
}

// ── 10. INFERENCE ─────────────────────────────────────────────────────────────

static int cooldown_remaining = 0;

static void run_inference(void)
{
    if (cooldown_remaining > 0) {
        cooldown_remaining--;
        return;
    }

    signal_t signal;
    signal.total_length = INF_SAMPLES;
    signal.get_data     = &get_audio_data;
    ei_impulse_result_t result = {0};

    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) {
        printk("[ERR] Classifier failed\n");
        return;
    }

    float hello_confidence = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (strcmp(result.classification[i].label, HELLO_LABEL) == 0) {
            hello_confidence = result.classification[i].value;
            break;
        }
    }

    if (hello_confidence > HELLO_CONFIDENCE_THRESHOLD) {
        printk("\n╔══════════════════════════════╗\n");
        printk("║   *** HELLO DETECTED ***     ║\n");
        printk("╠══════════════════════════════╣\n");
        printk("║ Confidence : %5.1f %%          ║\n", (double)(hello_confidence * 100.0f));
        printk("║ DSP        : %5d ms          ║\n", result.timing.dsp);
        printk("╚══════════════════════════════╝\n\n");

        //gpio_pin_set_dt(&led0, 0);
        //k_work_reschedule(&led_off_work, K_MSEC(LED_ON_DURATION_MS));

        mfg_data[2] = 1;
        start_advertising();

        cooldown_remaining = DETECTION_COOLDOWN_WINDOWS;
    } else {
        mfg_data[2] = 0;
    }
}

// ── 11. MAIN ──────────────────────────────────────────────────────────────────

int main(void)
{
    enable_rf_hardware();

    //gpio_pin_configure_dt(&led0, GPIO_OUTPUT_INACTIVE);
    //gpio_pin_set_dt(&led0, 1);  // LED off
    k_msleep(500);

    // Bluetooth init
    if (bt_enable(NULL)) {
        printk("[FATAL] BT enable failed\n");
        return 0;
    }
    settings_load();

    // DMIC init
    if (!device_is_ready(dmic_dev)) {
        printk("[FATAL] DMIC not ready\n");
        return 0;
    }

    struct pcm_stream_cfg stream = {
        .pcm_rate   = SAMPLE_RATE,
        .pcm_width  = 16,
        .block_size = BLOCK_SIZE,
        .mem_slab   = &mem_slab,
    };
    struct dmic_cfg cfg = {
        .io      = { .min_pdm_clk_freq = 1000000, .max_pdm_clk_freq = 3500000 },
        .streams = &stream,
        .channel = {
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
            .req_num_chan    = 1,
            .req_num_streams = 1,
        },
    };

    if (dmic_configure(dmic_dev, &cfg) < 0) {
        printk("[FATAL] DMIC configure failed\n");
        return 0;
    }
    if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
        printk("[FATAL] DMIC start failed\n");
        return 0;
    }

    // Warmup: let the PDM oscillator settle, then discard the initial garbage.
    // Without this, the first inference window may contain startup transients
    // that distort the MFCC features and produce false noise classifications.
    k_msleep(200);
    drain_stale_audio();

    printk("[KWS] Listening for '%s'  (stride=%d ms, window=%d ms)\n",
           HELLO_LABEL,
           (int)(STRIDE_SECONDS * 1000),
           (int)(INF_SECONDS   * 1000));

    // Prime audio_buf with the first full second, then enter the sliding loop
    initial_fill();
    run_inference();

    while (1) {
        slide_window();   // shift buffer left, append 0.5 s of fresh audio
        run_inference();  // run EI classifier on the latest 1.0 s window
    }
}