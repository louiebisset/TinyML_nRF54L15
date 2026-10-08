#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/settings/settings.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/regulator.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/pm/device_runtime.h>
#include <string.h>

#ifdef ROUND_UP
#undef ROUND_UP
#endif
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// --- 1. CONFIG ---
#define SAMPLE_RATE     16000
#define TOTAL_SAMPLES   ((int)(16000 * 1.5f))
#define DISCARD_SAMPLES ((int)(16000 * 0.5f))
#define INF_SAMPLES     ((int)(16000 * 1.0f))
#define BLOCK_SIZE      2048

#define BLE_ADV_INTERVAL_UNITS 160
#define BLE_ADV_DURATION_MS    600

// --- 2. HARDWARE NODES ---
static const struct device *const dmic_dev    = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct gpio_dt_spec button       = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static const struct device *const rfsw_pwr    = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(rfsw_pwr));
static const struct device *const pdm_imu_pwr = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(pdm_imu_pwr));

// --- 3. GLOBALS ---
static int16_t rx_buffer[TOTAL_SAMPLES];
K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 8, 4);
static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// --- 4. BLE DATA (all in RAM - SoftDevice DMA cannot access flash) ---
static uint8_t adv_flags  = BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR;
static uint8_t mfg_data[] = { 0x90, 0x05, 0x00 };
static char    dev_name[] = "XIAO_SENSE";

static struct bt_data ad[] = {
    BT_DATA(BT_DATA_FLAGS, &adv_flags, 1),
    BT_DATA(BT_DATA_MANUFACTURER_DATA, mfg_data, 3),
};
static struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, dev_name, sizeof(dev_name) - 1),
};

// --- 5. DEFERRED WORK ---
static void stop_adv_work_fn(struct k_work *w) {
    bt_le_adv_stop();
    bt_disable();

    if (rfsw_pwr && device_is_ready(rfsw_pwr)) {
        regulator_disable(rfsw_pwr);
    }
}
K_WORK_DELAYABLE_DEFINE(stop_adv_work, stop_adv_work_fn);

// --- 6. BLE BROADCAST ---
static void broadcast_result(uint8_t result_index) {

    if (rfsw_pwr && device_is_ready(rfsw_pwr)) {
        regulator_enable(rfsw_pwr);
        k_msleep(10);
    }

    if (bt_enable(NULL) != 0) return;

    // Required on every bt_enable() - CONFIG_SETTINGS_RUNTIME=y
    // means this works without UART/console
    settings_load();

    k_msleep(50);

    mfg_data[2] = result_index;

    struct bt_le_adv_param adv_param = {
        .id           = BT_ID_DEFAULT,
        .sid          = 0,
        .options      = 0,
        .interval_min = BLE_ADV_INTERVAL_UNITS,
        .interval_max = BLE_ADV_INTERVAL_UNITS,
    };

    int err = -EAGAIN;
    for (int attempt = 0; attempt < 3 && err != 0; attempt++) {
        if (attempt > 0) k_msleep(20);
        err = bt_le_adv_start(&adv_param, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    }

    if (err == 0) {
        k_work_reschedule(&stop_adv_work, K_MSEC(BLE_ADV_DURATION_MS));
    } else {
        bt_disable();
        if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_disable(rfsw_pwr);
    }
}

// --- 7. BUTTON ISR ---
static void button_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
    k_sem_give(&button_sem);
}

// --- 8. AUDIO CALLBACK ---
static int get_audio_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)rx_buffer[offset + i + DISCARD_SAMPLES] / 32768.0f;
    }
    return 0;
}

// --- 9. RECORDING ---
static void record_audio(void) {
    uint32_t bytes_recorded = 0;
    struct pcm_stream_cfg stream = {
        .pcm_rate = SAMPLE_RATE, .pcm_width = 16,
        .block_size = BLOCK_SIZE, .mem_slab = &mem_slab
    };
    struct dmic_cfg cfg = {
        .io = { .min_pdm_clk_freq = 1000000, .max_pdm_clk_freq = 3500000 },
        .streams = &stream,
        .channel = {
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
            .req_num_chan = 1, .req_num_streams = 1
        }
    };

    if (dmic_configure(dmic_dev, &cfg) < 0) return;

    dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
    while (bytes_recorded < sizeof(rx_buffer)) {
        void *ptr; uint32_t size;
        if (dmic_read(dmic_dev, 0, &ptr, &size, 500) >= 0) {
            memcpy((uint8_t *)rx_buffer + bytes_recorded, ptr, size);
            bytes_recorded += size;
            k_mem_slab_free(&mem_slab, ptr);
        } else {
            break;
        }
    }
    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
}

// --- 10. INFERENCE ---
static uint8_t run_inference(void) {
    signal_t signal;
    signal.get_data     = &get_audio_data;
    signal.total_length = (size_t)INF_SAMPLES;

    ei_impulse_result_t result = {0};

    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) return 0;

    int best_idx   = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    return (uint8_t)(best_idx + 1);
}

// --- 11. MAIN ---
int main(void) {
    k_msleep(1000);

    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    pm_device_runtime_enable(dmic_dev);
    pm_device_runtime_put(dmic_dev);

    while (1) {
        k_sem_take(&button_sem, K_FOREVER);

        if (pdm_imu_pwr && device_is_ready(pdm_imu_pwr)) {
            regulator_enable(pdm_imu_pwr);
            k_msleep(50);
        }

        pm_device_runtime_get(dmic_dev);
        k_msleep(10);
        record_audio();
        pm_device_runtime_put(dmic_dev);

        if (pdm_imu_pwr && device_is_ready(pdm_imu_pwr)) {
            regulator_disable(pdm_imu_pwr);
        }

        broadcast_result(run_inference());
    }
}