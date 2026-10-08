#include <stdio.h>
#include <string.h>
#include <math.h>

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/regulator.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/settings/settings.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>

#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

#define SAMPLE_RATE_HZ   50
#define TOTAL_SAMPLES    150
#define SAMPLE_DELAY_MS  (1000 / SAMPLE_RATE_HZ)

#define BLE_ADV_INTERVAL_UNITS 160
#define BLE_ADV_DURATION_MS    600

// ── 2. HARDWARE NODES ─────────────────────────────────────────────────────────

static const struct i2c_dt_spec imu           = I2C_DT_SPEC_GET(DT_NODELABEL(lsm6dso));
static const struct device *const imu_dev     = imu.bus;
static const struct gpio_dt_spec button       = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static const struct device *const rfsw_pwr    = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(rfsw_pwr));
static const struct device *const rfsw_ctl    = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(rfsw_ctl));
static const struct device *const pdm_imu_pwr = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(pdm_imu_pwr));

// ── 3. GLOBALS ────────────────────────────────────────────────────────────────

struct RawSample { int16_t ax, ay, az, gx, gy, gz; };
static struct RawSample rx_buffer[TOTAL_SAMPLES];
static float ei_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// ── 4. BLE DATA (all in RAM - SoftDevice DMA cannot access flash) ─────────────

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

// ── 5. DEFERRED WORK ──────────────────────────────────────────────────────────

static void stop_adv_work_fn(struct k_work *w)
{
    bt_le_adv_stop();
    bt_disable();

    if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_disable(rfsw_pwr);
    if (rfsw_ctl && device_is_ready(rfsw_ctl)) regulator_disable(rfsw_ctl);
}
K_WORK_DELAYABLE_DEFINE(stop_adv_work, stop_adv_work_fn);

// ── 6. BLE BROADCAST ─────────────────────────────────────────────────────────

static void broadcast_result(uint8_t result)
{
    // Power up RF rails
    if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_enable(rfsw_pwr);
    if (rfsw_ctl && device_is_ready(rfsw_ctl)) regulator_enable(rfsw_ctl);
    k_msleep(10);

    // Enable BLE stack
    if (bt_enable(NULL) != 0) {
        if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_disable(rfsw_pwr);
        if (rfsw_ctl && device_is_ready(rfsw_ctl)) regulator_disable(rfsw_ctl);
        return;
    }

    // Required on every bt_enable() - works without UART
    // because CONFIG_SETTINGS_RUNTIME=y is set in prj.conf
    settings_load();
    k_msleep(50);

    mfg_data[2] = result;

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
        // Adv failed, shut down cleanly
        bt_disable();
        if (rfsw_pwr && device_is_ready(rfsw_pwr)) regulator_disable(rfsw_pwr);
        if (rfsw_ctl && device_is_ready(rfsw_ctl)) regulator_disable(rfsw_ctl);
    }
}

// ── 7. BUTTON ISR ─────────────────────────────────────────────────────────────

static void button_isr(const struct device *dev, struct gpio_callback *cb,
                       uint32_t pins)
{
    k_sem_give(&button_sem);
}

// ── 8. IMU POWER ─────────────────────────────────────────────────────────────

static void imu_power_enable(void)
{
    if (pdm_imu_pwr && device_is_ready(pdm_imu_pwr)) {
        regulator_enable(pdm_imu_pwr);
        k_msleep(50);
    }
    pm_device_runtime_get(imu_dev);
    k_msleep(10);
}

static void imu_power_disable(void)
{
    pm_device_runtime_put(imu_dev);
    if (pdm_imu_pwr && device_is_ready(pdm_imu_pwr)) {
        regulator_disable(pdm_imu_pwr);
    }
}

// ── 9. IMU HELPERS ────────────────────────────────────────────────────────────

static void imu_wake_up(void)
{
    i2c_reg_write_byte_dt(&imu, 0x10, 0x60);
    i2c_reg_write_byte_dt(&imu, 0x11, 0x6C);
    k_msleep(20);
}

static void imu_power_down(void)
{
    i2c_reg_write_byte_dt(&imu, 0x10, 0x00);
    i2c_reg_write_byte_dt(&imu, 0x11, 0x00);
}

static int ei_get_data(size_t offset, size_t length, float *out_ptr)
{
    if (out_ptr == NULL) return -1;
    memcpy(out_ptr, ei_buffer + offset, length * sizeof(float));
    return 0;
}

static void record_gesture(void)
{
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        uint8_t raw[6];

        i2c_burst_read_dt(&imu, 0x28, raw, 6);
        rx_buffer[i].ax = (int16_t)(raw[0] | (raw[1] << 8));
        rx_buffer[i].ay = (int16_t)(raw[2] | (raw[3] << 8));
        rx_buffer[i].az = (int16_t)(raw[4] | (raw[5] << 8));

        i2c_burst_read_dt(&imu, 0x22, raw, 6);
        rx_buffer[i].gx = (int16_t)(raw[0] | (raw[1] << 8));
        rx_buffer[i].gy = (int16_t)(raw[2] | (raw[3] << 8));
        rx_buffer[i].gz = (int16_t)(raw[4] | (raw[5] << 8));

        k_msleep(SAMPLE_DELAY_MS);
    }
}

// ── 10. INFERENCE ─────────────────────────────────────────────────────────────

static uint8_t run_inference(void)
{
    size_t f_idx = 0;

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        float ax = (float)rx_buffer[i].ax;
        float ay = (float)rx_buffer[i].ay;
        float az = (float)rx_buffer[i].az;
        float gx = (float)rx_buffer[i].gx;
        float gy = (float)rx_buffer[i].gy;
        float gz = (float)rx_buffer[i].gz;

        float acc_mag  = sqrtf(ax * ax + ay * ay + az * az);
        float gyro_mag = sqrtf(gx * gx + gy * gy + gz * gz);

        float feat[8] = { ax, ay, az, acc_mag, gx, gy, gz, gyro_mag };

        for (int j = 0; j < 8; j++) {
            if (f_idx < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
                ei_buffer[f_idx++] = feat[j];
            }
        }
    }

    signal_t signal;
    memset(&signal, 0, sizeof(signal_t));
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data     = &ei_get_data;

    ei_impulse_result_t result = { 0 };
    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) return 0;

    int   best_idx = 0;
    float best_val = 0.0f;

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    return (uint8_t)(best_idx + 1);
}

// ── 11. MAIN ──────────────────────────────────────────────────────────────────

int main(void)
{
    k_msleep(500);

    // GPIO setup
    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    // IMU runtime PM enable, soft reset, then power down immediately
    pm_device_runtime_enable(imu_dev);
    imu_power_enable();
    i2c_reg_write_byte_dt(&imu, 0x12, 0x01);
    k_msleep(50);
    imu_power_down();
    imu_power_disable();

    // Everything off - CPU sleeps at ~12uA waiting for button
    while (1) {
        k_sem_take(&button_sem, K_FOREVER);
        k_msleep(150); // Debounce

        imu_power_enable();
        imu_wake_up();
        record_gesture();
        imu_power_down();
        imu_power_disable();

        uint8_t gesture_class = run_inference();

        // BLE powers up, broadcasts 600ms, shuts itself down via work queue
        broadcast_result((gesture_class >= 1 && gesture_class <= 5) ? gesture_class : 0);
    }

    return 0;
}