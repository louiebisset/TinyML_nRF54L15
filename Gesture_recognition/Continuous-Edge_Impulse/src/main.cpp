#include <stdio.h>
#include <string.h>
#include <math.h>

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>

/* Edge Impulse SDK */
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ      50
#define TOTAL_SAMPLES       150
#define SAMPLE_DELAY_MS     (1000 / SAMPLE_RATE_HZ)
#define IDLE_COOLDOWN_MS    500       

// ── 2. HARDWARE & KERNEL OBJECTS ──────────────────────────────────────────────
static const struct i2c_dt_spec imu = I2C_DT_SPEC_GET(DT_NODELABEL(lsm6dso));
static const struct gpio_dt_spec imu_int = GPIO_DT_SPEC_GET(DT_ALIAS(imu_int), irq_gpios);

static struct gpio_callback imu_int_cb_data;
K_SEM_DEFINE(motion_sem, 0, 1);

// ── 3. GLOBALS ────────────────────────────────────────────────────────────────
struct RawSample { int16_t ax, ay, az, gx, gy, gz; };
static struct RawSample rx_buffer[TOTAL_SAMPLES];
static float ei_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// ── 4. ISR ────────────────────────────────────────────────────────────────────
static void imu_int_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
    k_sem_give(&motion_sem);
}

// ── 5. IMU POWER MODES ───────────────────────────────────────────────────────

// LOW POWER: Accel at 12.5Hz only, gyro OFF. Just enough for wake-up detection.
// This is the idle state — draws ~0.01mA from IMU vs ~0.5mA at 104Hz both sensors.
static void imu_enter_wakeup_mode(void) {
    // Gyro OFF
    i2c_reg_write_byte_dt(&imu, 0x11, 0x00);
    // Accel 12.5Hz low-power (enough for wake-up threshold detection)
    i2c_reg_write_byte_dt(&imu, 0x10, 0x10);
}

// FULL POWER: Both sensors at 52Hz — matches the 50Hz firmware read rate.
// Only active during the 3-second recording window.
static void imu_enter_recording_mode(void) {
    i2c_reg_write_byte_dt(&imu, 0x10, 0x30);  // Accel 52Hz
    i2c_reg_write_byte_dt(&imu, 0x11, 0x30);  // Gyro 52Hz
    k_msleep(10);  // let sensors stabilise
}

static int imu_init(void) {
    if (!device_is_ready(imu.bus)) return -1;

    // Software reset
    i2c_reg_write_byte_dt(&imu, 0x12, 0x01);
    k_msleep(50);

    // Start in recording mode briefly to configure wake-up
    i2c_reg_write_byte_dt(&imu, 0x10, 0x40);  // 104Hz for init
    i2c_reg_write_byte_dt(&imu, 0x11, 0x00);  // Gyro off during init
    k_msleep(20);

    // Wake-up detection config
    i2c_reg_write_byte_dt(&imu, 0x56, 0x50);  // LIR + slope filter
    i2c_reg_write_byte_dt(&imu, 0x58, 0x80);  // Global interrupt enable
    i2c_reg_write_byte_dt(&imu, 0x5B, 0x02);  // Threshold ~62mg
    i2c_reg_write_byte_dt(&imu, 0x5C, 0x00);  // Immediate trigger
    i2c_reg_write_byte_dt(&imu, 0x5E, 0x20);  // Route to INT1

    // Drop to low-power wake-up mode
    imu_enter_wakeup_mode();

    return 0;
}

// Single 12-byte burst read instead of two 6-byte reads
// Gyro registers (0x22-0x27) and accel registers (0x28-0x2D) are contiguous
static void imu_read_sample(struct RawSample *s) {
    uint8_t raw[12];
    i2c_burst_read_dt(&imu, 0x22, raw, 12);
    s->gx = (int16_t)(raw[0]  | (raw[1]  << 8));
    s->gy = (int16_t)(raw[2]  | (raw[3]  << 8));
    s->gz = (int16_t)(raw[4]  | (raw[5]  << 8));
    s->ax = (int16_t)(raw[6]  | (raw[7]  << 8));
    s->ay = (int16_t)(raw[8]  | (raw[9]  << 8));
    s->az = (int16_t)(raw[10] | (raw[11] << 8));
}

// ── 6. RECORD GESTURE ────────────────────────────────────────────────────────
static void record_gesture(void) {
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        imu_read_sample(&rx_buffer[i]);
        k_msleep(SAMPLE_DELAY_MS);
    }
}

// ── 7. EDGE IMPULSE INFERENCE ─────────────────────────────────────────────────
static int ei_get_data(size_t offset, size_t length, float *out_ptr) {
    if (out_ptr == NULL) return -1;
    memcpy(out_ptr, ei_buffer + offset, length * sizeof(float));
    return 0;
}

static void run_inference(void) {
    size_t f_idx = 0;

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        float ax = (float)rx_buffer[i].ax;
        float ay = (float)rx_buffer[i].ay;
        float az = (float)rx_buffer[i].az;
        float gx = (float)rx_buffer[i].gx;
        float gy = (float)rx_buffer[i].gy;
        float gz = (float)rx_buffer[i].gz;

        float acc_mag  = sqrtf(ax*ax + ay*ay + az*az);
        float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);

        float feat[8] = {ax, ay, az, acc_mag, gx, gy, gz, gyro_mag};

        for (int j = 0; j < 8; j++) {
            if (f_idx < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
                ei_buffer[f_idx++] = feat[j];
            }
        }
    }

    signal_t signal = { 0 };
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &ei_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    if (err != EI_IMPULSE_OK) return;

    // Find best class — no printk, just use the result
    int best_idx = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }

    // TODO: Replace with BLE advertising or GPIO action
    // No printk here — UART adds ~6.5mA
}

// ── 8. MAIN ───────────────────────────────────────────────────────────────────
int main(void) {
    k_msleep(500);

    // Configure IMU interrupt pin
    if (!gpio_is_ready_dt(&imu_int)) return 0;
    gpio_pin_configure_dt(&imu_int, GPIO_INPUT);
    gpio_pin_interrupt_configure_dt(&imu_int, GPIO_INT_EDGE_RISING);
    gpio_init_callback(&imu_int_cb_data, imu_int_isr, BIT(imu_int.pin));
    gpio_add_callback(imu_int.port, &imu_int_cb_data);

    if (imu_init() != 0) return 0;

    // Clear any spurious interrupt from startup
    uint8_t dummy;
    i2c_reg_read_byte_dt(&imu, 0x1B, &dummy);
    k_sem_reset(&motion_sem);

    while (1) {
        // 1. Sleep until wake-up interrupt (IMU at 12.5Hz, gyro off)
        k_sem_take(&motion_sem, K_FOREVER);

        // 2. Ramp up to full recording mode
        imu_enter_recording_mode();

        // 3. Record 3 seconds of gesture data
        record_gesture();

        // 4. Run inference
        run_inference();

        // 5. Clear interrupt + drop back to low-power
        i2c_reg_read_byte_dt(&imu, 0x1B, &dummy);
        imu_enter_wakeup_mode();

        // 6. Cooldown + clear bounce
        k_msleep(IDLE_COOLDOWN_MS);
        k_sem_reset(&motion_sem);
    }
}