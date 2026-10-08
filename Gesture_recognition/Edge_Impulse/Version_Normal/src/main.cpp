#include <stdio.h>
#include <string.h>
#include <math.h>

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>

/* Edge Impulse SDK */
#include <edge-impulse-sdk/classifier/ei_run_classifier.h>

// ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ   50
#define TOTAL_SAMPLES    150
#define SAMPLE_DELAY_MS  (1000 / SAMPLE_RATE_HZ)

#define LED_ON  0
#define LED_OFF 1

// ── 2. HARDWARE ───────────────────────────────────────────────────────────────
static const struct i2c_dt_spec imu = I2C_DT_SPEC_GET(DT_NODELABEL(lsm6dso));
static const struct device *const imu_dev = imu.bus;
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(DT_ALIAS(sw0),  gpios);
static const struct gpio_dt_spec led    = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios); // ENABLED!

// ── 3. GLOBALS ────────────────────────────────────────────────────────────────
struct RawSample { int16_t ax, ay, az, gx, gy, gz; };
static struct RawSample rx_buffer[TOTAL_SAMPLES];
static float ei_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// ── 4. HELPERS ────────────────────────────────────────────────────────────────
static void imu_wake_up(void) {
    // Set ODR to 416Hz (0x60) and configure full-scale ranges
    int err1 = i2c_reg_write_byte_dt(&imu, 0x10, 0x60); // Accel 416Hz, ±2g
    int err2 = i2c_reg_write_byte_dt(&imu, 0x11, 0x6C); // Gyro 416Hz, ±250dps
    
    // Set full-scale ranges explicitly (match training data collection)
    int err3 = i2c_reg_write_byte_dt(&imu, 0x15, 0x00); // ±2g scale
    int err4 = i2c_reg_write_byte_dt(&imu, 0x16, 0x00); // ±250dps scale
    
    if (err1 != 0 || err2 != 0 || err3 != 0 || err4 != 0) {
        printk("WARNING: Failed to configure IMU! (I2C Error)\n");
    }
    
    // CRITICAL: Increased delay to let IMU stabilize after power-up
    k_msleep(100); // Changed from 20ms to 100ms
}

static void imu_power_down(void) {
    i2c_reg_write_byte_dt(&imu, 0x10, 0x00); 
    i2c_reg_write_byte_dt(&imu, 0x11, 0x00);
}

static void button_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
    k_sem_give(&button_sem);
}

static int ei_get_data(size_t offset, size_t length, float *out_ptr) {
    if (out_ptr == NULL) return -1;
    memcpy(out_ptr, ei_buffer + offset, length * sizeof(float));
    return 0;
}

static void record_gesture(void) {
    // ═══════════════════════════════════════════════════════════════════════
    // CRITICAL: LED ON signals user to PERFORM GESTURE NOW!
    // ═══════════════════════════════════════════════════════════════════════
    gpio_pin_set_dt(&led, LED_ON);
    
    printk("🔴 LED ON - PERFORM GESTURE NOW!\n");

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        uint8_t raw[6];
        int ret;
        
        // Read Accel (0x28) with error checking
        ret = i2c_burst_read_dt(&imu, 0x28, raw, 6);
        if (ret != 0) {
            printk("ERR: Accel read failed at sample %d\n", i);
        }
        rx_buffer[i].ax = (int16_t)(raw[0] | (raw[1] << 8));
        rx_buffer[i].ay = (int16_t)(raw[2] | (raw[3] << 8));
        rx_buffer[i].az = (int16_t)(raw[4] | (raw[5] << 8));

        // Read Gyro (0x22) with error checking
        ret = i2c_burst_read_dt(&imu, 0x22, raw, 6);
        if (ret != 0) {
            printk("ERR: Gyro read failed at sample %d\n", i);
        }
        rx_buffer[i].gx = (int16_t)(raw[0] | (raw[1] << 8));
        rx_buffer[i].gy = (int16_t)(raw[2] | (raw[3] << 8));
        rx_buffer[i].gz = (int16_t)(raw[4] | (raw[5] << 8));

        k_msleep(SAMPLE_DELAY_MS);
    }
 
    // LED OFF signals recording complete
    gpio_pin_set_dt(&led, LED_OFF);
    printk("⚫ LED OFF - Recording complete!\n");
    
    // ═══════════════════════════════════════════════════════════════════════
    // DEBUG: Print sample data to verify sensor is reading correctly
    // ═══════════════════════════════════════════════════════════════════════
    printk("\n╔═══════════════════════════════════════════════════════════╗\n");
    printk("║           RAW SENSOR DATA (every 15th sample)             ║\n");
    printk("╠═══════════════════════════════════════════════════════════╣\n");
    for (int i = 0; i < TOTAL_SAMPLES; i += 15) {
        printk("║[%3d] ax=%6d ay=%6d az=%6d | gx=%6d gy=%6d gz=%6d║\n",
               i,
               rx_buffer[i].ax, rx_buffer[i].ay, rx_buffer[i].az,
               rx_buffer[i].gx, rx_buffer[i].gy, rx_buffer[i].gz);
    }
    
    // Calculate magnitude statistics
    float min_acc = 1000000.0f, max_acc = 0.0f;
    float min_gyro = 1000000.0f, max_gyro = 0.0f;
    float sum_acc = 0.0f, sum_gyro = 0.0f;
    
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        float ax = (float)rx_buffer[i].ax;
        float ay = (float)rx_buffer[i].ay;
        float az = (float)rx_buffer[i].az;
        float gx = (float)rx_buffer[i].gx;
        float gy = (float)rx_buffer[i].gy;
        float gz = (float)rx_buffer[i].gz;
        
        float acc_mag = sqrtf(ax*ax + ay*ay + az*az);
        float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);
        
        if (acc_mag < min_acc) min_acc = acc_mag;
        if (acc_mag > max_acc) max_acc = acc_mag;
        if (gyro_mag < min_gyro) min_gyro = gyro_mag;
        if (gyro_mag > max_gyro) max_gyro = gyro_mag;
        
        sum_acc += acc_mag;
        sum_gyro += gyro_mag;
    }
    
    float avg_acc = sum_acc / TOTAL_SAMPLES;
    float avg_gyro = sum_gyro / TOTAL_SAMPLES;
    
    printk("╠═══════════════════════════════════════════════════════════╣\n");
    printk("║ Acc Magnitude:  min=%6.0f  avg=%6.0f  max=%6.0f     ║\n", 
           min_acc, avg_acc, max_acc);
    printk("║ Gyro Magnitude: min=%6.0f  avg=%6.0f  max=%6.0f     ║\n", 
           min_gyro, avg_gyro, max_gyro);
    printk("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Diagnostic hints
    if (max_acc < 5000) {
        printk("⚠️  WARNING: Very low accelerometer readings - sensor may not be working!\n");
    }
    if (max_gyro < 500 && min_gyro < 100) {
        printk("⚠️  WARNING: Very low gyro variation - did you move the device?\n");
    }
    if (max_acc == min_acc || max_gyro == min_gyro) {
        printk("⚠️  WARNING: No variation in readings - sensor is stuck!\n");
    }
}

// ── 5. INFERENCE ──────────────────────────────────────────────────────────────
static void run_inference(void) {
    size_t f_idx = 0;

    // Passing data to Edge Impulse in the SAME format as training
    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        float ax = (float)rx_buffer[i].ax;
        float ay = (float)rx_buffer[i].ay;
        float az = (float)rx_buffer[i].az;
        float gx = (float)rx_buffer[i].gx;
        float gy = (float)rx_buffer[i].gy;
        float gz = (float)rx_buffer[i].gz;

        float acc_mag  = sqrtf(ax*ax + ay*ay + az*az);
        float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);
        
        // Match the 8-feature axis order from your Python preprocessing
        float feat[8]  = {ax, ay, az, acc_mag, gx, gy, gz, gyro_mag};

        for (int j = 0; j < 8; j++) {
            if (f_idx < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
                ei_buffer[f_idx++] = feat[j];
            }
        }
    }
    
    printk("\nBuffer filled: %zu / %d samples\n", f_idx, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    
    if (f_idx != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        printk("⚠️  WARNING: Buffer size mismatch! Expected %d, got %zu\n", 
               EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, f_idx);
    }

    signal_t signal;
    memset(&signal, 0, sizeof(signal_t));
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &ei_get_data;

    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);

    if (err != EI_IMPULSE_OK) {
        printk("ERR: run_classifier failed (%d)\n", err);
        return;
    }

    // Classification Report
    printk("\n╔═══════════════════════════════════════╗\n");
    printk("║      GESTURE CLASSIFICATION REPORT    ║\n");
    printk("╠═══════════════════════════════════════╣\n");
    printk("║ DSP Time:       %6d ms            ║\n", result.timing.dsp);
    printk("║ Inference Time: %6d ms            ║\n", result.timing.classification);
    printk("╠═══════════════════════════════════════╣\n");
    printk("║ All Predictions:                      ║\n");
    
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        printk("║   %-12s: %5.2f%% (%7.4f)    ║\n", 
               result.classification[i].label, 
               result.classification[i].value * 100.0f,
               result.classification[i].value);
    }

    int best_idx = 0;
    float best_val = 0.0f;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > best_val) {
            best_val = result.classification[i].value;
            best_idx = i;
        }
    }
    
    printk("╠═══════════════════════════════════════╣\n");
    printk("║ >>> WINNER: %-12s            ║\n", result.classification[best_idx].label);
    printk("║ >>> CONFIDENCE: %5.1f%%               ║\n", best_val * 100.0f);
    printk("╚═══════════════════════════════════════╝\n\n");
}

// ── 6. MAIN ───────────────────────────────────────────────────────────────────
int main(void) {
    k_msleep(500);

    // 1. GPIO FIRST (no bus needed)
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);  // ENABLED!
    gpio_pin_set_dt(&led, LED_OFF);                      // Ensure LED starts OFF
    
    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    // 2. PM ENABLE, THEN GET (wake bus), THEN I2C
    pm_device_runtime_enable(imu_dev);
    pm_device_runtime_get(imu_dev);

    i2c_reg_write_byte_dt(&imu, 0x12, 0x01); // Software Reset
    k_msleep(50);

    // Optional: verify sensor NOW (bus is awake)
    imu_power_down();
    pm_device_runtime_put(imu_dev);

    printk("\n");
    printk("╔════════════════════════════════════════════════════════╗\n");
    printk("║  Edge Impulse Gesture Recognition - DEBUG MODE        ║\n");
    printk("╠════════════════════════════════════════════════════════╣\n");
    printk("║  Expected Input: 150 samples × 8 features             ║\n");
    printk("║  Features: ax, ay, az, acc_mag, gx, gy, gz, gyro_mag  ║\n");
    printk("║  Sample Rate: 50 Hz (20ms per sample)                 ║\n");
    printk("║  Duration: 3.0 seconds                                ║\n");
    printk("╠════════════════════════════════════════════════════════╣\n");
    printk("║  IMPORTANT INSTRUCTIONS:                              ║\n");
    printk("║  1. Press sw0 button                                  ║\n");
    printk("║  2. WAIT for LED to turn ON                           ║\n");
    printk("║  3. PERFORM GESTURE while LED is ON                   ║\n");
    printk("║  4. LED will turn OFF after 3 seconds                 ║\n");
    printk("╠════════════════════════════════════════════════════════╣\n");
    printk("║  READY - Press sw0 to start recording                 ║\n");
    printk("╚════════════════════════════════════════════════════════╝\n\n");

    while (1) {
        k_sem_take(&button_sem, K_FOREVER);
        
        printk("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printk("Button pressed! Preparing to record...\n");
        
        k_msleep(150); // Debounce delay
        
        pm_device_runtime_get(imu_dev);
        imu_wake_up();
        
        // CRITICAL: Add extra delay after wake-up before recording
        k_msleep(50); // Give IMU time to settle after configuration
        
        printk("\n⏳ GET READY - LED will turn ON in a moment...\n\n");
        
        record_gesture();
        
        imu_power_down();
        pm_device_runtime_put(imu_dev);
        
        printk("\n🔍 Analyzing...\n");
        run_inference();
        
        printk("\n✅ Ready for next gesture. Press sw0 again.\n");
        printk("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
    }
}