#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm> // For std::clamp

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/i2c.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys_clock.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Update these to match your latest exported file names!
#include "model_simple.h" 
#include "scaler_params.h"

// ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ   50
#define TOTAL_SAMPLES    150
#define SAMPLE_DELAY_MS  (1000 / SAMPLE_RATE_HZ)
#define NUM_CLASSES      5

#define LED_ON  0
#define LED_OFF 1

// ── 2. HARDWARE NODES ─────────────────────────────────────────────────────────
static const struct i2c_dt_spec imu = I2C_DT_SPEC_GET(DT_NODELABEL(lsm6dso));
static const struct device *const imu_dev = imu.bus;

static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(DT_ALIAS(sw0),  gpios);
static const struct gpio_dt_spec led    = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

// ── 3. GLOBALS ────────────────────────────────────────────────────────────────
struct RawSample { int16_t ax, ay, az, gx, gy, gz; };
static struct RawSample rx_buffer[TOTAL_SAMPLES];

static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// ── 4. AI BUFFERS ─────────────────────────────────────────────────────────────
// Increased to 20KB to ensure the Deep Model has enough workspace
constexpr int kArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

static tflite::MicroInterpreter *interpreter  = nullptr;
static TfLiteTensor             *input_tensor  = nullptr;
static TfLiteTensor             *output_tensor = nullptr;

// ── 5. BUTTON ISR ─────────────────────────────────────────────────────────────
static void button_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
    k_sem_give(&button_sem);
}

// ── 6. IMU LOW-LEVEL POWER HELPERS ───────────────────────────────────────────
static void imu_wake_up(void) {
    i2c_reg_write_byte_dt(&imu, 0x10, 0x60); 
    i2c_reg_write_byte_dt(&imu, 0x11, 0x6C);
    k_msleep(10); 
}

static void imu_power_down(void) {
    i2c_reg_write_byte_dt(&imu, 0x10, 0x00); 
    i2c_reg_write_byte_dt(&imu, 0x11, 0x00);
    k_busy_wait(100); 
}

// ── 7. AI SETUP ───────────────────────────────────────────────────────────────
static void setup_ai(void) {
    const tflite::Model *model = tflite::GetModel(model_simple);
    
    // We added BatchNormalization and Relu which are required for the Deep model
    static tflite::MicroMutableOpResolver<15> resolver;
    resolver.AddConv2D();
    resolver.AddL2Normalization();
    resolver.AddRelu();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddExpandDims();
    // Supporting Ops often needed by TFLite converters
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMean();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;
    
    // CRITICAL: If this fails, we must stop to prevent the MPU FAULT (Null Pointer)
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("!!! AI ALLOCATION FAILED. Check Arena Size or Ops.\n");
        return; 
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    if (input_tensor != nullptr) {
        printk("AI Ready. Input size: %d bytes\n", input_tensor->bytes);
    }
}

// ── 8. GESTURE RECORDING ──────────────────────────────────────────────────────
static void record_gesture(void) {
    //gpio_pin_set_dt(&led, LED_ON);

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

    //gpio_pin_set_dt(&led, LED_OFF);
}

// ── 9. INFERENCE ──────────────────────────────────────────────────────────────
static void run_inference(void) {
    if (interpreter == nullptr || input_tensor == nullptr) return;

    // ── 1. Preprocessing Timing Start ────────────────────────────────────────
    uint32_t dsp_start = k_cycle_get_32();

    int8_t *input_ptr = tflite::GetTensorData<int8_t>(input_tensor);
    const float in_scale = input_tensor->params.scale;
    const int in_zp = input_tensor->params.zero_point;
    size_t input_idx = 0;

    for (int i = 0; i < TOTAL_SAMPLES; i++) {
        float ax = (float)rx_buffer[i].ax;
        float ay = (float)rx_buffer[i].ay;
        float az = (float)rx_buffer[i].az;
        float gx = (float)rx_buffer[i].gx;
        float gy = (float)rx_buffer[i].gy;
        float gz = (float)rx_buffer[i].gz;

        float acc_mag  = sqrtf(ax*ax + ay*ay + az*az);
        float gyro_mag = sqrtf(gx*gx + gy*gy + gz*gz);
        float feat[8]  = {ax, ay, az, acc_mag, gx, gy, gz, gyro_mag};

        for (int j = 0; j < 8; j++) {
            if (input_idx < input_tensor->bytes) {
                float norm = (feat[j] - SENSOR_MEAN[j]) / SENSOR_STD[j];
                int8_t quant = (int8_t)std::clamp<int>((int)roundf(norm / in_scale + in_zp), -128, 127);
                input_ptr[input_idx++] = quant;
            }
        }
    }
    uint32_t dsp_us = k_cyc_to_us_near32(k_cycle_get_32() - dsp_start);

    // ── 2. Invoke Timing Start ───────────────────────────────────────────────
    uint32_t invoke_start = k_cycle_get_32();
    TfLiteStatus invoke_status = interpreter->Invoke();
    uint32_t invoke_us = k_cyc_to_us_near32(k_cycle_get_32() - invoke_start);

    if (invoke_status != kTfLiteOk) {
        printk("ERROR: Inference failed!\n");
        return;
    }

    // ── 3. Memory & Metadata ─────────────────────────────────────────────────
    size_t used = interpreter->arena_used_bytes();
    int    pct  = (int)((used * 100) / kArenaSize);
    uint32_t total_us = dsp_us + invoke_us;

    // Find best class
    int8_t max_val = -128;
    int best_idx = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (output_tensor->data.int8[i] > max_val) {
            max_val = output_tensor->data.int8[i];
            best_idx = i;
        }
    }

    // Dequantize the output for "Confidence" percentage
    float confidence = (max_val - output_tensor->params.zero_point) * output_tensor->params.scale;
    const char *labels[NUM_CLASSES] = {"double_tap", "flick", "lift", "rotate", "shake"};

    // ── 4. VISUAL REPORT ─────────────────────────────────────────────────────
    printk("\n╔══════════════════════════════╗\n");
    printk("║   GESTURE INFERENCE REPORT   ║\n");
    printk("╠══════════════════════════════╣\n");
    printk("║ Scaling/Mag : %6u us    ║\n", dsp_us);
    printk("║ TFLM Invoke : %6u us    ║\n", invoke_us);
    printk("║ Total Time  : %6u us    ║\n", total_us);
    printk("║              %4u.%03u ms   ║\n", total_us / 1000, total_us % 1000);
    printk("╠══════════════════════════════╣\n");
    printk("║ Arena Used  : %5zu bytes   ║\n", used);
    printk("║ Usage       : %5d %%       ║\n", pct);
    printk("╠══════════════════════════════╣\n");
    printk("║ Result      : %-12s ║\n", labels[best_idx]);
    printk("║ Confidence  : %5d %%        ║\n", (int)(confidence * 100));
    printk("╚══════════════════════════════╝\n\n");

    // Logic to confirm shape matches Python [1, 150, 8, 1]
    TfLiteTensor* inp = interpreter->input(0);
    printk("Input shape: [%d, %d, %d, %d]\n", 
           inp->dims->data[0], inp->dims->data[1], 
           inp->dims->data[2], inp->dims->data[3]);
}

// ── 10. MAIN ──────────────────────────────────────────────────────────────────
int main(void) {
    k_msleep(500); 
   
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
    gpio_pin_set_dt(&led, LED_OFF);
    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    setup_ai();

    pm_device_runtime_enable(imu_dev);
    pm_device_runtime_get(imu_dev);
    i2c_reg_write_byte_dt(&imu, 0x12, 0x01); // Reset
    k_msleep(50);
    imu_power_down(); 
    pm_device_runtime_put(imu_dev);

    printk("System Active. Waiting for button...\n");

    while (1) {
        k_sem_take(&button_sem, K_FOREVER);

        pm_device_runtime_get(imu_dev);
        imu_wake_up();
       
        printk("Recording...");
        record_gesture();
       
        imu_power_down();
        pm_device_runtime_put(imu_dev);

        printk("Processing...\n");
        run_inference();
    }
}