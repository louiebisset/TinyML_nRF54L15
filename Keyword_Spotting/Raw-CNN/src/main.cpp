#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/regulator.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>
#include <string.h>

// For high-precision timing helpers
#include <zephyr/sys_clock.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// --- 1. CONFIGURATION ---
#define SAMPLE_RATE     16000
#define TOTAL_SAMPLES   ((int)(16000 * 1.5f))
#define DISCARD_SAMPLES ((int)(16000 * 0.5f))
#define INF_SAMPLES     ((int)(16000 * 1.0f))
#define BLOCK_SIZE      2048

// --- 2. HARDWARE NODES ---
static const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct gpio_dt_spec button    = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static const struct gpio_dt_spec led0      = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

// --- 3. GLOBALS ---
static int16_t rx_buffer[TOTAL_SAMPLES];
K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 8, 4);
static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// --- 4. AI BUFFERS ---
constexpr int kArenaSize = 55 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor            = nullptr;
static TfLiteTensor *output_tensor           = nullptr;

// --- 5. BUTTON ISR ---
static void button_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins) {
    k_sem_give(&button_sem);
}

// --- 6. AI SETUP ---
static void setup_ai(void) {
    const tflite::Model *model = tflite::GetModel(g_audio_model_v2);
    static tflite::MicroMutableOpResolver<17> resolver;
    resolver.AddExpandDims();      resolver.AddReshape();        resolver.AddShape();
    resolver.AddStridedSlice();    resolver.AddPack();           resolver.AddConv2D();
    resolver.AddMaxPool2D();       resolver.AddFullyConnected(); resolver.AddLogistic();
    resolver.AddMul();             resolver.AddAdd();            resolver.AddMean();
    resolver.AddQuantize();        resolver.AddDequantize();     resolver.AddPad();
    resolver.AddDepthwiseConv2D(); resolver.AddReduceMax();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() == kTfLiteOk) {
        input_tensor  = interpreter->input(0);
        output_tensor = interpreter->output(0);
        
        size_t used = interpreter->arena_used_bytes();
        printk("AI Ready. Arena Usage: %d / %d bytes (%d%% used)\n", 
                used, kArenaSize, (used * 100) / kArenaSize);
    }
}

// --- 7. AUDIO RECORDING ---
static void record_audio(void) {
    uint32_t bytes_recorded = 0;
    struct pcm_stream_cfg stream = {
        .pcm_rate   = SAMPLE_RATE,
        .pcm_width  = 16,
        .block_size = BLOCK_SIZE,
        .mem_slab   = &mem_slab
    };
    struct dmic_cfg cfg = {
        .io = { .min_pdm_clk_freq = 1000000, .max_pdm_clk_freq = 3500000 },
        .streams = &stream,
        .channel = {
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
            .req_num_chan    = 1,
            .req_num_streams = 1
        }
    };

    if (dmic_configure(dmic_dev, &cfg) < 0) return;

    //gpio_pin_set_dt(&led0, 0); // Active Low LED ON
    dmic_trigger(dmic_dev, DMIC_TRIGGER_START);

    while (bytes_recorded < sizeof(rx_buffer)) {
        void *buffer;
        uint32_t size;
        if (dmic_read(dmic_dev, 0, &buffer, &size, 100) >= 0) {
            memcpy((uint8_t *)rx_buffer + bytes_recorded, buffer, size);
            bytes_recorded += size;
            k_mem_slab_free(&mem_slab, buffer);
        }
    }

    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    //gpio_pin_set_dt(&led0, 1); // LED OFF
}

// --- 8. INFERENCE & HIGH-PRECISION TIMING ---
static void run_inference(void) {
    if (interpreter == nullptr) return;

    // Load data into input tensor
    int8_t *input_ptr = tflite::GetTensorData<int8_t>(input_tensor);
    for (int i = 0; i < INF_SAMPLES; i++) {
        input_ptr[i] = (int8_t)(rx_buffer[i + DISCARD_SAMPLES] >> 8);
    }

    // --- HIGH PRECISION START ---
    uint32_t start_cycles = k_cycle_get_32();
    
    TfLiteStatus invoke_status = interpreter->Invoke();
    
    uint32_t end_cycles = k_cycle_get_32();
    // --- HIGH PRECISION END ---

    // Calculate delta and convert to microseconds
    uint32_t diff_cycles = end_cycles - start_cycles;
    uint32_t diff_us = k_cyc_to_us_near32(diff_cycles);

    if (invoke_status == kTfLiteOk) {
        int8_t raw = output_tensor->data.int8[0];
        float val  = (raw - output_tensor->params.zero_point) * output_tensor->params.scale;
        
        printk("\n>>> INFERENCE METRICS <<<\n");
        printk("Execution Time : %u us (%u.%03u ms)\n", 
                diff_us, diff_us / 1000, diff_us % 1000);
        printk("Result         : %s\n", (val > 0.8f) ? "DETECTED" : "NOISE");
        printk("Confidence     : %d%%\n", (int)(val * 100));
        printk("Arena Used     : %d bytes\n", (int)interpreter->arena_used_bytes());
        printk("--------------------------\n");
    } else {
        printk("Inference Invoke Failed!\n");
    }
}

// --- 9. MAIN ---
int main(void) {
    printk("\n*** nRF54L15 Audio AI Benchmarking Tool ***\n");
    gpio_pin_set_dt(&led0, 1); // LED OFF
    //gpio_pin_configure_dt(&led0, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    k_msleep(500);

    setup_ai();

    printk("System Ready. Press the button to run benchmark.\n");

    while (1) {
        k_sem_take(&button_sem, K_FOREVER);

        if (device_is_ready(dmic_dev)) {
            pm_device_runtime_get(dmic_dev);
            record_audio();
            pm_device_runtime_put(dmic_dev);
        }

        run_inference();
    }
}