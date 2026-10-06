#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/display.h>
#include <zephyr/display/cfb.h>
#include <zephyr/logging/log.h>
#include <arm_math.h>
#include <arm_const_structs.h>
#include <stdio.h>
#include <string.h>
#include <math.h>   // logf

// TensorFlow Includes
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "dsp_constants.h"

LOG_MODULE_REGISTER(ai_runner);

// Forward declaration
void update_screen(const char* status, int confidence);

// --- 1. CONFIGURATION & MACROS ---
#define SAMPLE_RATE     16000
#define TOTAL_SECONDS   1.5f
#define DISCARD_SECONDS 0.5f
#define INF_SECONDS     1.0f

#define TOTAL_SAMPLES   (int)(SAMPLE_RATE * TOTAL_SECONDS)    // 24,000
#define DISCARD_SAMPLES (int)(SAMPLE_RATE * DISCARD_SECONDS)  //  8,000
#define INF_SAMPLES     (int)(SAMPLE_RATE * INF_SECONDS)      // 16,000
#define BLOCK_SIZE      2048

#define N_MFCC          13
#define FFT_SIZE        512
#define HOP_LENGTH      512
#define TIME_STEPS      (INF_SAMPLES / HOP_LENGTH)   // 31
#define N_BINS          (FFT_SIZE / 2 + 1)            // 257  (RFFT output bins)

// --- 2. GLOBALS ---

K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 8, 4);
static arm_rfft_fast_instance_f32 rfft_inst;
static float32_t frame_buf[FFT_SIZE];       // windowed audio frame
static float32_t fft_buf[FFT_SIZE];         // complex RFFT output (interleaved re/im)
static float32_t power_buf[N_BINS];         // |X[k]|^2  (N_BINS = FFT_SIZE/2+1)
static float32_t mel_buf[N_MFCC];           // mel filterbank energy per band
static float32_t log_mel_buf[N_MFCC];       // ln(mel_energy)
static float32_t mfcc_buf[N_MFCC];          // DCT output = final MFCCs

// TFLite Globals
constexpr int kArenaSize = 10 * 1024;

static int16_t rx_buffer[TOTAL_SAMPLES];
alignas(16) static uint8_t tensor_arena[kArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model*      model_ptr   = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Hardware Devices
static const struct device *const dmic_dev    = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct device *const display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));
static const struct gpio_dt_spec led    = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(DT_ALIAS(sw1),  gpios);

// Precomputed offset table so we never recompute filter slice starts
static uint32_t filter_coef_offset[N_MFCC];

//
static void compute_mfcc_for_frame(const float32_t *audio_frame,
                                    float32_t       *mfcc_out)
{
    arm_mult_f32(audio_frame,
                 const_cast<float32_t *>(hann_window),
                 frame_buf,
                 FFT_SIZE);
    arm_rfft_fast_f32(&rfft_inst, frame_buf, fft_buf, 0 /* forward */);
    power_buf[0]         = fft_buf[0] * fft_buf[0];           // DC, im=0
    power_buf[N_BINS - 1] = fft_buf[1] * fft_buf[1];          // Nyquist, im=0
    for (int k = 1; k < N_BINS - 1; k++) {
        float re = fft_buf[2 * k];
        float im = fft_buf[2 * k + 1];
        power_buf[k] = re * re + im * im;
    }
    for (int m = 0; m < N_MFCC; m++) {
        uint32_t pos    = mel_filter_pos[m];
        uint32_t length = mel_filter_len[m];
        uint32_t start  = filter_coef_offset[m];

        float32_t energy = 0.0f;
        arm_dot_prod_f32(&power_buf[pos],
                         const_cast<float32_t *>(&mel_filters_packed[start]),
                         length,
                         &energy);
        mel_buf[m] = energy;
    }
    for (int m = 0; m < N_MFCC; m++) {
        log_mel_buf[m] = logf(mel_buf[m] + 1e-6f);
    }
    arm_matrix_instance_f32 dct_mat;
    arm_mat_init_f32(&dct_mat,
                     N_MFCC, N_MFCC,
                     const_cast<float32_t *>(dct_coefs));
    arm_mat_vec_mult_f32(&dct_mat, log_mel_buf, mfcc_out);
}
// --- 4. AI SETUP --
void setup_ai() {
    printk("Initializing AI system (manual MFCC pipeline)...\n");

    // Initialise the RFFT instance — this is the only CMSIS-DSP init call needed.
    // arm_rfft_fast_init_f32 supports lengths 32, 64, 128, 256, 512, 1024, 2048, 4096.
    if (arm_rfft_fast_init_f32(&rfft_inst, FFT_SIZE) != ARM_MATH_SUCCESS) {
        printk("RFFT Init Failed!\n");
        return;
    }

    // Precompute cumulative offsets into mel_filters_packed[] for each filter.
    // filter_coef_offset[m] = sum of mel_filter_len[0..m-1]
    uint32_t offset = 0;
    for (int m = 0; m < N_MFCC; m++) {
        filter_coef_offset[m] = offset;
        offset += mel_filter_len[m];
    }

    printk("RFFT ready. Mel filter offsets computed.\n");

    // Load TFLite model
    model_ptr = tflite::GetModel(g_audio_model_v2);

    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddExpandDims();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddDepthwiseConv2D();
    resolver.AddReduceMax();

    static tflite::MicroInterpreter static_interpreter(
        model_ptr, resolver, tensor_arena, kArenaSize
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("Tensor Allocation FAILED!\n");
        return;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    printk("AI System Ready.\n");
}

// --- 5. INFERENCE ---

void run_inference() {
    if (interpreter == nullptr) return;

    int8_t *input_ptr = tflite::GetTensorData<int8_t>(input_tensor);

    printk("Extracting Features...\n");
    printk("  [DBG] rx_buffer[DISCARD]=%d\n", (int)rx_buffer[DISCARD_SAMPLES]);

    float32_t mfcc_sum = 0.0f;

    for (int t = 0; t < TIME_STEPS; t++) {

        // Build float audio frame for this time step
        const int base = DISCARD_SAMPLES + t * HOP_LENGTH;
        float32_t audio_in[FFT_SIZE];
        for (int i = 0; i < FFT_SIZE; i++) {
            audio_in[i] = (float32_t)rx_buffer[base + i] / 32768.0f;
        }

        // Run the manual MFCC pipeline — result goes into mfcc_buf[]
        compute_mfcc_for_frame(audio_in, mfcc_buf);

        // Normalise and quantise into the TFLite input tensor
        for (int f = 0; f < N_MFCC; f++) {
            // Tensor layout: row-major (N_MFCC × TIME_STEPS)
            // index = f * TIME_STEPS + t
            int   idx          = f * TIME_STEPS + t;
            float normalised   = (mfcc_buf[f] - PYTHON_MEAN) / PYTHON_STD;
            float quantised    = (normalised / input_tensor->params.scale)
                                 + input_tensor->params.zero_point;
            input_ptr[idx]     = (int8_t)std::max(-128.0f, std::min(127.0f, quantised));
            mfcc_sum          += mfcc_buf[f];
        }
    }

    printk("  [DBG] mean raw MFCC: %d (x1000)  — expect variation between speech/silence\n",
           (int)(mfcc_sum / (N_MFCC * TIME_STEPS) * 1000.0f));

    printk("Invoking Interpreter...\n");
    if (interpreter->Invoke() != kTfLiteOk) {
        update_screen("Error: Invoke", -1);
        return;
    }

    int8_t output_raw = output_tensor->data.int8[0];
    float  value_f    = (output_raw - output_tensor->params.zero_point)
                        * output_tensor->params.scale;

    int pct = (int)(value_f * 100.0f);
    printk("Inference Done. Confidence: %d%%\n", pct);
    update_screen(value_f > 0.8f ? "HELLO!" : "Noise", pct);
}

// --- 6. RECORDING ---

void update_screen(const char* status, int confidence) {
    if (!device_is_ready(display_dev)) return;
    cfb_framebuffer_clear(display_dev, false);
    cfb_print(display_dev, "Status:", 0, 0);
    cfb_print(display_dev, status, 0, 16);
    if (confidence >= 0) {
        char conf_str[20];
        snprintf(conf_str, sizeof(conf_str), "Conf: %d %%", confidence);
        cfb_print(display_dev, conf_str, 0, 32);
    }
    cfb_framebuffer_finalize(display_dev);
}

void record_and_classify() {
    uint32_t bytes_recorded      = 0;
    uint32_t discard_limit_bytes = DISCARD_SAMPLES * sizeof(int16_t);
    bool led_is_on = false;

    update_screen("Warmup...", -1);

    struct pcm_stream_cfg stream_cfg = {
        .pcm_rate   = SAMPLE_RATE,
        .pcm_width  = 16,
        .block_size = BLOCK_SIZE,
        .mem_slab   = &mem_slab
    };

    struct dmic_cfg cfg = {
        .io = {
            .min_pdm_clk_freq = 1000000,
            .max_pdm_clk_freq = 3500000
        },
        .streams = &stream_cfg,
        .channel = {
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
            .req_chan_map_hi = 0,
            .req_num_chan    = 1,
            .req_num_streams = 1
        }
    };

    if (dmic_configure(dmic_dev, &cfg) < 0) {
        printk("DMIC Configuration Failed!\n");
        return;
    }

    dmic_trigger(dmic_dev, DMIC_TRIGGER_START);

    while (bytes_recorded < (TOTAL_SAMPLES * sizeof(int16_t))) {
        void    *buffer;
        uint32_t size;

        if (dmic_read(dmic_dev, 0, &buffer, &size, 1000) < 0) {
            printk("DMIC Read Error!\n");
            break;
        }

        if (bytes_recorded + size <= sizeof(rx_buffer)) {
            memcpy((uint8_t *)rx_buffer + bytes_recorded, buffer, size);
        }
        bytes_recorded += size;

        if (!led_is_on && bytes_recorded >= discard_limit_bytes) {
            update_screen("SPEAK NOW", -1);
            gpio_pin_set_dt(&led, 0);
            led_is_on = true;
        }
        k_mem_slab_free(&mem_slab, buffer);
    }

    gpio_pin_set_dt(&led, 1);
    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);

    printk("Recording finished, starting inference.\n");
    run_inference();
}

// --- 7. MAIN ---

int main(void) {
    k_msleep(2000);
    printk("\n\n*** nRF54L15 Solar TinyML System Starting ***\n");

    if (!device_is_ready(dmic_dev)    ||
        !device_is_ready(display_dev) ||
        !gpio_is_ready_dt(&button)    ||
        !gpio_is_ready_dt(&led)) {
        printk("Error: Hardware peripherals not ready!\n");
        return 0;
    }

    display_set_pixel_format(display_dev, PIXEL_FORMAT_MONO10);
    cfb_framebuffer_init(display_dev);
    cfb_framebuffer_clear(display_dev, true);
    display_blanking_off(display_dev);
    cfb_framebuffer_set_font(display_dev, 0);

    gpio_pin_configure_dt(&button, GPIO_INPUT);
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);

    setup_ai();
    update_screen("Ready", 0);

    while (1) {
        if (gpio_pin_get_dt(&button) > 0) {
            record_and_classify();
            k_msleep(1000);
            update_screen("Ready", 0);
        }
    }
}