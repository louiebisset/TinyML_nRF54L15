#include <stdio.h>
#include <string.h>
#include <math.h>

#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/pm/device.h>
#include <zephyr/pm/device_runtime.h>
#include <zephyr/sys_clock.h>
#include <arm_math.h>
#include <arm_const_structs.h>
#include <algorithm>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "dsp_constants.h"

// ── 1. CONFIGURATION 

#define SAMPLE_RATE      16000
#define TOTAL_SECONDS    1.5f
#define DISCARD_SECONDS  0.5f
#define INF_SECONDS      1.0f

#define TOTAL_SAMPLES    ((int)(SAMPLE_RATE * TOTAL_SECONDS))    // 24,000
#define DISCARD_SAMPLES  ((int)(SAMPLE_RATE * DISCARD_SECONDS))  //  8,000
#define INF_SAMPLES      ((int)(SAMPLE_RATE * INF_SECONDS))      // 16,000
#define BLOCK_SIZE       2048

#define N_MFCC           13
#define FFT_SIZE         512
#define HOP_LENGTH       512
#define TIME_STEPS       (INF_SAMPLES / HOP_LENGTH)   // 31
#define N_BINS           (FFT_SIZE / 2 + 1)           // 257

// ── 2. HARDWARE NODES 

static const struct device *const dmic_dev =
    DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct gpio_dt_spec button =
    GPIO_DT_SPEC_GET(DT_ALIAS(sw0),  gpios);
//static const struct gpio_dt_spec led0 = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

// ── 3. AUDIO GLOBALS 

static int16_t rx_buffer[TOTAL_SAMPLES];
K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 8, 4);

static K_SEM_DEFINE(button_sem, 0, 1);
static struct gpio_callback button_cb_data;

// ── 4. DSP BUFFERS 
// All static — no VLAs, no stack pressure.

static arm_rfft_fast_instance_f32 rfft_inst;

static float32_t frame_buf[FFT_SIZE];   // windowed audio frame
static float32_t fft_buf[FFT_SIZE];     // complex RFFT output (interleaved re/im)
static float32_t power_buf[N_BINS];     // |X[k]|^2
static float32_t mel_buf[N_MFCC];      // mel filterbank energy
static float32_t log_mel_buf[N_MFCC];  // ln(mel_energy)
static float32_t mfcc_buf[N_MFCC];     // DCT-II output

// Precomputed cumulative offsets into mel_filters_packed[]
static uint32_t filter_coef_offset[N_MFCC];

// ── 5. AI BUFFERS 

constexpr int kArenaSize = 10 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor            = nullptr;
static TfLiteTensor *output_tensor           = nullptr;

// ── 6. BUTTON ISR 

static void button_isr(const struct device *dev,
                        struct gpio_callback *cb,
                        uint32_t pins)
{
    k_sem_give(&button_sem);
}
static void compute_mfcc_for_frame(const float32_t *audio_frame,
                                    float32_t       *mfcc_out)
{
    // Step 1: Hann window
    arm_mult_f32(audio_frame,
                 const_cast<float32_t *>(hann_window),
                 frame_buf,
                 FFT_SIZE);

    // Step 2: Real FFT
    // Layout: [re0, re_N/2, re1, im1, re2, im2, ..., re_{N/2-1}, im_{N/2-1}]
    arm_rfft_fast_f32(&rfft_inst, frame_buf, fft_buf, 0 /* forward */);

    // Step 3: Power spectrum
    // DC (k=0) and Nyquist (k=N/2) are packed as pure reals in fft_buf[0..1]
    power_buf[0]          = fft_buf[0] * fft_buf[0];
    power_buf[N_BINS - 1] = fft_buf[1] * fft_buf[1];
    for (int k = 1; k < N_BINS - 1; k++) {
        float re = fft_buf[2 * k];
        float im = fft_buf[2 * k + 1];
        power_buf[k] = re * re + im * im;
    }

    // Step 4: Sparse mel filterbank
    for (int m = 0; m < N_MFCC; m++) {
        float32_t energy = 0.0f;
        arm_dot_prod_f32(
            &power_buf[mel_filter_pos[m]],
            const_cast<float32_t *>(&mel_filters_packed[filter_coef_offset[m]]),
            mel_filter_len[m],
            &energy);
        mel_buf[m] = energy;
    }

    // Step 5: Natural log — +1e-6 prevents -inf on silent frames
    for (int m = 0; m < N_MFCC; m++) {
        log_mel_buf[m] = logf(mel_buf[m] + 1e-6f);
    }

    // Step 6: DCT-II via matrix-vector multiply
    arm_matrix_instance_f32 dct_mat;
    arm_mat_init_f32(&dct_mat,
                     N_MFCC, N_MFCC,
                     const_cast<float32_t *>(dct_coefs));
    arm_mat_vec_mult_f32(&dct_mat, log_mel_buf, mfcc_out);
}

// ── 8. AI SETUP 

static void setup_ai(void)
{
    printk("Initializing AI system (manual MFCC pipeline)...\n");

    // Initialise RFFT — the only CMSIS-DSP init needed
    if (arm_rfft_fast_init_f32(&rfft_inst, FFT_SIZE) != ARM_MATH_SUCCESS) {
        printk("ERROR: RFFT init failed!\n");
        return;
    }

    // Precompute cumulative offsets into mel_filters_packed[]
    // filter_coef_offset[m] = sum of mel_filter_len[0 .. m-1]
    uint32_t offset = 0;
    for (int m = 0; m < N_MFCC; m++) {
        filter_coef_offset[m] = offset;
        offset += mel_filter_len[m];
    }
    printk("RFFT ready. Mel filter offsets computed. Total packed coeffs: %u\n",
           (unsigned)offset);

    // Load TFLite model
    const tflite::Model *model = tflite::GetModel(g_audio_model_v2);

    static tflite::MicroMutableOpResolver<17> resolver;
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
        model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("ERROR: Tensor allocation failed!\n");
        return;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    size_t used = interpreter->arena_used_bytes();
    printk("AI Ready. Arena: %u / %u bytes (%u%% used)\n",
           (unsigned)used,
           (unsigned)kArenaSize,
           (unsigned)((used * 100) / kArenaSize));
}

// ── 9. AUDIO RECORDING 

static void record_audio(void)
{
    uint32_t bytes_recorded      = 0;
    const uint32_t discard_bytes = (uint32_t)DISCARD_SAMPLES * sizeof(int16_t);
    bool speak_led_on            = false;

    struct pcm_stream_cfg stream = {
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
        .streams = &stream,
        .channel = {
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
            .req_chan_map_hi = 0,
            .req_num_chan    = 1,
            .req_num_streams = 1
        }
    };

    if (dmic_configure(dmic_dev, &cfg) < 0) {
        printk("ERROR: DMIC configuration failed!\n");
        return;
    }
    printk("Warming up mic...\n");
    dmic_trigger(dmic_dev, DMIC_TRIGGER_START);

    while (bytes_recorded < sizeof(rx_buffer)) {
        void    *buffer;
        uint32_t size;

        if (dmic_read(dmic_dev, 0, &buffer, &size, 1000) < 0) {
            printk("ERROR: DMIC read failed!\n");
            break;
        }

        if (bytes_recorded + size <= sizeof(rx_buffer)) {
            memcpy((uint8_t *)rx_buffer + bytes_recorded, buffer, size);
        }
        bytes_recorded += size;

        // Switch LED on as soon as the discard window has passed
        if (!speak_led_on && bytes_recorded >= discard_bytes) {
            //gpio_pin_set_dt(&led0, 0);  // LED ON — speak now
            printk("SPEAK NOW\n");
            speak_led_on = true;
        }

        k_mem_slab_free(&mem_slab, buffer);
    }

    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    //gpio_pin_set_dt(&led0, 1);  // LED OFF — done recording
    printk("Recording complete.\n");
}

// ── 10. INFERENCE WITH SEPARATE TIMING 
//
// Timing breakdown:
//   dsp_us   – time spent extracting all MFCC frames (compute_mfcc_for_frame
//               for every TIME_STEPS hop, including normalisation & quantisation)
//   invoke_us – time spent inside interpreter->Invoke() only


static void run_inference(void)
{
    if (interpreter == nullptr) {
        printk("ERROR: Interpreter not initialised.\n");
        return;
    }

    if (input_tensor == nullptr || output_tensor == nullptr) {
        printk("ERROR: Input or output tensor is not initialised.\n");
        return;
    }

    int8_t *input_ptr = tflite::GetTensorData<int8_t>(input_tensor);
    float32_t mfcc_sum = 0.0f;

    printk("Extracting MFCC features (%d frames x %d coefficients)...\n",
           TIME_STEPS, N_MFCC);

    // Start DSP timing
    uint32_t dsp_start = k_cycle_get_32();

    for (int t = 0; t < TIME_STEPS; t++) {
        const int base = DISCARD_SAMPLES + t * HOP_LENGTH;

        // Convert PCM audio samples to normalised floating-point values
        float32_t audio_in[FFT_SIZE];

        for (int i = 0; i < FFT_SIZE; i++) {
            audio_in[i] =
                static_cast<float32_t>(rx_buffer[base + i]) / 32768.0f;
        }

        // Calculate MFCC features for this frame
        compute_mfcc_for_frame(audio_in, mfcc_buf);

        // Normalise, quantise and store features in the input tensor
        // Tensor layout: [N_MFCC, TIME_STEPS]
        // Index: f * TIME_STEPS + t
        for (int f = 0; f < N_MFCC; f++) {
            const int idx = f * TIME_STEPS + t;

            const float normalised =
                (mfcc_buf[f] - PYTHON_MEAN) / PYTHON_STD;

            const float q =
                (normalised / input_tensor->params.scale) +
                input_tensor->params.zero_point;

            const float q_clamped =
                std::max(-128.0f, std::min(127.0f, q));

            input_ptr[idx] = static_cast<int8_t>(q_clamped);

            mfcc_sum += mfcc_buf[f];
        }
    }

    // End DSP timing
    const uint32_t dsp_us =
        k_cyc_to_us_near32(k_cycle_get_32() - dsp_start);

    printk("  [DBG] Mean raw MFCC: %d (x1000)\n",
           static_cast<int>(
               (mfcc_sum / (N_MFCC * TIME_STEPS)) * 1000.0f));

    // Start model inference timing
    const uint32_t invoke_start = k_cycle_get_32();

    const TfLiteStatus invoke_status = interpreter->Invoke();

    // End model inference timing
    const uint32_t invoke_us =
        k_cyc_to_us_near32(k_cycle_get_32() - invoke_start);

    if (invoke_status == kTfLiteOk) {
        const int8_t raw = output_tensor->data.int8[0];

        const float val =
            (static_cast<int>(raw) - output_tensor->params.zero_point) *
            output_tensor->params.scale;

        printk("\n>>> INFERENCE METRICS <<<\n");
        printk("DSP Time       : %u us (%u.%03u ms)\n",
               dsp_us, dsp_us / 1000, dsp_us % 1000);
        printk("Invoke Time    : %u us (%u.%03u ms)\n", invoke_us, invoke_us / 1000, invoke_us % 1000);
        printk("Total Time     : %u us (%u.%03u ms)\n", dsp_us + invoke_us,
               (dsp_us + invoke_us) / 1000,
               (dsp_us + invoke_us) % 1000);
        printk("Result         : %s\n", (val > 0.8f) ? "DETECTED" : "NOISE");
        printk("Confidence     : %d%%\n",
               static_cast<int>(val * 100.0f));
        printk("Arena Used     : %u bytes\n",
               static_cast<unsigned int>(
                   interpreter->arena_used_bytes()));
        printk("--------------------------\n");
    } else {
        printk("Inference Invoke Failed! Status: %d\n",
               static_cast<int>(invoke_status));
    }
}
// ── 11. MAIN 

int main(void)
{
    k_msleep(500);  // Allow USB-serial to settle
    printk("\n*** nRF54L15 Audio TinyML Benchmark ***\n");
    printk("    DSP: manual MFCC pipeline\n");
    printk("    Timing: DSP + Invoke() measured separately\n\n");

    if (!device_is_ready(dmic_dev)) {
        printk("ERROR: DMIC not ready!\n");
        return 0;
    }

    // Configure GPIO
    gpio_pin_configure_dt(&button, GPIO_INPUT | GPIO_PULL_UP);
    gpio_pin_interrupt_configure_dt(&button, GPIO_INT_EDGE_TO_ACTIVE);
    gpio_init_callback(&button_cb_data, button_isr, BIT(button.pin));
    gpio_add_callback(button.port, &button_cb_data);

    setup_ai();

    printk("Ready — press the button to record and classify.\n\n");

    while (1) {
        // Block until button press (ISR gives the semaphore)
        k_sem_take(&button_sem, K_FOREVER);

        pm_device_runtime_get(dmic_dev);
        record_audio();
        pm_device_runtime_put(dmic_dev);

        run_inference();

        printk("Press button to run again.\n\n");
    }
}