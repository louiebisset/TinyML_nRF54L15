/* * main.cpp - Fixed Active-Low LED Logic
 * Natural State: OFF (High)
 * Recording State: ON (Low) during the 1.0s window
 */

#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/sys/printk.h>
#include <string.h>

#define SAMPLE_RATE     16000
#define TOTAL_SAMPLES   24000 // 1.5s
#define DISCARD_SAMPLES 8000  // 0.5s
#define BLOCK_SIZE      2048

static const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
static const struct gpio_dt_spec button = GPIO_DT_SPEC_GET(DT_ALIAS(sw0), gpios);
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

// DEFINING THE INVERSION CLEARLY
// For Active-Low: 0 is LIGHT, 1 is DARK
#define MY_LED_ON  0
#define MY_LED_OFF 1

int16_t rx_buffer[TOTAL_SAMPLES]; 
K_MEM_SLAB_DEFINE_STATIC(mem_slab, BLOCK_SIZE, 4, 4);

void send_data_to_python() {
    printk("---START_DATA---\n");
    for (int i = DISCARD_SAMPLES; i < TOTAL_SAMPLES; i++) {
        printk("%d\n", rx_buffer[i]);
        if (i % 100 == 0) k_busy_wait(50); 
    }
    printk("---END_DATA---\n");
}

void record_clip() {
    uint32_t samples_collected = 0;
    
    // Ensure LED is OFF before starting
    gpio_pin_set_dt(&led, MY_LED_OFF); 

    struct dmic_cfg cfg = {
        .io = {.min_pdm_clk_freq = 1000000, .max_pdm_clk_freq = 3500000},
        .streams = &(struct pcm_stream_cfg){
            .pcm_rate = SAMPLE_RATE, .pcm_width = 16, .block_size = BLOCK_SIZE, .mem_slab = &mem_slab
        },
        .channel = {.req_num_streams = 1, .req_num_chan = 1, .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT)}
    };
    dmic_configure(dmic_dev, &cfg);
    dmic_trigger(dmic_dev, DMIC_TRIGGER_START);

    while (samples_collected < TOTAL_SAMPLES) {
        void *buffer;
        uint32_t size;
        int ret = dmic_read(dmic_dev, 0, &buffer, &size, 1000);
        
        if (ret == 0) {
            uint32_t num_new_samples = size / sizeof(int16_t);
            if (samples_collected + num_new_samples <= TOTAL_SAMPLES) {
                memcpy(&rx_buffer[samples_collected], buffer, size);
            }
            samples_collected += num_new_samples;

            // --- THE 1.0s WINDOW LOGIC ---
            if (samples_collected >= DISCARD_SAMPLES && samples_collected < TOTAL_SAMPLES) {
                gpio_pin_set_dt(&led, MY_LED_ON); // ON (0)
            } else {
                gpio_pin_set_dt(&led, MY_LED_OFF); // OFF (1)
            }
            
            k_mem_slab_free(&mem_slab, buffer);
        }
    }

    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    
    // Final force OFF
    gpio_pin_set_dt(&led, MY_LED_OFF); 
    
    send_data_to_python();
}

int main(void) {
    k_msleep(1000);
    if (!device_is_ready(dmic_dev) || !gpio_is_ready_dt(&button) || !gpio_is_ready_dt(&led)) return 0;
    
    gpio_pin_configure_dt(&button, GPIO_INPUT);
    
    // CRITICAL: Initialize as OUTPUT_HIGH so it starts in the OFF state
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_HIGH); 

    while (1) {
        // Double-check the LED stays OFF while idling
        gpio_pin_set_dt(&led, MY_LED_OFF);

        if (gpio_pin_get_dt(&button) > 0) {
            record_clip();
            // Debounce delay
            k_msleep(500); 
        }
        k_msleep(50);
    } 
}