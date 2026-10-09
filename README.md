# TinyML on the Seeed Studio XIAO nRF54L15 Sense

Energy-efficient keyword spotting and gesture recognition running fully on-device on the Seeed Studio XIAO nRF54L15 Sense, using its onboard microphone and IMU. Built with the nRF Connect SDK (Zephyr) and TensorFlow Lite for Microcontrollers, and compared against Edge Impulse models.

This is the code for my University of Bristol final-year project, *Optimisation of Machine Learning Algorithms for Memory Constrained Edge Devices*. To my knowledge it is the first publicly documented TinyML deployment on the XIAO nRF54L15.

<!-- Add a system diagram or demo GIF here: ![Demo](docs/demo.gif) -->

## Projects

| Project | What it does |
|---|---|
| [Keyword spotting](Keyword_Spotting) | Detects the spoken word "Hello" from the onboard PDM microphone (1 second of 16 kHz audio) |
| [Gesture recognition](Gesture_recognition) | Classifies five gestures (lift, rotate, shake, double tap, flick) from the onboard 6-axis IMU (3 seconds at 50 Hz) |
| [BLE scanner](BLE_scanner) | Scans for the BLE advertisements the models broadcast |

Each project can run triggered by a button press, or continuously (keyword spotting listens with a sliding window; gesture recognition wakes on motion using the IMU's wake-up detection). The result is broadcast as a short non-connectable BLE advertisement.

## Models compared

For each project, three models were built and compared:

- **Keyword spotting:** raw audio 1D CNN, MFCC 2D CNN, and an Edge Impulse model
- **Gesture recognition:** baseline 1D CNN, deeper 1D CNN, and an Edge Impulse model

All manual models were trained in TensorFlow, quantised to INT8 and deployed with TensorFlow Lite for Microcontrollers.

## Results

### Keyword spotting

| Model | Accuracy | Inference | Energy per cycle | Total RAM | Total flash |
|---|---|---|---|---|---|
| Raw audio 1D CNN | 75% | 865 ms | 12.26 mJ | 155.9 kB | 147.1 kB |
| MFCC 2D CNN | 96% | 127 ms | 6.83 mJ | 116.2 kB | 262.2 kB |
| Edge Impulse | 98% | 65 ms | 6.36 mJ | 105.8 kB | 164.3 kB |

### Gesture recognition

| Model | Accuracy | Inference | Energy per cycle | Total RAM | Total flash |
|---|---|---|---|---|---|
| Baseline 1D CNN | 94% | 75 ms | 13.39 mJ | 63.2 kB | 140.8 kB |
| Deeper 1D CNN | 98% | 707 ms | 18.70 mJ | 83.2 kB | 155.9 kB |
| Edge Impulse | 100% | 24 ms | 12.98 mJ | 50.7 kB | 203.5 kB |

Energy was measured with a Nordic Power Profiler Kit 2 at 3.7 V.

### Estimated battery life (1,200 mAh LiPo, Edge Impulse models)

| Application | Duty cycle | Average current | Battery life |
|---|---|---|---|
| Keyword spotting | 15 s | 0.12 mA | about 416 days |
| Keyword spotting | 5 s | 0.35 mA | about 143 days |
| Gesture recognition | 15 s | 0.38 mA | about 131 days |
| Gesture recognition | 5 s | 0.77 mA | about 64 days |

Continuous keyword spotting draws 1.67 mA (about 30 days). Battery life was calculated from measured currents, not from a long-term discharge test.

## Key findings

- **Feature extraction matters most for audio.** Moving from raw audio to MFCC features raised accuracy from 75% to 96% and cut energy per cycle by 44%.
- **Data collection dominates gesture energy.** The 3-second IMU recording used most of each cycle's energy, so shrinking the model alone gives limited savings.
- **Removing the UART and `printk` logging saved about 6.5 mA.** The OLED expansion board was also removed for battery use.
- **RAM barely grows with model depth.** Adding layers increased flash and latency roughly linearly but RAM only slightly, because TensorFlow Lite Micro reuses its tensor arena.
- **Non-connectable BLE advertising is cheap:** about 0.16 mJ per 600 ms burst.
- **Solar:** with the PowerFilm kit and a 3.0 V regulated output, idle power was too high for indoor or partial sun. Full outdoor sun was only marginal.

## Hardware

- Seeed Studio XIAO nRF54L15 Sense (onboard PDM microphone and LSM6DS3TR-C IMU)
- 1,200 mAh LiPo battery on the rear battery pads (for power measurements)
- Nordic Power Profiler Kit 2 (for energy measurements)

## How it works

1. **Collect data:** firmware records sensor data and sends it over UART to a Python script (pyserial), which saves CSV files.
2. **Train:** TensorFlow (or Edge Impulse), then INT8 quantisation.
3. **Deploy:** the model is exported as a C header and run with TensorFlow Lite for Microcontrollers, or the Edge Impulse C++ library.

## Build and flash

Built with nRF Connect SDK v3.1.1 in VS Code.

TensorFlow Lite Micro in NCS v3.1.1 was not fully compatible with the XIAO nRF54L15, so three source files (`flatbuffer_conversions.cc`, `error_reporter.cc`, `flatbuffer_conversions_bridge.cc`) were added to the module, `CMakeLists.txt` was pointed at the modified module, and C++ was enabled in `prj.conf`. See [setup notes](docs/tflm_setup.md) for details.

```bash
git clone https://github.com/louiebisset/TinyML_nRF54L15.git
cd TinyML_nRF54L15/[project folder]
west build -b [board target]
west flash
```

## Limitations

- Keyword spotting used 200 samples from a single speaker; the gesture set had 250 samples from a single user
- Keyword spotting is a binary task ("Hello" vs noise)
- Battery life is estimated, not measured over a full discharge

## Future work

Multi-speaker datasets, multi-keyword classification, shorter gesture windows, full battery-discharge characterisation and other low-power microcontrollers.

## Report

[Full project report](docs/report.pdf)
