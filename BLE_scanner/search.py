import asyncio
from bleak import BleakScanner
import datetime
import time

# The name we set in the C++ code
TARGET_NAME = "XIAO_SENSE"

# ── Label maps (0 = error/misclassified, 1–5 = valid classes) ──
KWS_LABELS = {
    0: "hello",
    1: "noise",
}
"double_tap", "flick", "lift", "rotate", "shake"
GESTURE_LABELS = {
    0: "UNKNOWN (misclassified)",
    1: "Double tap",
    2: "Flick",
    3: "Lift",
    4: "Rotate",
    5: "Shake",
}

# ── Ask the user which model is running ──
def select_model():
    print("Which model is running on the device?")
    print("  1) KWS (Keyword Spotting)")
    print("  2) Gesture Control")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            print("→ Using KWS labels\n")
            return KWS_LABELS, "KWS"
        elif choice == "2":
            print("→ Using Gesture Control labels\n")
            return GESTURE_LABELS, "Gesture"
        else:
            print("Invalid choice, please enter 1 or 2.")

labels, model_name = select_model()

def detection_callback(device, advertisement_data):
    # Check if the device name matches
    if device.name == TARGET_NAME:
        # Get the manufacturer data (where we hid the result)
        # 0x0590 is the Nordic ID we used in C++
        mfg_data = advertisement_data.manufacturer_data.get(0x0590)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if mfg_data:
            # The 3rd byte [2] is our inference result
            result_code = mfg_data[0]  # Note: index depends on how data is packed
            result_label = labels.get(result_code, f"UNMAPPED ({result_code})")

            log_entry = (
                f"[{timestamp}] [{model_name}] "
                f"Data: {mfg_data.hex()} | "
                f"Result Code: {result_code} | "
                f"Class: {result_label}"
            )
            print(log_entry)
            time.sleep(0.4)  # Small delay to ensure print output is flushed
            # Record it to a file
            with open("xiao_data_log.txt", "a") as f:
                f.write(log_entry + "\n")

async def main():
    print(f"Searching for {TARGET_NAME}... (Press Ctrl+C to stop)")
    scanner = BleakScanner(detection_callback)

    # Start scanning
    await scanner.start()

    # Keep the script running forever
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await scanner.stop()

if __name__ == "__main__":
    asyncio.run(main())