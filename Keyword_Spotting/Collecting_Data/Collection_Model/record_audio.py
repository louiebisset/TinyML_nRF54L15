import serial
import wave
import struct
import os
import re
import csv
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM11'   # Make sure this matches your port
BAUD_RATE = 1000000     # Must match your C++ code
# ---------------------
def save_wav(filename, data):
    """Saves a list of integers as a 16-bit PCM WAV file."""
    try:
        with wave.open(filename, 'w') as f:
            f.setnchannels(1)       # Mono
            f.setsampwidth(2)       # 16-bit
            f.setframerate(16000)   # 16kHz
            binary_data = struct.pack('<' + 'h'*len(data), *data)
            f.writeframes(binary_data)
        return True
    except Exception as e:
        print(f"Error saving WAV: {e}")
        return False
def save_csv(filename, data):
    """Saves a list of integers as a CSV file with a header."""
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["AudioData"]) # Header for ML compatibility
            for val in data:
                writer.writerow([val])
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def main():
    print("--- WAV & CSV MULTI-RECORDER ---")
    
    label = input("Enter the label (e.g., 'hello', 'noise'): ").strip()
    if not label: label = "default"
    
    save_folder = f"dataset/{label}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"  Created folder: {save_folder}")

    print(f"\nLISTENING on {SERIAL_PORT} @ {BAUD_RATE} baud...")
    print(" Press button to record. (Ctrl+C to exit)\n")

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            audio_data = []
            recording = False
            # Count existing sets of files
            existing_wavs = [n for n in os.listdir(save_folder) if n.endswith('.wav')]
            sample_count = len(existing_wavs) + 1
            while True:
                if ser.in_waiting > 0:
                    try:
                        raw_line = ser.readline().decode('utf-8', errors='replace').strip()
                    except:
                        continue
                    if "---START_DATA---" in raw_line:
                        print(f" Rec #{sample_count:03d}...", end="", flush=True)
                        recording = True
                        audio_data = []
                        continue
                    if "---END_DATA---" in raw_line:
                        if recording:
                            print(" Done.")
                            # 1. Define filenames
                            base_name = f"{save_folder}/{label}_{sample_count:03d}"
                            wav_file = f"{base_name}.wav"
                            csv_file = f"{base_name}.csv"
                            
                            # 2. Save both formats
                            if save_wav(wav_file, audio_data):
                                print(f" Saved WAV")
                            if save_csv(csv_file, audio_data):
                                print(f" Saved CSV ({len(audio_data)} samples)")
                            
                            sample_count += 1
                            recording = False
                        continue

                    if recording and raw_line:
                        try:
                            # Robust conversion logic
                            val = int(raw_line)
                            audio_data.append(val)
                        except ValueError:
                            numbers = re.findall(r'-?\d+', raw_line)
                            if numbers:
                                audio_data.append(int(numbers[-1]))

    except serial.SerialException:
        print(f"\n Error: Could not open {SERIAL_PORT}.")
    except KeyboardInterrupt:
        print("\n Stopping...")

if __name__ == "__main__":
    main()