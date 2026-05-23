import serial
import time
import math

# --- Tunable Parameters ---
# TODO: CHANGE THIS! Find your Arduino's port in the Arduino IDE.
COM_PORT = 'COM16'  # <-- e.g., "COM3" (Windows) or "/dev/ttyACM0" (Linux/Mac)
BAUD_RATE = 9600
N_WAY = 2 # We are only using 2 sensors

# These are the "determined thresholds" you mentioned.
# You will need to experiment with these values!
# Start with these and see how it feels.
LEARNED_THRESHOLD = 0.765057  # (Deactivate when score > 50)
LEARNED_DURATION = 5     # (Deactivate for 10 seconds)
# --- End of Parameters ---





def calculate_transformed_aura(r1, r2):
    """
    Calculates the instantaneous AURA score for two readings.
    """
    # Normalize readings from 0-1023 (Arduino) to 0.0-1.0 (Algorithm)
    # Your LDR circuit (5V -> 10k -> A0 -> LDR -> GND) gives a 
    # HIGHER reading (e.g., ~800-900) in DARK
    # LOWER reading (e.g., ~100-300) in BRIGHT
    # We need to *invert* this for the AURA math, which assumes
    # higher value = more signal (like the IoT data).
    # We invert by (1023 - reading).
    
    r1_inverted = (1023.0 - r1) / 1023.0
    r2_inverted = (1023.0 - r2) / 1023.0
    
    readings = [r1_inverted, r2_inverted]
    total = sum(readings)
    
    # Avoid division by zero if both LDRs are completely dark
    if total < 1e-9:
        return 0.0 # No confidence update

    # AURA math
    numerator = sum(math.sin(math.pi * r / total)**2 for r in readings)
    denominator = 2.0  # From (2.0 * (math.sin(math.pi / 2.0) ** 2))
    aura_index = numerator / denominator
    
    # Transform from [0, 1] (correlation) to [-1, 1] (confidence update)
    transformed_aura = (2.0 * aura_index) - 1.0
    return transformed_aura


def run_experiment():
    print("--- AURA Live Feasibility Experiment ---")
    print(f"Connecting to Arduino on {COM_PORT}...")
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for serial connection to establish
        print("Connection successful.")
    except serial.SerialException as e:
        print(f"FATAL: Could not connect to {COM_PORT}.")
        print("Please check the following:")
        print(f" 1. Is the Arduino plugged in?")
        print(f" 2. Is '{COM_PORT}' the correct port? (Check Arduino IDE)")
        print(f" 3. Is the Serial Monitor in the Arduino IDE *closed*?")
        print(f"Error details: {e}")
        return

    # --- Real-time State Variables ---
    acc_aura = 0.0 
    sensor_states = [True, True] # [Sensor 0, Sensor 1] (True=ON, False=OFF)
    deactivation_end_times = [0.0, 0.0] 

    print("Experiment running. Press Ctrl+C to stop.")
    print("\n--- GOAL ---")
    print("Shine BRIGHT, EVEN light on both LDRs to increase AURA score.")
    print("Block one LDR or make light uneven to decrease AURA score.")
    print("--------------------------------------------------")

    try:
        while True:
            current_time = time.time()
            
            # --- 1. Check for Reactivation ---
            for i in range(2):
                if not sensor_states[i] and current_time > deactivation_end_times[i]:
                    print(f"\n*** SENSOR {i} (LED {i}) REACTIVATED (Duration Over) ***")
                    sensor_states[i] = True
                    ser.write(f"L{i}_ON\n".encode('utf-8'))
                    acc_aura = 0.0 # Reset confidence
                    print("--- AURA score reset to 0 ---")

            # --- 2. Request and Read Data from Arduino ---
            ser.write(b"R\n") # Send "Read" command
            
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue 
                r0, r1 = map(int, line.split(','))
            except Exception as e:
                print(f"Warning: Could not parse serial data: '{line}'. Error: {e}")
                time.sleep(0.5)
                continue

            # --- 3. Run AURA Logic ---
            
            # We only update confidence if BOTH sensors are currently active
            if all(sensor_states):
                transformed_aura = calculate_transformed_aura(r0, r1)
                acc_aura += transformed_aura
                acc_aura = max(-100.0, min(100.0, acc_aura)) # Clamp score
                
                # Display raw readings and the score
                print(f"Readings (Raw): [S0: {r0}, S1: {r1}] | Acc. AURA: {acc_aura:.2f}")

                # Check for deactivation
                if acc_aura > LEARNED_THRESHOLD:
                    sensor_to_deactivate = 0 # Always deactivate Sensor 0
                    
                    print(f"\n!!! CONFIDENCE THRESHOLD REACHED ({acc_aura:.2f}) !!!")
                    print(f"--- DEACTIVATING SENSOR {sensor_to_deactivate} (LED {sensor_to_deactivate}) for {LEARNED_DURATION}s ---")
                    
                    sensor_states[sensor_to_deactivate] = False
                    deactivation_end_times[sensor_to_deactivate] = current_time + LEARNED_DURATION
                    ser.write(f"L{sensor_to_deactivate}_OFF\n".encode('utf-8'))
            else:
                # One sensor is OFF.
                active_sensor = 1 
                deactivated_sensor = 0
                
                readings = [r0, r1]
                true_val = 1023.0 - readings[deactivated_sensor] # Inverted value
                est_val = 1023.0 - readings[active_sensor]     # Inverted value
                error = (true_val - est_val)**2
                
                print(f"DEACTIVATED (S{deactivated_sensor} OFF). True: {true_val:.0f}, Est: {est_val:.0f} | Sq. Error: {error:.0f}")

            # --- 4. Loop Delay ---
            time.sleep(0.2) # Read 5 times per second

    except KeyboardInterrupt:
        print("\nExperiment stopped by user.")
    finally:
        # Cleanup: Turn LEDs back on and close serial port
        if ser and ser.is_open:
            print("Cleaning up... turning LEDs back on.")
            ser.write(b"L0_ON\n")
            ser.write(b"L1_ON\n")
            ser.close()
            print("Serial port closed.")

if __name__ == "__main__":
    run_experiment()