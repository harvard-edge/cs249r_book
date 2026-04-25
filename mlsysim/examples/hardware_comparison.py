"""
Tutorial: Comparing Concrete Hardware
====================================
Models the Smart Doorbell across different real-world microcontrollers.
"""

import mlsysim

def main():
    scenario = mlsysim.Applications.Doorbell
    
    devices = [
        mlsysim.Hardware.Tiny.nRF52840,
        mlsysim.Hardware.Tiny.ESP32_S3,
        mlsysim.Hardware.Tiny.HimaxWE1,
    ]
    
    for hw in devices:
        print("\n--- Evaluating " + hw.name + " ---")
        test_scenario = scenario.model_copy(update={"system": hw})
        
        try:
            result = test_scenario.evaluate()
            print(result.scorecard())
        except Exception as e:
            print("CRITICAL SYSTEM FAILURE: " + str(e))

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.1):
# --- Evaluating Nordic nRF52840 (Cortex-M4F) ---
# +============================================================+
# | MLSys-im SYSTEM EVALUATION
# | Scenario: Smart Doorbell
# +============================================================+
# | Level 1: Feasibility [PASS]
# |   Model fits in memory (0.5 MB / 1.0 MB)
# +------------------------------------------------------------+
# | Level 2: Performance [FAIL]
# |   Latency: 781.76 millisecond (Target: 200 ms)
# +------------------------------------------------------------+
# | Level 3: Macro/Economics [PASS]
# |   Annual Carbon: 0.1 kg | TCO: $11,500
# +============================================================+
#
# --- Evaluating ESP32-S3 (AI) ---
# +============================================================+
# | Level 1: Feasibility [PASS]
# |   Model fits in memory (0.5 MB / 8.0 MB)
# +------------------------------------------------------------+
# | Level 2: Performance [PASS]
# |   Latency: 101.01 millisecond (Target: 200 ms)
# +------------------------------------------------------------+
# | Level 3: Macro/Economics [PASS]
# |   Annual Carbon: 1.7 kg | TCO: $11,500
# +============================================================+
#
# --- Evaluating Himax WE-I Plus ---
# +============================================================+
# | Level 1: Feasibility [PASS]
# |   Model fits in memory (0.5 MB / 2.0 MB)
# +------------------------------------------------------------+
# | Level 2: Performance [FAIL]
# |   Latency: 252.01 millisecond (Target: 200 ms)
# +------------------------------------------------------------+
# | Level 3: Macro/Economics [PASS]
# |   Annual Carbon: 0.0 kg | TCO: $11,500
# +============================================================+
