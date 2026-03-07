"""
Tutorial: Comparing Concrete Hardware
====================================
Models the Smart Doorbell across different real-world microcontrollers.
"""

import mlsysim

def main():
    scenario = mlsysim.Applications.Doorbell
    
    devices = [
        mlsysim.Hardware.Tiny.ArduinoNano33,
        mlsysim.Hardware.Tiny.ESP32_S3
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
