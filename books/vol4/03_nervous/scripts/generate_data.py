import os
import pandas as pd

def generate_fieldbus_data():
    """
    Generates historical performance data for fieldbus protocols.
    
    Data sources and rationale:
    - Modbus RTU (1979): Typical maximum baud rate is 115.2 kbps (0.115 Mbps). 
      Cycle time is estimated around 20.0 ms for a typical multi-node serial loop.
    - CAN (1986): Released by Bosch. Maximum bandwidth is 1.0 Mbps.
      Typical cycle time for basic multi-axis control is around 5.0 ms.
    - EtherCAT (2003): Introduced by Beckhoff. Operates over standard 100BASE-TX Ethernet (100.0 Mbps).
      Provides highly deterministic hardware-based synchronization, achieving cycle times around 33 microseconds (0.033 ms).
    - CAN-FD (2012): Extension of CAN allowing data phase up to 8.0 Mbps.
      Improved cycle time over standard CAN, around 1.16 ms.
    - TSN (GbE) (2018): Gigabit Ethernet (1000.0 Mbps) with Time-Sensitive Networking extensions.
      Allows for strict scheduling down to 10 microseconds (0.010 ms) cycle times.
    """
    
    data = [
        {
            "protocol": "Modbus RTU",
            "year": 1979,
            "bandwidth_mbps": 0.115,
            "cycle_time_ms": 20.0
        },
        {
            "protocol": "CAN",
            "year": 1986,
            "bandwidth_mbps": 1.0,
            "cycle_time_ms": 5.0
        },
        {
            "protocol": "EtherCAT",
            "year": 2003,
            "bandwidth_mbps": 100.0,
            "cycle_time_ms": 0.033
        },
        {
            "protocol": "CAN-FD",
            "year": 2012,
            "bandwidth_mbps": 8.0,
            "cycle_time_ms": 1.16
        },
        {
            "protocol": "TSN (GbE)",
            "year": 2018,
            "bandwidth_mbps": 1000.0,
            "cycle_time_ms": 0.010
        }
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure the target directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the dataframe to CSV
    csv_path = os.path.join(data_dir, 'fieldbus_evolution.csv')
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path}")

if __name__ == "__main__":
    generate_fieldbus_data()
