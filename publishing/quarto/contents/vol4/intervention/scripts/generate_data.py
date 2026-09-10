import pandas as pd
import numpy as np

# Data parameters based on the system dynamics described in the chapter:
# - Reaction Time (T_react): 1.0 s
# - Handshake Protocol Latency (T_handshake): 0.020 s
# - Bumpless Blend Window (tau_blend): 0.075 s
# - Maximum Deceleration (a_max): 6.0 m/s^2

def generate_intervention_stopping_distance():
    # 100 velocity points between 5.0 m/s (18 km/h) and 35.0 m/s (126 km/h)
    velocity_mps = np.linspace(5.0, 35.0, 100)
    
    # Constants
    T_react = 1.0
    T_handshake = 0.020
    tau_blend = 0.075
    a_max = 6.0
    
    # Calculations
    velocity_kmph = velocity_mps * 3.6
    
    reaction_drift = velocity_mps * T_react
    handshake_drift = velocity_mps * T_handshake
    
    # Blend displacement over C^2 quintic spline
    # d_blend = v_0 * tau_blend - 0.15 * a_max * tau_blend^2
    blend_drift = (velocity_mps * tau_blend) - (0.15 * a_max * (tau_blend ** 2))
    
    # Post-blend velocity and braking distance
    # v_post_blend = v_0 - 0.5 * a_max * tau_blend
    v_post_blend = velocity_mps - (0.5 * a_max * tau_blend)
    braking_distance = (v_post_blend ** 2) / (2 * a_max)
    
    total_distance = reaction_drift + handshake_drift + blend_drift + braking_distance
    
    df = pd.DataFrame({
        'Velocity_mps': velocity_mps,
        'Velocity_kmph': velocity_kmph,
        'Reaction_Drift_m': reaction_drift,
        'Handshake_Drift_m': handshake_drift,
        'Blend_Drift_m': blend_drift,
        'Braking_Distance_m': braking_distance,
        'Total_Distance_m': total_distance
    })
    
    df.to_csv('../data/intervention_stopping_distance.csv', index=False)
    print("Successfully generated ../data/intervention_stopping_distance.csv")

if __name__ == "__main__":
    generate_intervention_stopping_distance()
