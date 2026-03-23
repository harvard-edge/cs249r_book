# The Deployed Device

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <b>🔬 TinyML</b>
</div>

---

*How you update firmware and keep it alive for years*

FOTA updates, connectivity, monitoring, security, and long-term reliability — operating ML on devices that must run unattended for years.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/03_deployed_device.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---


### Deployment & Updates


#### 🟢 L1/L2


<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Power Drain</b> · <code>duty-cycle-power</code></summary>

- **Interviewer:** "You're designing a battery-powered audio sensor using a Cortex-M4 microcontroller. The device is active for 2 seconds to perform an inference, consuming 50 mW. It then goes into a deep sleep mode for 8 seconds, consuming 10 µW. What is the average power consumption of the device over this 10-second cycle?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often mistake peak power for average power, or incorrectly average the power states without weighting by time. Another common error is to ignore sleep power, which, while small, can be significant for devices with very low duty cycles.

  **Realistic Solution:** The average power is the total energy consumed divided by the total time period. The device is active for 20% of the time (2s out of 10s). The total energy is the sum of energy used in the active and sleep states. This results in an average power consumption of just over 10 mW.

  > **Napkin Math:** Energy_active = P_active × t_active = 50 mW × 2s = 100 mJ
Energy_sleep = P_sleep × t_sleep = 10 µW × 8s = 0.01 mW × 8s = 0.08 mJ
Total Energy = 100 mJ + 0.08 mJ = 100.08 mJ
Average Power = Total Energy / Total Time = 100.08 mJ / 10s = 10.008 mW ≈ 10 mW

  > **Key Equation:** $$ P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}} $$

  > **Options:**
  > [ ] ~25 mW
  > [ ] 50 mW
  > [x] ~10 mW
  > [ ] ~0.1 mW

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FOTA Flash Tax</b> · <code>fota-flash-budget</code></summary>

- **Interviewer:** "You're deploying a keyword-spotting model to a fleet of microcontrollers, each with 1MB of flash memory. To enable remote updates, you must implement a Firmware-Over-The-Air (FOTA) update strategy. When partitioning the flash, which of these components typically consumes the largest portion of the memory?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers new to the embedded world often misjudge the memory budget. They focus on the size of their ML model and application code, underestimating the significant 'tax' imposed by a robust FOTA mechanism. In a common A/B scheme, you must reserve space for an entire second copy of the firmware, which dwarfs the memory footprint of the bootloader or the RTOS.

  **Realistic Solution:** The OTA (or 'inactive') partition for the next firmware image. To perform a safe, atomic update, you can't overwrite the currently running code. Instead, you download the new firmware to a separate partition. Once the download is complete and verified, the bootloader is instructed to boot from the new partition on the next reset. This A/B partitioning scheme implies that you must set aside roughly 40-50% of your total flash just to hold the next update.

  > **Napkin Math:** With 1MB (~1024 KB) of flash:
- A minimal bootloader: ~32 KB.
- A compact RTOS: ~64 KB.
- Total for system software: `32 + 64 = 96 KB`.
- Remaining space for the application and OTA: `1024 KB - 96 KB = 928 KB`.
- For an A/B scheme, this is split in two: `928 KB / 2 = 464 KB`.
- Thus, the OTA partition alone is ~464 KB, far larger than the bootloader or RTOS.

  > **Options:**
  > [ ] The Bootloader
  > [ ] The Real-Time Operating System (RTOS)
  > [x] The OTA Update Partition
  > [ ] The model's activation tensors (Tensor Arena)

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/03_deployed_device.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Sensor's Average Power</b> · <code>duty-cycle-power</code></summary>

- **Interviewer:** "A wildlife audio sensor uses a Cortex-M4 microcontroller to listen for a specific animal call. It runs a model for 1 second, consuming 10 mW (active power), and then enters a deep sleep state for 9 seconds, consuming 10 µW (sleep power). Explain to your colleague how to calculate the average power consumption over this 10-second cycle."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the average power by either completely ignoring the sleep power consumption or by incorrectly averaging the two power states without considering the time spent in each. The former leads to an underestimation, while the latter grossly overestimates the average power because it doesn't account for the fact that the device spends most of its time in the low-power state.

  **Realistic Solution:** The correct way to calculate the average power is to compute a time-weighted average. You calculate the total energy consumed during one full cycle (active + sleep) and then divide by the total period of the cycle. This accounts for the significant energy savings from the long sleep interval.

  > **Napkin Math:** 1.  **Calculate Active Energy:**
    $E_{\text{active}} = P_{\text{active}} \times t_{\text{active}} = 10\ \text{mW} \times 1\ \text{s} = 10\ \text{mJ}$
2.  **Calculate Sleep Energy:**
    $E_{\text{sleep}} = P_{\text{sleep}} \times t_{\text{sleep}} = 10\ \mu\text{W} \times 9\ \text{s} = 0.01\ \text{mW} \times 9\ \text{s} = 0.09\ \text{mJ}$
3.  **Calculate Total Energy per Cycle:**
    $E_{\text{total}} = E_{\text{active}} + E_{\text{sleep}} = 10\ \text{mJ} + 0.09\ \text{mJ} = 10.09\ \text{mJ}$
4.  **Calculate Average Power:**
    $P_{\text{avg}} = E_{\text{total}} / t_{\text{period}} = 10.09\ \text{mJ} / 10\ \text{s} = 1.009\ \text{mW}$

  > **Key Equation:** $P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] 10 mW. The active power dominates so much that sleep is negligible.
  > [ ] 1.0 mW. This comes from ignoring the sleep power contribution.
  > [ ] 5.005 mW. A simple average of the active and sleep power values.
  > [x] 1.009 mW. A time-weighted average of active and sleep power.

  📖 **Deep Dive:** [TinyML Microcontroller Architectures](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Coin Cell Battery Lifetime</b> · <code>battery-life-estimation</code></summary>

- **Interviewer:** "A remote bird-call detector has a duty cycle where it's active for 500ms (drawing 50mW) and then sleeps for 4.5s (drawing 10µW). It's powered by a standard 3V, 225 mAh coin cell battery. Can this device realistically run for a full week in the field? Calculate its expected lifetime in days."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to calculate the lifetime using only the active power consumption (50 mW), which leads to a massive underestimation of battery life. Another error is confusing power (mW) with energy (mWh) or failing to properly convert the battery's mAh capacity into mWh by multiplying by the voltage.

  **Realistic Solution:** To find the lifetime, you must first calculate the true average power consumption based on the duty cycle. Then, calculate the total energy capacity of the battery in milliamp-hours (mWh). Finally, divide the total energy capacity by the average power consumption to get the total operational hours, which can then be converted to days.

  > **Napkin Math:** 1.  **Calculate Average Power:**
    $t_{\text{period}} = 0.5\text{s} + 4.5\text{s} = 5\text{s}$
    $P_{\text{avg}} = \frac{(50\ \text{mW} \times 0.5\ \text{s}) + (0.01\ \text{mW} \times 4.5\ \text{s})}{5\ \text{s}}$
    $P_{\text{avg}} = \frac{25\ \text{mJ} + 0.045\ \text{mJ}}{5\ \text{s}} = \frac{25.045\ \text{mJ}}{5\ \text{s}} = 5.009\ \text{mW}$
2.  **Calculate Battery Energy:**
    $E_{\text{battery}} = \text{Capacity (mAh)} \times \text{Voltage (V)} = 225\ \text{mAh} \times 3\ \text{V} = 675\ \text{mWh}$
3.  **Calculate Lifetime in Hours:**
    $\text{Lifetime}_{\text{hours}} = \frac{E_{\text{battery}}}{P_{\text{avg}}} = \frac{675\ \text{mWh}}{5.009\ \text{mW}} \approx 134.75\ \text{hours}$
4.  **Calculate Lifetime in Days:**
    $\text{Lifetime}_{\text{days}} = \frac{134.75\ \text{hours}}{24\ \text{hours/day}} \approx 5.6\ \text{days}$

No, it cannot reliably run for a full week (7 days).

  > **Key Equation:** $\text{Lifetime} = \frac{\text{Battery Capacity (Wh)}}{\text{Average Power (W)}}$

  > **Options:**
  > [ ] ~0.56 days. (Calculated using only the active power draw)
  > [ ] ~1.1 days. (Calculated by incorrectly averaging power states)
  > [x] ~5.6 days. (Calculated using the time-weighted average power)
  > [ ] ~45 days. (Calculated by ignoring battery voltage and dividing mAh by mW)

  📖 **Deep Dive:** [TinyML Microcontroller Architectures](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Power Tax</b> · <code>duty-cycling</code></summary>

- **Interviewer:** "A wildlife tracking device using a Cortex-M4 microcontroller wakes up for 1 second to perform an inference, consuming 50 mW. It then returns to a deep sleep state for the next 119 seconds, consuming 10 µW. State the approximate average power consumption of the device over this 2-minute cycle."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often neglect the power consumed during the 'sleep' phase, assuming it's zero. While small, sleep current can dominate the energy budget in very low duty cycle applications. Another common error is mixing units (milliwatts and microwatts) or incorrectly calculating the total period for the average.

  **Realistic Solution:** The average power is the total energy consumed during the cycle divided by the total cycle time. The device is active for 1 second and sleeps for 119 seconds, for a total period of 120 seconds. The average power is therefore slightly above 0.4 mW.

  > **Napkin Math:** Total Period = 1s (active) + 119s (sleep) = 120s
Energy_active = 50 mW * 1s = 50 mJ
Energy_sleep = 10 µW * 119s = 0.01 mW * 119s = 1.19 mJ
Total Energy = 50 mJ + 1.19 mJ = 51.19 mJ
Average Power = Total Energy / Total Period = 51.19 mJ / 120s ≈ 0.426 mW

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} t_{\text{active}} + P_{\text{sleep}} t_{\text{sleep}}}{t_{\text{active}} + t_{\text{sleep}}}$

  > **Options:**
  > [ ] 50 mW
  > [x] ~0.42 mW
  > [ ] 10 µW
  > [ ] ~0.41 mW

  📖 **Deep Dive:** [Scaling Rules - TinyML](NUMBERS.md#tinyml)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Camera's Power Budget</b> · <code>tinyml-duty-cycle-average-power</code></summary>

- **Interviewer:** "You are designing a wildlife camera that uses a Cortex-M4 MCU for motion detection. The system consumes 50 mW when actively processing an image for 500ms. In deep sleep, it consumes 10 µW. If the camera is triggered on average once every 5 minutes, what is its average power consumption? This will determine the required battery size for a 1-year deployment."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate a simple arithmetic mean of the active and sleep power, completely ignoring the time spent in each state. The correct approach is a time-weighted average, where the minuscule sleep power consumption dominates the total energy budget because the device spends over 99% of its time asleep.

  **Realistic Solution:** The average power is the total energy consumed during one cycle divided by the cycle's duration. Here, a cycle is 5 minutes (300 seconds). The device is active for 0.5s and sleeps for 299.5s. The average power is therefore heavily weighted towards the sleep consumption, as the brief, high-power active periods are amortized over long, low-power sleep periods.

  > **Napkin Math:** 1. **Define the period:** A full cycle is 5 minutes, which is $5 \times 60 = 300$ seconds.
2. **Define active/sleep times:** $t_{\text{active}} = 0.5$ s. Therefore, $t_{\text{sleep}} = 300 - 0.5 = 299.5$ s.
3. **Calculate energy per state:**
   - $E_{\text{active}} = P_{\text{active}} \times t_{\text{active}} = 50 \text{ mW} \times 0.5 \text{ s} = 25 \text{ mJ}$
   - $E_{\text{sleep}} = P_{\text{sleep}} \times t_{\text{sleep}} = 10 \text{ µW} \times 299.5 \text{ s} = 0.01 \text{ mW} \times 299.5 \text{ s} \approx 2.995 \text{ mJ}$
4. **Calculate total energy per period:** $E_{\text{total}} = E_{\text{active}} + E_{\text{sleep}} = 25 + 2.995 = 27.995 \text{ mJ}$
5. **Calculate average power:** $P_{\text{avg}} = E_{\text{total}} / t_{\text{period}} = 27.995 \text{ mJ} / 300 \text{ s} \approx 0.093 \text{ mW}$

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} t_{\text{active}} + P_{\text{sleep}} t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~0.083 mW
  > [ ] ~25.0 mW
  > [ ] ~10.1 mW
  > [x] ~0.093 mW

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Environmental Sensor's Duty Cycle</b> · <code>tinyml-duty-cycle-lifetime</code></summary>

- **Interviewer:** "You are designing a battery-powered environmental sensor with a target battery life of 1 year. The device is powered by a 2400 mAh, 3.7V battery. When taking a measurement and transmitting data, it consumes 150 mW. In its deep sleep state, it consumes 10 µW. To meet the 1-year lifetime goal, what is the maximum total time, in seconds, that the device can afford to be in its active state each hour?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common error is to miscalculate the total energy available in the battery (e.g., by ignoring the voltage and incorrectly using Amp-hours as a unit of energy). Another mistake is to ignore the energy consumed during the sleep state, which, although small per second, accumulates significantly over a year and reduces the budget available for active operations.

  **Realistic Solution:** First, calculate the total energy in the battery in Watt-hours by multiplying capacity (in Amp-hours) by voltage. Second, calculate the average power budget in milliwatts by dividing the total energy by the target lifetime in hours. Finally, use the average power equation, plugging in the known power values and the 1-hour period (3600s), to solve for the unknown active time ($t_{\text{active}}$).

  > **Napkin Math:** 1. **Calculate total battery energy:** $E_{\text{battery}} = 2400 \text{ mAh} \times 3.7 \text{ V} = 2.4 \text{ Ah} \times 3.7 \text{ V} = 8.88 \text{ Wh}$.
2. **Calculate target lifetime in hours:** $1 \text{ year} = 365 \times 24 = 8760 \text{ hours}$.
3. **Calculate average power budget:** $P_{\text{budget}} = E_{\text{battery}} / \text{Lifetime} = 8.88 \text{ Wh} / 8760 \text{ h} \approx 0.001014 \text{ W} \approx 1.014 \text{ mW}$.
4. **Set up the duty cycle equation for one hour (3600s):** Let $t_a$ be the active time in seconds.
   $P_{\text{budget}} = \frac{P_{\text{active}} t_a + P_{\text{sleep}} (3600 - t_a)}{3600}$.
5. **Solve for $t_a$:**
   $1.014 = \frac{150 t_a + 0.01 (3600 - t_a)}{3600}$
   $1.014 \times 3600 = 150 t_a + 36 - 0.01 t_a$
   $3650.4 \approx 149.99 t_a + 36$
   $3614.4 \approx 149.99 t_a$
   $t_a \approx 3614.4 / 149.99 \approx 24.1$ seconds.

  > **Key Equation:** $t_{\text{active}} = \frac{P_{\text{avg}} t_{\text{period}} - P_{\text{sleep}} t_{\text{period}}}{P_{\text{active}} - P_{\text{sleep}}}$

  > **Options:**
  > [ ] ~6.3 seconds
  > [x] ~24.1 seconds
  > [ ] ~24.3 seconds
  > [ ] The budget is impossible; it's below sleep power.

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Power Drain</b> · <code>duty-cycling-power</code></summary>

- **Interviewer:** "You are designing a battery-powered sensor with a Cortex-M4 microcontroller. To save energy, you use a duty cycle where the device is active for 1 second and then enters a deep sleep mode for 9 seconds. If the active power is 10 mW and the deep sleep power is 10 µW, what is the approximate average power consumption over the full 10-second period?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget how dominant the active power is, even for a short duration, and incorrectly average the two power numbers (e.g., (10mW + 10µW)/2) without considering the time spent in each state. This ignores the core principle of a duty cycle and leads to a massive overestimation of power drain and underestimation of battery life.

  **Realistic Solution:** The average power is the time-weighted average of the active and sleep power consumption. The device is active 10% of the time (1s out of 10s) and in deep sleep 90% of the time (9s out of 10s). The active phase dominates the calculation, while the sleep phase's contribution is almost negligible.

  > **Napkin Math:** Total Period = 1s (active) + 9s (sleep) = 10s
Active Energy = 10 mW × 1s = 10 mJ
Sleep Energy = 10 µW × 9s = 90 µJ = 0.09 mJ
Total Energy = 10 mJ + 0.09 mJ = 10.09 mJ
Average Power = Total Energy / Total Period = 10.09 mJ / 10s ≈ 1 mW

  > **Key Equation:** $P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~5 mW
  > [ ] 10 mW
  > [x] ~1 mW
  > [ ] ~100 µW

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Forever-Camera Battery Life</b> · <code>duty-cycling-battery-drain</code></summary>

- **Interviewer:** "You are designing a wildlife camera powered by a 2400 mAh, 3.7V Li-ion battery. The device uses a Cortex-M4 MCU. It wakes up for 1 second to perform inference and then goes into deep sleep for 99 seconds. Explain how you would calculate the expected battery life in days. Use the provided hardware constants."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to average the power states (Active, Sleep) without weighting them by time. For example, calculating `(50mW + 10µW) / 2`. This completely ignores that the device spends 99% of its time in the low-power state, leading to a massive underestimation of battery life.

  **Realistic Solution:** The correct approach is to calculate the time-weighted average power consumption. The device is active 1% of the time and asleep 99% of the time. This average power is then divided into the total energy capacity of the battery (Voltage × Amp-hours) to find the total operational time.

  > **Napkin Math:** 1. **Calculate Total Battery Energy:** `Energy (mWh) = 2400 mAh × 3.7V = 8880 mWh`
2. **Calculate Energy per Cycle (100s):** `E_cycle = (P_active × t_active) + (P_sleep × t_sleep)`
   - `E_cycle = (50 mW × 1s) + (10 µW × 99s)`
   - `E_cycle = 50 mWs + 990 µWs = 50 mWs + 0.99 mWs = 50.99 mWs`
3. **Calculate Average Power:** `P_avg = E_cycle / t_cycle = 50.99 mWs / 100s = 0.5099 mW`
4. **Calculate Battery Life (Hours):** `Life (h) = Total Energy (mWh) / P_avg (mW) = 8880 mWh / 0.5099 mW ≈ 17,415 hours`
5. **Convert to Days:** `17,415 hours / 24 hours/day ≈ 725 days`

  > **Key Equation:** P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}

  > **Options:**
  > [ ] ~15 days (Incorrectly averaging power states)
  > [ ] ~739 days (Ignoring sleep power contribution)
  > [x] ~725 days
  > [ ] ~196 days (Using battery mAh directly as mWh)

  📖 **Deep Dive:** [TinyML Scaling Rules](https://github.com/ml-prefect/staffml-book-public/blob/main/interviews/NUMBERS.md#4-scaling-rules-arithmetic--hardware-independent)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Solar-Powered Sensor's Energy Budget</b> · <code>energy-harvesting-battery-drain</code></summary>

- **Interviewer:** "A TinyML vibration sensor for industrial machinery is active for 500ms every 10 seconds. It's powered by a small solar cell that generates 0.2 mW in the ambient factory lighting. The sensor's MCU consumes 50 mW when active and 10 µW in sleep mode. Contrast the energy consumed per hour with the energy generated per hour to determine if the device has an energy surplus or deficit."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often compare the peak active power (50 mW) directly to the solar generation rate (0.2 mW). While the conclusion (energy deficit) is the same in this case, the reasoning is flawed. This approach doesn't account for the duty cycle. A device can have a very high peak power draw but a very low average power consumption if it spends most of its time asleep. The correct comparison is always between *average* power consumed and *average* power generated.

  **Realistic Solution:** To determine the energy balance, you must compare the average energy consumed per hour to the energy generated per hour. First, calculate the device's duty cycle to find the total time spent in active and sleep states. Use this to calculate the total energy consumed in Watt-seconds. Then, calculate the energy generated by the solar cell over the same period and compare the two values.

  > **Napkin Math:** 1. **Calculate Duty Cycle:**
   - `Active time % = 0.5s / 10s = 5%`
   - `Sleep time % = 9.5s / 10s = 95%`
2. **Calculate Energy Consumed per Hour (E_consumed):**
   - `t_active_per_hour = 3600s × 0.05 = 180s`
   - `t_sleep_per_hour = 3600s × 0.95 = 3420s`
   - `E_consumed = (50 mW × 180s) + (10 µW × 3420s)`
   - `E_consumed = 9000 mWs + 34,200 µWs = 9000 mWs + 34.2 mWs = 9034.2 mWs`
3. **Calculate Energy Generated per Hour (E_generated):**
   - `E_generated = 0.2 mW × 3600s = 720 mWs`
4. **Compare:** `9034.2 mWs (consumed) > 720 mWs (generated)`. The device has a significant energy deficit of 8314.2 mWs each hour.
5. **Average Power View:** `P_avg_consumed = 9034.2 mWs / 3600s ≈ 2.51 mW`. `P_avg_generated = 0.2 mW`. `2.51 mW > 0.2 mW` confirms the deficit.

  > **Key Equation:** E_{\text{balance}} = (P_{\text{generated}} \cdot t) - (P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}})

  > **Options:**
  > [ ] Energy surplus; average power is low
  > [x] Energy deficit; consumes ~2.5 mW average, generates 0.2 mW
  > [ ] Energy deficit; peak power of 50 mW is greater than 0.2 mW
  > [ ] Energy neutral; sleep power makes up for the active draw

  📖 **Deep Dive:** [TinyML Scaling Rules](https://github.com/ml-prefect/staffml-book-public/blob/main/interviews/NUMBERS.md#4-scaling-rules-arithmetic--hardware-independent)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The OTA Flash Memory Tax</b> · <code>ota-memory-footprint</code></summary>

- **Interviewer:** "You are scoping the flash memory requirements for a new microcontroller that will run a keyword spotting model. The device must support robust Over-the-Air (OTA) firmware updates using an A/B partitioning scheme to prevent bricking during an update. If the total flash on the chip is 1MB, what is the approximate memory cost you must pay for this reliability feature? In other words, how much flash must be reserved specifically for the OTA update partition?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers unfamiliar with embedded constraints often assume they can overwrite the application 'in-place', forgetting that a power failure or failed write would permanently brick the device. Others assume only a small 'patch' or 'diff' is sent, underestimating the need to store a full, second application binary for a safe swap.

  **Realistic Solution:** A robust A/B OTA update scheme requires two separate memory partitions of equal size: one for the currently running application ('A') and one to receive the new application ('B'). After the new firmware is fully downloaded and verified in partition 'B', the bootloader will swap to it on the next boot. This means approximately 50% of the total available flash must be reserved for the update partition. It's a direct 'tax' on memory in exchange for update reliability.

  > **Napkin Math:** Total Flash Memory: 1MB (1024 KB).
A small bootloader might occupy ~32 KB.
Remaining Space: 1024 KB - 32 KB = 992 KB.
This remaining space must be split into two equal partitions for the A/B scheme.
Size of Partition A (and B) = 992 KB / 2 = 496 KB.
Therefore, the cost of the OTA feature is the ~496 KB reserved for Partition B, which the main application cannot use.

  > **Options:**
  > [ ] ~50 KB, for storing a small patch file.
  > [ ] Effectively 0 KB, as you can overwrite the existing application in-place.
  > [x] ~500 KB, to hold a complete second copy of the application binary.
  > [ ] ~32 KB, the space taken by the bootloader itself.

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The OTA Flash Budget</b> · <code>ota-flash-budget</code></summary>

- **Interviewer:** "Interviewer: You've deployed a fleet of 10,000 environmental sensors based on a Cortex-M4 microcontroller. Each device has 1MB of Flash. The memory is partitioned for a bootloader (32KB), an RTOS (64KB), and the current application binary which is 450KB. For safe Over-the-Air (OTA) updates, you've reserved a partition of the same size as the application. A new model improves accuracy but increases the application binary size to 454KB. Explain whether this OTA update is possible across the fleet."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to only calculate the total *remaining* space on the device (1024KB - 32KB - 64KB - 450KB = 478KB) and incorrectly conclude there is plenty of room. This fails to account for the fact that OTA updates require a dedicated, pre-allocated partition to download the new image before it can be verified and swapped. You cannot use fragmented free space for a monolithic binary download.

  **Realistic Solution:** The update is not possible. The system was designed with an OTA partition exactly matching the original application size: 450KB. The new 454KB binary is too large to fit into this pre-allocated slot. The device cannot dynamically resize the OTA partition in the field; this requires a physical re-flash. The fleet is effectively bricked from receiving this update until a smaller binary can be produced or the devices are manually recalled.

  > **Napkin Math:** 1.  **Total Flash:** 1MB = 1024 KB
2.  **Fixed Overhead:** 32 KB (Bootloader) + 64 KB (RTOS) = 96 KB
3.  **Application A size:** 450 KB
4.  **OTA (Application B) Partition size:** 450 KB (sized for the original app)
5.  **Total Allocated Space:** 96 KB (Overhead) + 450 KB (App A) + 450 KB (OTA Slot) = 996 KB
6.  **Remaining Free Space:** 1024 KB - 996 KB = 28 KB
7.  **Conclusion:** The new 454 KB binary is larger than the 450 KB OTA slot. The update will fail.

  > **Key Equation:** $\text{Flash}_{\text{Required}} > \text{Flash}_{\text{OTA Partition}}$

  > **Options:**
  > [ ] Yes, it works. The total free space is 478KB, which is more than enough for the 454KB update.
  > [x] No, it fails. The 454KB new binary is larger than the 450KB reserved OTA partition.
  > [ ] Yes, it works. The total application space is 928KB (1024KB - 96KB), which can fit two 454KB apps.
  > [ ] No, it fails. The device only has 28KB of total free space left.

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Power Drain</b> · <code>duty-cycle-power</code></summary>

- **Interviewer:** "You are designing a battery-powered audio sensor that uses a Cortex-M4 microcontroller to detect a wake-word. It wakes up for 1 second to listen and run an inference, then goes into deep sleep for 9 seconds to conserve energy. What is the approximate *average* power consumption of this device?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget the massive, orders-of-magnitude difference between active and sleep power. A common error is to only consider the active power (~10 mW), ignoring the fact that the device spends 90% of its time in a low-power state. Another is to incorrectly average the power states without weighting by time.

  **Realistic Solution:** The average power is the time-weighted average of the active and sleep power states. The device is active 10% of the time (1s / 10s) and asleep 90% of the time (9s / 10s).

- **Active Power (Cortex-M4):** ~10 mW
- **Sleep Power (Deep Sleep):** ~10 µW (or 0.01 mW)

Using the duty cycle formula:
`P_avg = (P_active * t_active + P_sleep * t_sleep) / t_period`
`P_avg = (10 mW * 1s + 0.01 mW * 9s) / 10s`
`P_avg = (10 + 0.09) mW / 10`
`P_avg ≈ 1.01 mW`

The sleep power is about 1000x smaller than the active power, so its contribution to the average is almost negligible.

  > **Napkin Math:** The device is active 10% of the time (1s out of 10s). The average power will be roughly 10% of its active power. Active power is ~10 mW, so the average is ~1 mW. The sleep power is 1000x smaller, so it's basically a rounding error.

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{(\text{P}_{\text{active}} \cdot \text{t}_{\text{active}}) + (\text{P}_{\text{sleep}} \cdot \text{t}_{\text{sleep}})}{\text{t}_{\text{period}}}$

  > **Options:**
  > [ ] ~10 mW
  > [ ] ~5 mW
  > [x] ~1 mW
  > [ ] ~10 µW

  📖 **Deep Dive:** [TinyML Microcontrollers](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Battery Life Budget</b> · <code>duty-cycling-battery-drain</code></summary>

- **Interviewer:** "You're designing a wildlife audio monitor using a Cortex-M4 microcontroller. It wakes up to run an inference for 1 second, consuming 10mW, then goes into a deep sleep mode for 9 seconds, consuming 10µW. Explain the concept of duty cycling and calculate the approximate average power consumption of the device."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often just average the active and sleep power values, e.g., `(10mW + 10µW) / 2`, completely ignoring the *time* spent in each state. This is arithmetically simple but conceptually wrong, as it doesn't account for the 90% of the time the device is in the ultra-low-power sleep state, leading to a massive overestimation of the power budget.

  **Realistic Solution:** Duty cycling is a core power-saving technique in embedded systems. Instead of running continuously, the device operates in a cycle, spending the vast majority of its time in a low-power 'sleep' state and only waking 'actively' for brief periods to perform tasks. The average power consumption is the time-weighted average of the active and sleep power states over one full cycle.

Here, the total cycle is 1s + 9s = 10s. The device is active 10% of the time and asleep 90% of the time. This allows a device with a relatively high active power draw to achieve a very low average power draw, enabling long battery life.

  > **Napkin Math:** 1. **Unify Units:** Convert all power values to milliwatts (mW). Active Power = 10mW. Sleep Power = 10µW = 0.01mW.
2. **Define Period:** The total cycle period is `t_active + t_sleep = 1s + 9s = 10s`.
3. **Calculate Energy per Period:** Calculate the total energy (in milliwatt-seconds) consumed in one cycle.
   `Energy = (P_active × t_active) + (P_sleep × t_sleep)`
   `Energy = (10mW × 1s) + (0.01mW × 9s) = 10 mWs + 0.09 mWs = 10.09 mWs`
4. **Calculate Average Power:** Divide the total energy by the period duration.
   `P_avg = Energy / Period = 10.09 mWs / 10s ≈ 1.01 mW`

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~5.0 mW
  > [ ] ~10.0 mW
  > [x] ~1.0 mW
  > [ ] ~9.0 mW

  📖 **Deep Dive:** [TinyML Microcontrollers](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Energy Cost of an OTA Update</b> · <code>ota-updates</code></summary>

- **Interviewer:** "A battery-powered environmental sensor needs to receive a 450 KB firmware update over-the-air (OTA). The device uses a low-power microcontroller and a Bluetooth Low Energy (BLE) radio. State which phase of the OTA process typically consumes the most energy from the battery."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Focusing on the peak power consumption of the CPU (during verification) rather than the total energy (Power × Time). While CPU-intensive tasks have high peak power, the long duration of the radio transmission usually dominates the total energy budget.

  **Realistic Solution:** Downloading the firmware image. Radio communication is one of the most energy-intensive operations on a constrained device, not because of its peak power draw, but because of the long time it must remain active to transfer a large amount of data. Even with a low-power protocol like BLE, transferring hundreds of kilobytes can take tens of seconds, far longer than any on-chip computation or memory operation.

  > **Napkin Math:** ### P.I.C.O. (Parameters, Invariants, Calculation, Outputs)

**Parameters:**
- Firmware Size: 450 KB = 3.6 million bits
- BLE Throughput: ~100 kbps (a realistic rate in a noisy environment)
- Radio + Idle CPU Power: ~15 mW
- CPU Active Power (Flash Write/Verify): ~40 mW (from 'TinyML Active' range)
- Flash Write Time: ~1 second (for 450 KB)
- Verification Time: < 1 second

**Invariant:**
- Energy is the integral of power over time ($E = P \times t$). Total energy, not peak power, drains the battery.

**Calculation:**
1.  **Download Energy:**
    -   Time = 3,600,000 bits / 100,000 bps = 36 seconds
    -   Energy = 36 s × 15 mW = **540 mJ**
2.  **Flash Write Energy:**
    -   Time ≈ 1 s
    -   Energy = 1 s × 40 mW = **40 mJ**
3.  **Verification Energy:**
    -   Time ≈ 0.5 s
    -   Energy = 0.5 s × 40 mW = **20 mJ**

**Output:** The download phase (~540 mJ) consumes over 10x more energy than the flash write (~40 mJ) or verification (~20 mJ) phases because the radio must stay on for a significant duration.

  > **Key Equation:** $$ E = P \times t $$

  > **Options:**
  > [ ] Writing the 450 KB image to the internal flash memory
  > [x] Downloading the 450 KB image via the BLE radio
  > [ ] Verifying the cryptographic signature of the downloaded image
  > [ ] Rebooting the device to activate the new firmware

  📖 **Deep Dive:** [TinyML: Microcontrollers](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The OTA Flash Budget</b> · <code>ota-updates</code></summary>

- **Interviewer:** "You are deploying a keyword spotting model to a fleet of battery-powered devices. Each device has a microcontroller with 1 MB of total flash memory. For safe Over-the-Air (OTA) updates, the system uses an A/B partitioning scheme, where one partition holds the active firmware while the other receives the update, allowing for a rollback if the update fails. The current firmware includes a 32 KB bootloader and a 64 KB Real-Time Operating System (RTOS). Calculate the absolute maximum size for a new model that can be safely deployed via an OTA update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget that for a robust OTA update with rollback capability, the flash memory must be partitioned. This effectively halves the available space for the application (model + OS). A common error is to calculate the available space as `Total Flash - OS Size`, failing to account for the second partition needed for the incoming update.

  **Realistic Solution:** With an A/B partitioning scheme, the 1 MB (1024 KB) of flash is split into two 512 KB partitions. Each partition must be able to hold a complete, bootable image, which includes the OS components and the model. Therefore, the maximum model size is limited by the space remaining in a single partition after the bootloader and RTOS are accounted for.

  > **Napkin Math:** 1. Total Flash: 1 MB = 1024 KB
2. Partition Size for A/B OTA: 1024 KB / 2 = 512 KB
3. OS & Bootloader Footprint: 32 KB + 64 KB = 96 KB
4. Maximum Model Size: 512 KB (Partition Size) - 96 KB (OS Footprint) = 416 KB

  > **Key Equation:** $\text{Max Model Size} = (\frac{\text{Total Flash}}{2}) - (\text{Bootloader Size} + \text{RTOS Size})$

  > **Options:**
  > [ ] 928 KB
  > [ ] 512 KB
  > [x] 416 KB
  > [ ] 480 KB

  📖 **Deep Dive:** [Deployed Device](https://mlsysbook.ai/tinyml/03_deployed_device.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Deep Sleep Power Chasm</b> · <code>duty-cycling-fundamentals</code></summary>

- **Interviewer:** "To maximize battery life for a device that is idle most of the time, you use duty cycling. What is the approximate ratio of power consumption for a typical microcontroller in an active state versus a deep sleep state?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers from a cloud or mobile background often underestimate this ratio, thinking it's around 10-100×. They don't realize that a microcontroller's deep sleep state is not just 'idle' but a state where nearly all clocks and peripherals are powered off, leading to a much more dramatic power drop.

  **Realistic Solution:** The ratio is approximately 10,000:1. A typical Cortex-M4 microcontroller consumes around 10 mW while active, but its power consumption can drop to as low as 1 µW in deep sleep. This enormous difference is the fundamental physical invariant that makes duty cycling the most critical power-saving technique in the TinyML playbook.

  > **Napkin Math:** Active Power (Cortex-M4): ~10 mW
Sleep Power (Deep Sleep): ~1 µW
Ratio = Active Power / Sleep Power = 10,000 µW / 1 µW = 10,000×

  > **Key Equation:** $\bar{P} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~100×
  > [ ] ~1,000×
  > [x] ~10,000×
  > [ ] ~1,000,000×

  📖 **Deep Dive:** [TinyML Hardware](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Remote Wildlife Camera's Lifespan</b> · <code>duty-cycling-battery-drain</code></summary>

- **Interviewer:** "You are designing a remote wildlife camera that uses a Cortex-M4 microcontroller for bird detection. It runs inference for 200ms, 120 times per hour. For the rest of the time, it's in deep sleep. The device is powered by a 4,000 mAh, 3.7V battery. Given the Cortex-M4's active power consumption is 50mW and its deep sleep power is 10µW, approximately how long will the battery last?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate or completely ignore the power consumption during sleep mode. While small, over long periods (months or years), the cumulative energy drain from sleep is significant and can't be treated as zero. Another common error is to only consider the active power drain, leading to a wild underestimation of device lifetime.

  **Realistic Solution:** The correct way to solve this is to calculate the *average* power consumption by creating a weighted average of the active and sleep power based on the duty cycle. This average power is then used to determine the total lifetime based on the battery's energy capacity.

  > **Napkin Math:** 1. **Calculate total active time per hour:** 120 inferences/hour × 0.2 seconds/inference = 24 seconds.
2. **Calculate the duty cycle:** The fraction of time the device is active is 24 seconds / 3600 seconds = 0.00667 (or 0.667%).
3. **Calculate average power:**
`P_avg = (P_active × Duty Cycle) + (P_sleep × (1 - Duty Cycle))`
`P_avg = (50mW × 0.00667) + (10µW × (1 - 0.00667))`
`P_avg = 0.3335mW + (0.01mW × 0.99333) ≈ 0.3335mW + 0.0099mW ≈ 0.3434mW`
4. **Calculate total battery energy:** 4,000 mAh × 3.7V = 14,800 mWh = 14.8 Wh.
5. **Calculate lifetime in hours:** 14,800 mWh / 0.3434mW ≈ 43,100 hours.
6. **Convert to days:** 43,100 hours / 24 hours/day ≈ 1,795 days.

  > **Key Equation:** P_{\text{avg}} = (P_{\text{active}} \times \text{Duty Cycle}) + (P_{\text{sleep}} \times (1 - \text{Duty Cycle}))

  > **Options:**
  > [ ] ~12 days
  > [ ] ~60 days
  > [x] ~1,800 days
  > [ ] ~5,500 days

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Deception</b> · <code>duty-cycling-power</code></summary>

- **Interviewer:** "You're designing a battery-powered keyword spotting device using a Cortex-M4 microcontroller. It has a 10% duty cycle: it's active for 1 second to run an inference, then sleeps for 9 seconds. Based on the typical power numbers for this class of device, which component contributes the most to the average power consumption over one cycle?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers accustomed to systems with less extreme active/sleep power ratios often assume that because the device is asleep 90% of the time, the sleep current must be the dominant factor in the long-term energy budget. They forget that the active power can be orders of magnitude higher, making even brief moments of activity the primary energy drain.

  **Realistic Solution:** The active power consumption during the 1-second 'on' window completely dominates the average power draw. The active power is around 10mW, while the deep sleep power is around 10µW—a 1000x difference. Even though the device is only active for 10% of the time, that period accounts for over 99% of the energy consumed.

  > **Napkin Math:** Using the standard TinyML power formula:

1.  **Parameters:**
    *   $P_{\text{active}} \approx 10\text{ mW}$ (From constants: Cortex-M4 Active)
    *   $P_{\text{sleep}} \approx 10\text{ µW}$ (From constants: Deep Sleep)
    *   $t_{\text{active}} = 1\text{ s}$
    *   $t_{\text{sleep}} = 9\text{ s}$

2.  **Calculate Energy per Phase:**
    *   $E_{\text{active}} = P_{\text{active}} \times t_{\text{active}} = 10\text{ mW} \times 1\text{ s} = 10,000\text{ µJ}$
    *   $E_{\text{sleep}} = P_{\text{sleep}} \times t_{\text{sleep}} = 10\text{ µW} \times 9\text{ s} = 90\text{ µJ}$

3.  **Compare:**
    *   The active energy (10,000 µJ) is more than 100x greater than the sleep energy (90 µJ). Therefore, the active phase is the dominant consumer.

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} t_{\text{active}} + P_{\text{sleep}} t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] The sleep power, because the device is in that state 90% of the time.
  > [x] The active power during the 1-second inference window.
  > [ ] They contribute roughly equally to the average power.
  > [ ] The energy needed to transition from sleep to active state (wake-up energy).

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Thermal Duty Cycle</b> · <code>thermal-throttling-duty-cycling</code></summary>

- **Interviewer:** "You're designing a wildlife camera that uses a Cortex-M4 based chip for animal detection. Your thermal tests show the enclosure can only dissipate 15 mW of average power before the chip overheats and throttles. The chip consumes 50 mW when running inference and a negligible amount in sleep mode. To prevent overheating, what is the maximum duty cycle (the percentage of time the chip can be active) you can sustain?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often invert the ratio, calculating `50 / 15` and getting a nonsensical number, or they subtract the powers (`50 - 15 = 35%`) which has no physical meaning. Another common mistake is to fail to connect that 'average power' is directly controlled by the duty cycle.

  **Realistic Solution:** The solution is to calculate the ratio of the power budget to the active power consumption. Since the sleep power is negligible, the average power is simply the active power multiplied by the duty cycle. To stay within the 15 mW budget, the duty cycle must be limited.

  > **Napkin Math:** Average Power = Active Power × Duty Cycle
15 mW = 50 mW × Duty Cycle
Duty Cycle = 15 mW / 50 mW = 0.3
Therefore, the chip can be active for a maximum of 30% of the time.

  > **Key Equation:** $\text{Duty Cycle} = \frac{P_{\text{avg\_budget}}}{P_{\text{active}}}$

  > **Options:**
  > [ ] 333%
  > [ ] 35%
  > [x] 30%
  > [ ] 15%

  📖 **Deep Dive:** [Microcontrollers](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Solar-Powered Sensor</b> · <code>battery-drain-energy-harvesting</code></summary>

- **Interviewer:** "You are designing a remote environmental sensor. When active (sensing and transmitting), it consumes 50 mW. In deep sleep, it consumes 10 µW. The device wakes for 1 second every minute. Your small solar panel provides an average of 2 mW of power over a 24-hour cycle. Calculate the sensor's average power consumption and explain if the system is energy-positive and therefore sustainable."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is unit confusion: failing to convert micro-watts (µW) to milli-watts (mW) correctly, leading to an answer that is off by orders of magnitude (e.g., adding 10 to 50 instead of 0.01). Another mistake is to ignore the sleep power entirely, which, while small, is necessary for a precise answer.

  **Realistic Solution:** To determine sustainability, we must calculate the average power consumed over a full cycle (one minute) and compare it to the average power generated by the solar panel. The average power is the weighted sum of the active power and sleep power.

  > **Napkin Math:** 1. **Define the cycle:** Period = 1 minute = 60s. Active time = 1s. Sleep time = 59s.
2. **Convert units:** Sleep Power = 10 µW = 0.01 mW.
3. **Calculate total energy per cycle:**
   Energy = (P_active × t_active) + (P_sleep × t_sleep)
   Energy = (50 mW × 1s) + (0.01 mW × 59s) = 50 mWs + 0.59 mWs = 50.59 mWs
4. **Calculate average power:**
   P_avg = Energy / Period = 50.59 mWs / 60s ≈ 0.843 mW
5. **Compare:** The solar panel generates 2 mW on average, which is greater than the 0.843 mW consumed. The system is energy-positive with a surplus of ~1.16 mW, so it is sustainable.

  > **Key Equation:** $P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] Not sustainable. Consumes ~10.67 mW, generates 2 mW.
  > [x] Sustainable. Consumes ~0.84 mW, generates 2 mW.
  > [ ] Sustainable. Consumes ~0.83 mW, generates 2 mW.
  > [ ] Not sustainable. Consumes 50 mW, generates 2 mW.

  📖 **Deep Dive:** [Sensing Pipeline](https://mlsysbook.ai/tinyml/02_sensing_pipeline.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The 'Forever' Sensor's Battery Life</b> · <code>duty-cycling-battery-drain</code></summary>

- **Interviewer:** "You're designing a battery-powered wildlife tracking sensor using a Cortex-M4 MCU. The device wakes up to perform a 1-second inference, then goes back to deep sleep. This cycle repeats every 10 minutes (600 seconds). The MCU consumes 50 mW while active and 10 µW while in deep sleep. The device is powered by a 600 mAh, 3V coin cell battery. Calculate the approximate operational lifetime of the device in days."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often miscalculate battery life by either ignoring the energy consumed during the long sleep periods or by ignoring the duty cycle altogether and calculating based on the active power draw, leading to wildly inaccurate estimates. Another common error is mixing units (e.g., Wh and mW) without conversion.

  **Realistic Solution:** The correct approach is to calculate the *average* power consumption over a full cycle, then use that to determine the battery lifetime. The key is to account for both the high-power active state and the low-power sleep state, weighted by their duration.

1.  **Calculate Energy per Cycle:**
    -   Active Energy: 50 mW × 1 s = 50 mJ
    -   Sleep Energy: 10 µW × 599 s = 0.01 mW × 599 s ≈ 5.99 mJ
    -   Total Energy per Cycle: 50 mJ + 5.99 mJ = 55.99 mJ
2.  **Calculate Average Power:**
    -   `P_avg = Total Energy / Total Time = 55.99 mJ / 600 s ≈ 0.0933 mW`
3.  **Calculate Battery Capacity in Watt-hours (Wh):**
    -   `Capacity (Wh) = (600 mAh × 3 V) / 1000 = 1.8 Wh`
4.  **Calculate Lifetime:**
    -   First, convert `P_avg` to Watts: `0.0933 mW = 0.0000933 W`
    -   `Lifetime (hours) = 1.8 Wh / 0.0000933 W ≈ 19,292 hours`
    -   `Lifetime (days) = 19,292 hours / 24 hours/day ≈ 803 days`

  > **Napkin Math:** P_avg = (P_active * t_active + P_sleep * t_sleep) / t_period
P_avg = (50mW * 1s + 0.01mW * 599s) / 600s ≈ 0.0933 mW

Battery Energy = 600mAh * 3V = 1800 mWh
Lifetime = 1800 mWh / 0.0933 mW ≈ 19,292 hours
Lifetime_days = 19,292 / 24 ≈ 803 days

  > **Key Equation:** P_{\text{avg}} = \frac{P_{\text{active}} t_{\text{active}} + P_{\text{sleep}} t_{\text{sleep}}}{t_{\text{period}}}

  > **Options:**
  > [ ] ~36 hours
  > [x] ~803 days
  > [ ] ~900 days
  > [ ] ~19 hours

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Duty-Cycling Wildlife Monitor</b> · <code>duty-cycling-energy</code></summary>

- **Interviewer:** "You are designing a wildlife monitoring device that uses a Cortex-M4. It wakes up for 1 second to run an image classification model, then goes into deep sleep for 59 seconds to conserve battery. According to the `NUMBERS.md` table, the M4 consumes about 10 mW while active and 10 µW in deep sleep. To select the right battery and estimate field longevity, you need to calculate the device's average power consumption. What is it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the power consumed during the sleep state, assuming it's zero. While sleep power is tiny, the device spends the vast majority of its time (~98% in this case) in this state, so its contribution to the total energy budget is not negligible. Another critical error is failing to convert microwatts (µW) to milliwatts (mW) correctly, leading to an answer that's off by orders of magnitude.

  **Realistic Solution:** The average power is the total energy consumed during one full cycle (active + sleep) divided by the cycle's duration. First, ensure all units are consistent (e.g., milliwatts). The sleep power is 10 µW, which is 0.01 mW. Then, calculate the energy for each phase and divide by the total period to find the average power, which dictates battery life.

  > **Napkin Math:** P_active = 10 mW
t_active = 1 s
P_sleep = 10 µW = 0.01 mW  (Key unit conversion!)
t_sleep = 59 s
t_period = 1s + 59s = 60 s

Energy_active = 10 mW * 1 s = 10.0 mJ
Energy_sleep = 0.01 mW * 59 s = 0.59 mJ
Energy_total = 10.0 mJ + 0.59 mJ = 10.59 mJ

Power_avg = Energy_total / t_period = 10.59 mJ / 60 s ≈ 0.177 mW

  > **Key Equation:** $\overline{P} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] 0.167 mW
  > [ ] 10 mW
  > [x] 0.177 mW
  > [ ] 10.59 mW

  📖 **Deep Dive:** [TinyML: The Microcontroller](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Duty Cycle Power Drain</b> · <code>tinyml-duty-cycle</code></summary>

- **Interviewer:** "You're designing a wildlife camera powered by a small battery. It uses a Cortex-M4 microcontroller. The device stays in a deep sleep state, consuming 10 µW. A motion sensor wakes it up for 1 second to run an image classification inference, during which it consumes 50 mW. This cycle repeats every 10 seconds (1 second active, 9 seconds sleep). What is the average power consumption of the device? Explain the formula you used."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A frequent mistake is to incorrectly average the power values (e.g., (50mW + 10µW)/2), forgetting that they must be weighted by time. Another error is unit confusion, mixing up milliwatts (mW) and microwatts (µW), which differ by a factor of 1000. Finally, engineers sometimes calculate the active-phase energy but forget to average it over the entire period, giving an answer of 50mJ instead of an average power in mW.

  **Realistic Solution:** The correct way to calculate the average power is to find the total energy consumed in one full cycle and then divide by the cycle's duration. The total energy is the sum of the energy used during the active phase and the energy used during the sleep phase.

1.  **Energy_active** = 50 mW × 1 s = 50 mJ
2.  **Energy_sleep** = 10 µW × 9 s = 90 µJ = 0.09 mJ
3.  **Energy_total** = 50 mJ + 0.09 mJ = 50.09 mJ
4.  **Power_average** = 50.09 mJ / 10 s = 5.009 mW

Notice that the energy consumed during sleep is three orders of magnitude smaller than the active energy, making its contribution almost negligible in this specific case. However, it is critical to account for it, as in systems with very long sleep times, the sleep power can dominate the energy budget.

  > **Napkin Math:** P_avg = (P_active * t_active + P_sleep * t_sleep) / t_period
P_avg = ((50 mW * 1 s) + (10 µW * 9 s)) / 10 s
P_avg = (50 mJ + 90 µJ) / 10 s
P_avg = (50.09 mJ) / 10 s
P_avg ≈ 5.0 mW

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{\text{P}_{\text{active}} \times t_{\text{active}} + \text{P}_{\text{sleep}} \times t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] 25.0 mW
  > [ ] 5.9 mW
  > [x] ~5.0 mW
  > [ ] 50.0 mW

  📖 **Deep Dive:** [TinyML Hardware](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Camera's Energy Budget</b> · <code>duty-cycling-power</code></summary>

- **Interviewer:** "You're designing a battery-powered wildlife camera using a Cortex-M4 microcontroller. The device wakes up for 1 second to perform inference, then goes into deep sleep for the remaining 9 seconds of its 10-second cycle. From the playbook, you know the Cortex-M4 consumes about 10 mW while active and 10 µW while in deep sleep. Explain how you would calculate the average power consumption for this device."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to only consider the active power (10 mW), completely ignoring the massive power savings from the long deep sleep period. Another error is to do a simple average of the active and sleep power numbers, ignoring that the device spends 90% of its time in the low-power state.

  **Realistic Solution:** The correct approach is to calculate the time-weighted average of the power consumption. The device is active for 10% of the time and sleeping for 90%. Therefore, the average power is the sum of the energy consumed in each state, divided by the total time period. The sleep power is so low that it's almost negligible in the final calculation, demonstrating the effectiveness of duty cycling.

  > **Napkin Math:** 1. **Define Parameters:**
   - Active Power (P_active): 10 mW
   - Sleep Power (P_sleep): 10 µW = 0.01 mW
   - Active Time (t_active): 1 s
   - Sleep Time (t_sleep): 9 s
   - Total Period (t_period): 10 s

2. **Calculate Energy per Phase:**
   - Active Energy = 10 mW * 1 s = 10 mJ
   - Sleep Energy = 0.01 mW * 9 s = 0.09 mJ

3. **Calculate Average Power:**
   - Total Energy = 10 mJ + 0.09 mJ = 10.09 mJ
   - Average Power = Total Energy / Total Period = 10.09 mJ / 10 s = 1.009 mW

  > **Key Equation:** $$P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$$

  > **Options:**
  > [ ] 10 mW (Misconception: Ignores the power savings from the sleep cycle).
  > [ ] 5.005 mW (Misconception: Simple average of active and sleep power, ignoring time weighting).
  > [x] ~1.01 mW
  > [ ] 0.1 mW (Misconception: Arithmetical error, likely dividing by 100 instead of 10).

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Birdwatcher's Battery Budget</b> · <code>tinyml-duty-cycle-battery</code></summary>

- **Interviewer:** "You're designing a battery-powered acoustic sensor to detect a rare bird species. The device uses a Cortex-M4 microcontroller that wakes for 0.5 seconds every minute to run an inference, then goes into deep sleep. The MCU consumes 50 mW while active and 10 µW in deep sleep. If the device is powered by a 150 mAh, 3.7V battery, explain how you would calculate its approximate battery life in days."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to ignore the power consumed during deep sleep. While small, over long periods (millions of sleep cycles), this energy consumption becomes significant and can lead to an overestimation of battery life. A second common error is unit mismatch—failing to convert µW to mW, or treating battery capacity in mAh as a direct measure of energy without accounting for voltage.

  **Realistic Solution:** The correct approach is to first calculate the battery's total energy capacity in milli-Watt-hours (mWh). Then, calculate the *average* power consumption across a full active/sleep cycle. Finally, divide the total energy capacity by the average power consumption to determine the total device lifetime.

  > **Napkin Math:** 1. **Calculate Battery Energy:** `Energy (mWh) = Capacity (mAh) × Voltage (V)`
   `150 mAh × 3.7V = 555 mWh`

2. **Calculate Average Power:** The cycle is 60s (0.5s active, 59.5s sleep). Note: `10 µW = 0.010 mW`
   `P_avg = (P_active × t_active + P_sleep × t_sleep) / t_period`
   `P_avg = (50 mW × 0.5s + 0.010 mW × 59.5s) / 60s`
   `P_avg = (25 mWs + 0.595 mWs) / 60s ≈ 0.427 mW`

3. **Calculate Lifetime (Hours):** `Lifetime = Total Energy / Average Power`
   `555 mWh / 0.427 mW ≈ 1300 hours`

4. **Calculate Lifetime (Days):** `1300 hours / 24 hours/day ≈ 54 days`

  > **Key Equation:** $$P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}$$

  > **Options:**
  > [ ] ~11 hours
  > [ ] ~2.2 days
  > [x] ~54 days
  > [ ] ~55.5 days

  📖 **Deep Dive:** [TinyML](tinyml/README.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Camera's Battery Budget</b> · <code>duty-cycling-and-power</code></summary>

- **Interviewer:** "You're designing a remote wildlife camera using a Cortex-M4 MCU. To conserve power, the device operates on a duty cycle: it wakes up for 1 second to capture and classify an image, consuming 50 mW. It then enters a deep sleep state for 59 seconds, consuming only 10 µW. The entire system is powered by a 400 mAh, 3.7V LiPo battery. Explain how you would calculate the operational lifetime of the device and what that lifetime is."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistakes are: 1) Forgetting to convert the battery's mAh capacity into Watt-hours (or milliWatt-hours) by multiplying by the voltage, leading to unit mismatch. 2) Only considering the active power consumption and ignoring the duty cycle, which dramatically underestimates the device's lifetime. 3) Incorrectly averaging the active and sleep power values without weighting by the time spent in each state.

  **Realistic Solution:** First, calculate the average power consumption over one full cycle (60 seconds) by time-weighting the active and sleep power states. Second, calculate the total energy stored in the battery by multiplying its capacity in Amp-hours by its voltage to get Watt-hours. Finally, divide the battery's total energy by the system's average power consumption to find the operational lifetime. The device can operate for approximately 73 days.

  > **Napkin Math:** 1. **Calculate Average Power (P_avg):**
   - Energy per active period: `E_active = 50 mW * 1 s = 50 mJ`
   - Energy per sleep period: `E_sleep = 10 µW * 59 s = 0.01 mW * 59 s = 0.59 mJ`
   - Total energy per 60s cycle: `E_total = 50 mJ + 0.59 mJ = 50.59 mJ`
   - Average power: `P_avg = E_total / 60 s = 50.59 mJ / 60 s ≈ 0.843 mW`

2. **Calculate Battery Energy (E_batt):**
   - Energy (mWh): `E_batt = 400 mAh * 3.7 V = 1480 mWh`

3. **Calculate Lifetime:**
   - Lifetime (hours): `Time = E_batt / P_avg = 1480 mWh / 0.843 mW ≈ 1756 hours`
   - Lifetime (days): `1756 hours / 24 hours/day ≈ 73.1 days`

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] About 1.2 days
  > [ ] About 20 days
  > [x] About 73 days
  > [ ] About 2.5 days

  📖 **Deep Dive:** [Numbers Every ML Systems Engineer Should Know](https://github.com/ml-sys/foundation-models-book/blob/main/playbook/NUMBERS.md#tinyml)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Keyword Spotter's Battery Budget</b> · <code>duty-cycling-energy</code></summary>

- **Interviewer:** "You are designing a battery-powered keyword-spotting device using a Cortex-M4 microcontroller. The device wakes up to analyze a 0.5-second audio clip for a keyword, a process that consumes 40 mW. This check happens once every 5 seconds. When not active, the device is in a deep sleep mode, consuming just 5 µW.

Given this duty cycle, explain how you would calculate the average power consumption. If the device is powered by a standard 240 mAh, 3V coin cell battery, how long can you expect it to last?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to calculate battery life based only on the active power consumption, completely ignoring the long periods of deep sleep. This leads to a wildly pessimistic estimate. Another error is to incorrectly average the power values without weighting them by time, or to confuse power (mW) with energy (mWh) and battery capacity (mAh).

  **Realistic Solution:** The correct approach is to calculate the time-weighted average power across one full cycle. The device is active for 0.5s and sleeps for the remaining 4.5s of its 5-second period. The average power is dominated by the active phase but amortized over the full period.

Once the average power is known, we calculate the total energy stored in the battery (Voltage × Amp-hours) and divide that by the average power to find the operational lifetime.

  > **Napkin Math:** 1. **Calculate time durations for one period:**
   - `t_period` = 5 s
   - `t_active` = 0.5 s
   - `t_sleep` = `t_period` - `t_active` = 5s - 0.5s = 4.5 s

2. **Calculate average power consumption:**
   - `P_sleep` = 5 µW = 0.005 mW
   - `Energy_active` = 40 mW × 0.5 s = 20 mJ
   - `Energy_sleep` = 0.005 mW × 4.5 s = 0.0225 mJ
   - `P_avg` = (`Energy_active` + `Energy_sleep`) / `t_period` = (20 + 0.0225) mJ / 5 s ≈ 4.005 mW

3. **Calculate total battery energy:**
   - `Battery_Energy` = 3 V × 240 mAh = 720 mWh

4. **Calculate device lifetime:**
   - `Lifetime` = `Battery_Energy` / `P_avg` = 720 mWh / 4.005 mW ≈ 180 hours
   - `Lifetime_days` = 180 hours / 24 hours/day = 7.5 days

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~18 hours
  > [ ] ~36 hours
  > [x] ~7.5 days
  > [ ] ~2.5 days

  📖 **Deep Dive:** [TinyML Deployment & Power](https://mlsysbook.ai/tinyml/03_deployed_device)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Battery-Sipping Wearable</b> · <code>duty-cycling</code></summary>

- **Interviewer:** "You're designing a wearable activity tracker powered by a Cortex-M4. To maximize battery life, it operates on a duty cycle: it's active for 2 seconds while processing sensor data, then enters a deep sleep mode for 8 seconds. This 10-second cycle repeats continuously.

Given the hardware specs, the active power consumption is 50 mW and the deep sleep power consumption is 100 µW.

Interpret this scenario and calculate the device's average power consumption."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the power consumed during the sleep state because it's so small compared to the active state. This leads to an underestimation of the total energy drain. Another error is to take a simple, unweighted average of the active and sleep power, which ignores that the device spends significantly more time sleeping than active.

  **Realistic Solution:** The correct way to calculate the average power consumption is to find the time-weighted average over one full cycle. You calculate the total energy consumed during the cycle (active energy + sleep energy) and then divide by the total duration of the cycle.

- Active Energy: 50 mW × 2 s = 100 mJ
- Sleep Energy: 100 µW (or 0.1 mW) × 8 s = 0.8 mJ
- Total Energy: 100 mJ + 0.8 mJ = 100.8 mJ
- Average Power: 100.8 mJ / 10 s = 10.08 mW

The average power consumption is approximately 10.1 mW.

  > **Napkin Math:** P_avg = ( (P_active × t_active) + (P_sleep × t_sleep) ) / t_period
P_avg = ( (50mW × 2s) + (0.1mW × 8s) ) / 10s
P_avg = ( 100mJ + 0.8mJ ) / 10s
P_avg = 100.8mJ / 10s
P_avg = 10.08mW

  > **Key Equation:** $$P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}}$$

  > **Options:**
  > [ ] 10.0 mW
  > [ ] 25.05 mW
  > [x] 10.08 mW
  > [ ] 50.0 mW

  📖 **Deep Dive:** [TinyML Systems](tinyml/README.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Birdwatcher's Power Budget</b> · <code>duty-cycle-energy</code></summary>

- **Interviewer:** "You're designing a remote acoustic sensor to detect an endangered bird species. It uses a Cortex-M4 microcontroller that wakes up for 1 second to run an audio classification model, then sleeps for the remaining 59 seconds of a 1-minute cycle.

Using the hardware constants from the playbook, calculate the *average* power consumption of the device. Explain your reasoning."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to confuse peak power with average power. Engineers often state the active power (10 mW) as the answer, forgetting that for battery life calculations, it's the average power consumption over a full cycle that matters. Another error is to ignore the sleep power contribution, which, while small, is necessary for a precise calculation.

  **Realistic Solution:** To find the average power, you must calculate the total energy consumed during one full cycle (active + sleep) and then divide that by the total cycle time. This correctly weights the high-power active state and the low-power sleep state.

First, make the units consistent: Active power is 10 mW, and Sleep power is 10 µW, which is 0.01 mW.

The average power is the time-weighted average of the active and sleep power states.

  > **Napkin Math:** 1. **Parameters**:
   - `P_active`: 10 mW (Cortex-M4 Active Power)
   - `P_sleep`: 10 µW = 0.01 mW (Deep Sleep Power)
   - `t_active`: 1 second
   - `t_sleep`: 59 seconds
   - `t_period`: 60 seconds

2. **Calculate Total Energy per Cycle**:
   - `Energy_active` = `P_active` × `t_active` = 10 mW × 1 s = 10 mJ
   - `Energy_sleep` = `P_sleep` × `t_sleep` = 0.01 mW × 59 s = 0.59 mJ
   - `Energy_total` = 10 mJ + 0.59 mJ = 10.59 mJ

3. **Calculate Average Power**:
   - `P_average` = `Energy_total` / `t_period` = 10.59 mJ / 60 s ≈ **0.177 mW**

  > **Key Equation:** $$ P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}} $$

  > **Options:**
  > [ ] 10 mW
  > [ ] 5.0 mW
  > [x] ~0.18 mW
  > [ ] ~0.17 mW

  📖 **Deep Dive:** [TinyML Microcontrollers](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Solar-Powered Wildlife Cam</b> · <code>duty-cycling-energy-consumption</code></summary>

- **Interviewer:** "You're designing a remote wildlife camera using a Cortex-M4 microcontroller. The device stays in deep sleep and wakes up once every minute to capture an image and run an inference, which takes 200ms. Given the specs from our cheat sheet, calculate the average power consumption of the device. The Cortex-M4 consumes 50 mW during active inference and 10 µW during deep sleep."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to simply average the active and sleep power ratings, or to ignore the sleep power entirely because it seems negligible. Both approaches fail to account for the fact that the device spends the vast majority of its time in the low-power state, which dominates the total energy budget over time.

  **Realistic Solution:** The correct way to solve this is to calculate the total energy consumed over one full cycle (active + sleep) and then divide by the cycle's duration to find the average power. The device is active for 0.2 seconds and sleeps for 59.8 seconds in each 60-second period. This weighted average is the only accurate method to determine if a small solar panel can sustain the device.

  > **Napkin Math:** 1.  **Parameters:**
    -   `P_active = 50 mW`
    -   `P_sleep = 10 µW = 0.01 mW`
    -   `t_active = 200 ms = 0.2 s`
    -   `t_period = 1 minute = 60 s`
2.  **Calculate sleep time:**
    -   `t_sleep = t_period - t_active = 60 s - 0.2 s = 59.8 s`
3.  **Calculate energy per cycle:**
    -   `E_cycle = (P_active × t_active) + (P_sleep × t_sleep)`
    -   `E_cycle = (50 mW × 0.2 s) + (0.01 mW × 59.8 s) = 10 mJ + 0.598 mJ = 10.598 mJ`
4.  **Calculate average power:**
    -   `P_avg = E_cycle / t_period = 10.598 mJ / 60 s ≈ 0.177 mW`

  > **Key Equation:** $\bar{P} = \frac{ (P_{\text{active}} \cdot t_{\text{active}}) + (P_{\text{sleep}} \cdot t_{\text{sleep}}) }{ t_{\text{period}} }$

  > **Options:**
  > [ ] ~10.13 mW (Misinterprets µW as mW)
  > [ ] ~25.0 mW (Averages power without time weighting)
  > [ ] ~0.167 mW (Ignores sleep power in calculation)
  > [x] ~0.177 mW

  📖 **Deep Dive:** [Numbers Every ML Systems Engineer Should Know](NUMBERS.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Duty-Cycled Bird Watcher</b> · <code>tinyml-duty-cycling</code></summary>

- **Interviewer:** "You're designing a remote bird-song detector using a Cortex-M4 based microcontroller. It needs to operate for months on a small battery. To save energy, you use a duty cycle: the device listens for 1 second (active) and then enters deep sleep for the next 9 seconds. Based on the *Numbers Every ML Systems Engineer Should Know*, a Cortex-M4 consumes about 10 mW when active and 10 µW in deep sleep. What is the average power consumption of the device?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the average power by simply taking the arithmetic mean of the active and sleep power values, completely ignoring the *time* spent in each state. This leads to a massive overestimation of the power budget because the device spends the vast majority of its time (90% in this case) in the ultra-low-power sleep state.

  **Realistic Solution:** The correct way to calculate average power for a duty-cycled system is to find the total energy consumed over one full cycle (active energy + sleep energy) and then divide by the total cycle duration. The active phase dominates the energy budget per second, but the long sleep phase dramatically reduces the *average* power consumption over time, which is the critical metric for battery life estimation.

  > **Napkin Math:** 1. **Identify parameters:**
   - `P_active` = 10 mW
   - `P_sleep` = 10 µW = 0.01 mW
   - `t_active` = 1 s
   - `t_sleep` = 9 s
   - `t_period` = 1 s + 9 s = 10 s

2. **Calculate total energy per cycle:**
   - `Energy_cycle` = (`P_active` × `t_active`) + (`P_sleep` × `t_sleep`)
   - `Energy_cycle` = (10 mW × 1 s) + (0.01 mW × 9 s)
   - `Energy_cycle` = 10 mJ + 0.09 mJ = 10.09 mJ

3. **Calculate average power:**
   - `P_avg` = `Energy_cycle` / `t_period`
   - `P_avg` = 10.09 mJ / 10 s = 1.009 mW

4. **Result:** The average power is approximately **1.01 mW**.

  > **Key Equation:** $$P_{\text{avg}} = \frac{ (P_{\text{active}} \cdot t_{\text{active}}) + (P_{\text{sleep}} \cdot t_{\text{sleep}}) }{ t_{\text{period}} }$$

  > **Options:**
  > [ ] 5.005 mW
  > [ ] 1.0 mW
  > [x] 1.009 mW
  > [ ] 10.01 mW

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Silent Power Drain</b> · <code>duty-cycle-power</code></summary>

- **Interviewer:** "You're designing a battery-powered acoustic sensor to detect anomalies in factory machinery. The device uses a Cortex-M4 microcontroller. It wakes up for 1 second every minute to run an inference, then goes into deep sleep. The MCU consumes 50 mW while active and 10 µW in deep sleep. Explain how you would calculate the average power consumption and what the final value is."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the power consumed during deep sleep, assuming it's zero. While small, sleep current is a dominant factor in the energy budget for devices that sleep most of the time. Another error is to simply average the active and sleep power values, ignoring the time spent in each state.

  **Realistic Solution:** The correct way to calculate average power is to determine the total energy consumed over a single cycle (active + sleep) and then divide by the total cycle duration. The energy is the sum of (power × time) for each state.

First, convert all power units to be the same (milliwatts): 10 µW = 0.01 mW.
The cycle period is 60 seconds (1 minute). The active time is 1 second, and the sleep time is 59 seconds.

Energy = (P_active × t_active) + (P_sleep × t_sleep)
Energy = (50 mW × 1 s) + (0.01 mW × 59 s) = 50 mJ + 0.59 mJ = 50.59 mJ

Average Power = Total Energy / Total Time
Average Power = 50.59 mJ / 60 s ≈ 0.843 mW.

  > **Napkin Math:** 1. **Active Energy:** 50 mW × 1 s = 50 mJ
2. **Sleep Energy:** 10 µW × 59 s = 0.01 mW × 59 s = 0.59 mJ
3. **Total Energy per Cycle:** 50 mJ + 0.59 mJ = 50.59 mJ
4. **Average Power:** 50.59 mJ / 60 s ≈ 0.843 mW

  > **Key Equation:** $$P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$$

  > **Options:**
  > [ ] ~0.833 mW (Mistake: Ignored sleep power)
  > [ ] ~10.67 mW (Mistake: Unit confusion, treated 10µW as 10mW)
  > [x] ~0.843 mW
  > [ ] ~25.0 mW (Mistake: Averaged power values directly)

  📖 **Deep Dive:** [TinyML](tinyml/README.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Battery-Powered Birdwatcher</b> · <code>duty-cycle-energy</code></summary>

- **Interviewer:** "You're designing a battery-powered wildlife camera using a Cortex-M4 microcontroller. In its active state (running a 1-second inference), it consumes 50 mW. In deep sleep, it consumes 10 µW. The device is expected to wake up on average 6 times per hour. Your power source is an 888 mWh battery. Calculate the device's approximate battery life in days."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to either ignore the sleep power, assuming it's negligible, or to miscalculate the average power by mixing units (mW vs µW). Ignoring sleep power overestimates the battery life, while the unit error drastically underestimates it by orders of magnitude.

  **Realistic Solution:** The solution requires calculating the weighted average power consumption based on the duty cycle. The device is active for 6 seconds per hour and in sleep for the remaining 3594 seconds. This average power is then used to determine how long the 888 mWh battery will last.

  > **Napkin Math:** # 1. Calculate time in each state per hour (3600s)
t_active = 6 triggers/hr * 1 s/trigger = 6 s
t_sleep = 3600 s - 6 s = 3594 s

# 2. Convert all power to mW
P_active = 50 mW
P_sleep = 10 µW = 0.01 mW

# 3. Calculate average power (P_avg) using the duty cycle formula
P_avg = ((P_active * t_active) + (P_sleep * t_sleep)) / 3600 s
P_avg = ((50 mW * 6s) + (0.01 mW * 3594s)) / 3600s
P_avg = (300 mJ + 35.94 mJ) / 3600s = 335.94 mJ / 3600s ≈ 0.0933 mW

# 4. Calculate battery life
Battery_Energy = 888 mWh
Life_hours = Battery_Energy / P_avg = 888 mWh / 0.0933 mW ≈ 9517 hours
Life_days = 9517 hours / 24 hours/day ≈ 397 days

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{(\text{P}_{\text{active}} \times \text{t}_{\text{active}}) + (\text{P}_{\text{sleep}} \times \text{t}_{\text{sleep}})}{\text{t}_{\text{period}}}$

  > **Options:**
  > [ ] ~18 hours
  > [ ] ~4 days
  > [x] ~397 days
  > [ ] ~444 days

  📖 **Deep Dive:** [TinyML Systems](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Duty-Cycled Wildlife Sensor</b> · <code>tinyml-duty-cycle-power</code></summary>

- **Interviewer:** "You're designing a battery-powered wildlife sensor that uses a Cortex-M4 to listen for a specific animal call. It spends most of its time in a deep sleep mode consuming 10 µW (micro-watts). When it detects a potential sound, it wakes up and runs a keyword spotting model for 1 second, consuming 10 mW (milli-watts), before going back to sleep. Assuming this cycle repeats every 10 seconds (1 second active, 9 seconds sleep), how do you calculate the *average* power consumption, and what is the result?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the sleep power, assuming it's effectively zero. While small, it can dominate the total energy budget over long deployments. Another frequent error is to incorrectly average the power ratings directly (e.g., (10mW + 10µW)/2) without weighting them by the time spent in each state. A third error is to simply report the active power, ignoring the averaging effect of the duty cycle.

  **Realistic Solution:** To find the average power, you must calculate the total energy consumed over one full period and then divide by the period's duration. The total energy is the sum of the energy used during the active phase and the energy used during the sleep phase. The device is active for 10% of the time and asleep for 90% of the time.

  > **Napkin Math:** 1. **Calculate Active Energy per cycle:**
   E_active = P_active × t_active
   E_active = 10 mW × 1 s = 10 mJ (milli-Joules)

2. **Calculate Sleep Energy per cycle:**
   E_sleep = P_sleep × t_sleep
   E_sleep = 10 µW × 9 s = 90 µJ (micro-Joules) = 0.09 mJ

3. **Calculate Total Energy per cycle:**
   E_total = E_active + E_sleep
   E_total = 10 mJ + 0.09 mJ = 10.09 mJ

4. **Calculate Average Power:**
   P_avg = E_total / t_period
   P_avg = 10.09 mJ / 10 s = 1.009 mW

The average power consumption is ~1.01 mW.

  > **Key Equation:** P_{\text{avg}} = \frac{(P_{\text{active}} \cdot t_{\text{active}}) + (P_{\text{sleep}} \cdot t_{\text{sleep}})}{t_{\text{period}}}

  > **Options:**
  > [ ] 10 mW (Mistake: Ignores the averaging effect of the duty cycle)
  > [ ] 10 µW (Mistake: Unit confusion, treating the 10mW active power as 10µW)
  > [x] ~1.01 mW
  > [ ] ~5 mW (Mistake: Incorrectly averaging the power ratings without weighting by time)

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Energy-Neutral Wildlife Camera</b> · <code>duty-cycling</code></summary>

- **Interviewer:** "You're designing a remote wildlife camera using a Cortex-M4. It's powered by a small solar panel that provides an average of 1 mW of power. The device has two states: 'active' (running inference, consuming 50 mW) and 'deep sleep' (consuming 10 µW). To remain energy-neutral, the device's average power consumption over its duty cycle must not exceed the 1 mW provided by the panel. For a 60-second cycle period, calculate the maximum time the camera can be in the 'active' state."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the power consumed during deep sleep. While it's very small (10µW is 1/5000th of the active power), it is not zero. In a tight power budget over millions of cycles, this small, constant drain matters. Ignoring it leads to a slight overestimation of the available active time, which can cause battery drain over the long term.

  **Realistic Solution:** The correct way to solve this is to use the average power equation for a duty cycle, including both active and sleep states. The average power is the weighted sum of the power in each state, divided by the total period. We set this equal to the 1 mW provided by the solar panel and solve for the active time.

The sleep power (10 µW) must be converted to mW (0.01 mW) to keep units consistent. The calculation shows that the device can only be active for about 1.19 seconds out of every minute to stay within its power budget.

  > **Napkin Math:** 1. **Define the average power equation:**
   P_avg = (P_active * t_active + P_sleep * t_sleep) / t_period

2. **Substitute known values and relationships:**
   - P_avg = 1 mW
   - P_active = 50 mW
   - P_sleep = 10 µW = 0.01 mW
   - t_period = 60 s
   - t_sleep = t_period - t_active = 60 - t_active

3. **Set up the equation to solve for t_active:**
   1 mW = (50 * t_active + 0.01 * (60 - t_active)) / 60

4. **Solve for t_active:**
   - Multiply by 60:  60 = 50 * t_active + 0.01 * (60 - t_active)
   - Distribute:     60 = 50 * t_active + 0.6 - 0.01 * t_active
   - Combine terms:    60 - 0.6 = (50 - 0.01) * t_active
   - Simplify:         59.4 = 49.99 * t_active
   - Isolate t_active: t_active = 59.4 / 49.99
   - Result:           t_active ≈ 1.188 seconds

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{\text{P}_{\text{active}} \cdot \text{t}_{\text{active}} + \text{P}_{\text{sleep}} \cdot (\text{t}_{\text{period}} - \text{t}_{\text{active}})}{\text{t}_{\text{period}}}$

  > **Options:**
  > [ ] 1.20 seconds
  > [ ] ~58.81 seconds
  > [x] ~1.19 seconds
  > [ ] 0.012 seconds

  📖 **Deep Dive:** [Microcontrollers](https://mlsysbook.ai/tinyml/01_microcontroller.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Camera's Power Budget</b> · <code>duty-cycling-power</code></summary>

- **Interviewer:** "You're designing a remote wildlife camera using a Cortex-M4 based microcontroller. The system stays in a deep sleep mode, wakes up once every minute to capture an image and run a 2-second inference workload, and then goes back to sleep. Using the hardware constants, explain the average power consumption of the device. This is the single most critical number for sizing the battery and determining deployment lifetime."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often make two mistakes here: 1) They completely ignore the energy consumed during the sleep state, assuming it's zero. While small, it becomes significant over days or weeks. 2) They mismanage units, mixing up milliwatts (mW) and microwatts (µW), leading to an answer that's off by orders of magnitude.

  **Realistic Solution:** The correct way to calculate average power is to create a weighted average of the power consumed in each state (active and sleep) over one full operational period. You must convert all power figures to the same unit (e.g., mW) before calculating.

- Active State: The device is active for 2 seconds.
- Sleep State: The device sleeps for the rest of the minute (60s - 2s = 58s).
- Total Period: 60 seconds.

  > **Napkin Math:** 1.  **Define States & Power:**
    -   `P_active` = 50 mW (from constants, upper bound for Cortex-M4)
    -   `P_sleep` = 10 µW = 0.01 mW (from constants, upper bound for deep sleep)
    -   `t_active` = 2 s
    -   `t_sleep` = 58 s
    -   `t_period` = 60 s

2.  **Calculate Energy per State (Power × Time):**
    -   `E_active` = 50 mW × 2 s = 100 mJ
    -   `E_sleep` = 0.01 mW × 58 s = 0.58 mJ

3.  **Calculate Total Energy and Average Power:**
    -   `E_total` = 100 mJ + 0.58 mJ = 100.58 mJ
    -   `P_avg` = `E_total` / `t_period` = 100.58 mJ / 60 s ≈ 1.68 mW

  > **Key Equation:** $P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$

  > **Options:**
  > [ ] ~1.67 mW
  > [ ] ~9.7 mW
  > [x] ~1.68 mW
  > [ ] ~25 mW

  📖 **Deep Dive:** [TinyML Microcontrollers](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Sensor's Power Budget</b> · <code>tinyml-duty-cycle-power</code></summary>

- **Interviewer:** "You're designing a battery-powered sensor for a remote wildlife monitoring system. It uses a Cortex-M4 microcontroller to run a simple motion detection model. The device wakes up once every 20 seconds to perform a 2-second inference.

Given the following specs from our hardware sheet:
- **Active Power** (during inference): 50 mW
- **Deep Sleep Power**: 10 µW

Calculate the average power consumption of the device over one cycle. This is the number you'd use to size the battery and estimate device lifetime."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is either completely ignoring the energy consumed during sleep or making a unit conversion error between milliwatts (mW) and microwatts (µW). While the sleep power is tiny, it dominates the energy budget over long periods in ultra-low-power applications. A second mistake is incorrectly averaging the power figures directly without weighting them by the time spent in each state.

  **Realistic Solution:** The correct approach is to calculate the total energy consumed in one full cycle (active + sleep) and then divide by the total cycle time to find the average power. It's critical to convert all power units to be the same (e.g., milliwatts) before calculating.

The cycle consists of 2 seconds active and 18 seconds sleeping, for a total period of 20 seconds. The sleep power of 10 µW is equal to 0.01 mW. By calculating the time-weighted energy for each state, we get the true average power.

  > **Napkin Math:** 1.  **Convert Units:** Ensure all power values are in the same unit.
    - `P_sleep = 10 µW = 0.01 mW`

2.  **Calculate Energy per State:** Energy (in millijoules) = Power (in milliwatts) × Time (in seconds).
    - `E_active = 50 mW × 2 s = 100 mJ`
    - `E_sleep = 0.01 mW × 18 s = 0.18 mJ`

3.  **Calculate Total Energy per Cycle:**
    - `E_total = E_active + E_sleep = 100 mJ + 0.18 mJ = 100.18 mJ`

4.  **Calculate Average Power:** Average Power = Total Energy / Total Time.
    - `P_avg = 100.18 mJ / 20 s = 5.009 mW`

  > **Key Equation:** $$ P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}} $$

  > **Options:**
  > [ ] ~14.0 mW
  > [ ] ~25.0 mW
  > [ ] ~5.00 mW
  > [x] ~5.01 mW

  📖 **Deep Dive:** [TinyML Systems](tinyml/README.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Sensor's Energy Budget</b> · <code>duty-cycling</code></summary>

- **Interviewer:** "You are designing a battery-powered audio sensor to detect rare bird calls in a remote rainforest. The device uses a Cortex-M4 microcontroller. It remains in a deep sleep state most of the time, waking up for 1 second every minute to run an inference. From the datasheet, you know:

- Active power (inference): **40 mW**
- Deep sleep power: **10 µW**

To select the right battery and solar harvesting setup, you first need a baseline. Explain how you would calculate the device's average power consumption, and then calculate it."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to ignore the power consumed during the sleep state, assuming it's zero. This underestimates the total energy drain, especially over long periods. Another error is to incorrectly average the power values without weighting them by the time spent in each state, leading to a wildly inaccurate estimate.

  **Realistic Solution:** The correct way to calculate the average power is to find the total energy consumed over one full cycle (active + sleep) and then divide by the cycle's total time period. The energy for each state is its power multiplied by the time spent in that state. The sleep power, though tiny, is consumed for a much longer duration than the active power and must be included.

First, ensure all units are consistent (e.g., convert everything to microwatts). Then, apply the duty cycle formula to find the time-weighted average power.

  > **Napkin Math:** 1.  **Standardize Units:**
    -   `P_active` = 40 mW = 40,000 µW
    -   `P_sleep` = 10 µW
2.  **Define Time Periods:**
    -   `t_active` = 1 second
    -   `t_sleep` = 60 seconds - 1 second = 59 seconds
    -   `t_period` = 60 seconds
3.  **Calculate Total Energy per Cycle:**
    -   `E_cycle` = (`P_active` × `t_active`) + (`P_sleep` × `t_sleep`)
    -   `E_cycle` = (40,000 µW × 1 s) + (10 µW × 59 s) = 40,000 µJ + 590 µJ = 40,590 µJ
4.  **Calculate Average Power:**
    -   `P_avg` = `E_cycle` / `t_period`
    -   `P_avg` = 40,590 µJ / 60 s ≈ 676.5 µW

  > **Key Equation:** $\text{P}_{\text{avg}} = \frac{(\text{P}_{\text{active}} \times \text{t}_{\text{active}}) + (\text{P}_{\text{sleep}} \times \text{t}_{\text{sleep}})}{\text{t}_{\text{period}}}$

  > **Options:**
  > [ ] ~667 µW
  > [ ] ~20,005 µW
  > [x] ~677 µW
  > [ ] ~40,590 µW

  📖 **Deep Dive:** [TinyML Readme](tinyml/README.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Duty Cycle Constraint</b> · <code>real-time-duty-cycle</code></summary>

- **Interviewer:** "You are designing a battery-powered keyword spotting device using a Cortex-M4 microcontroller. It must process a continuous stream of 1-second audio windows to detect a wake word. What is the correct term for the percentage of time the CPU is in an active state processing audio versus in a low-power sleep state?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the duty cycle with related metrics like latency or throughput. Latency is the duration of the active period (`t_active`), and throughput is the rate of processing. Duty cycle, however, is the specific *ratio* of active time to the total time period, which is the critical factor for calculating energy consumption in battery-powered devices.

  **Realistic Solution:** The correct term is **Duty Cycle**. It is the fraction of one period in which a system is active. For a real-time system that must process data arriving every `t_period` (e.g., 1 second), the processing time (`t_active`) must be less than this period. The resulting duty cycle directly determines the average power consumption and thus the device's battery life.

  > **Napkin Math:** Given a 1-second period and a 150ms inference time:

1.  **Calculate Duty Cycle:**
    `t_active / t_period = 150ms / 1000ms = 15%`

2.  **Calculate Average Power (using TinyML constants):**
    `P_active ≈ 50 mW`
    `P_sleep ≈ 10 µW`
    `Avg Power = (P_active × 0.15) + (P_sleep × 0.85)`
    `Avg Power = (50 mW × 0.15) + (10 µW × 0.85) ≈ 7.5 mW + 0.0085 mW ≈ 7.51 mW`

This shows that average power is dominated by the active power multiplied by the duty cycle.

  > **Key Equation:** $\text{Duty Cycle} = \frac{t_{\text{active}}}{t_{\text{period}}}$

  > **Options:**
  > [ ] Latency
  > [ ] Throughput
  > [x] Duty Cycle
  > [ ] Power Draw

  📖 **Deep Dive:** [Microcontrollers & Real-Time Constraints](tinyml/01_microcontroller.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Wildlife Sensor's Energy Budget</b> · <code>duty-cycle-power</code></summary>

- **Interviewer:** "You're designing a battery-powered wildlife monitoring sensor using a Cortex-M4 microcontroller. The device wakes up once every minute (the 'period') to perform a 1-second audio classification task, then returns to deep sleep.

Based on the datasheet, the MCU consumes 40 mW while active and 10 µW (micro-watts) in deep sleep. Calculate the average power consumption of the device over a full one-minute cycle. This average power is what determines your battery life."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers new to embedded systems often either (a) forget that the long sleep duration still contributes to the total energy budget, even if the power is tiny, or (b) incorrectly average the power values (e.g., (40mW + 10µW)/2) without weighting them by the time spent in each state.

  **Realistic Solution:** The correct way to calculate average power for a duty-cycled system is to find the total energy consumed over one full period and then divide by the duration of that period. The total energy is the sum of the energy consumed in the active state and the energy consumed in the sleep state.

1.  **Active State:** 1 second at 40 mW
2.  **Sleep State:** 59 seconds at 10 µW (or 0.010 mW)
3.  **Total Period:** 60 seconds

  > **Napkin Math:** 1. **Calculate Active Energy:**
   $E_{\text{active}} = P_{\text{active}} \times t_{\text{active}} = 40\text{ mW} \times 1\text{ s} = 40\text{ mJ}$

2. **Calculate Sleep Energy:**
   $E_{\text{sleep}} = P_{\text{sleep}} \times t_{\text{sleep}} = 10\text{ µW} \times 59\text{ s} = 0.01\text{ mW} \times 59\text{ s} = 0.59\text{ mJ}$

3. **Calculate Total Energy:**
   $E_{\text{total}} = E_{\text{active}} + E_{\text{sleep}} = 40\text{ mJ} + 0.59\text{ mJ} = 40.59\text{ mJ}$

4. **Calculate Average Power:**
   $P_{\text{avg}} = E_{\text{total}} / t_{\text{period}} = 40.59\text{ mJ} / 60\text{ s} \approx 0.6765\text{ mW}$

  > **Key Equation:** $$P_{\text{avg}} = \frac{(P_{\text{active}} \times t_{\text{active}}) + (P_{\text{sleep}} \times t_{\text{sleep}})}{t_{\text{period}}}$$

  > **Options:**
  > [ ] ~0.667 mW
  > [ ] ~20 mW
  > [x] ~0.677 mW
  > [ ] 40 mW

  📖 **Deep Dive:** [Deployed Device](https://mlsysbook.ai/tinyml/03_deployed_device.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Bird-Call Battery Drain</b> · <code>battery-duty-cycle</code></summary>

- **Interviewer:** "You're designing a battery-powered acoustic sensor to detect a rare bird call in a remote forest. Your device uses a Cortex-M4 microcontroller. It wakes up for 2 seconds every minute (60 seconds) to listen and run an inference model. From the playbook, we know the Cortex-M4 consumes about 50 mW while active and 10 µW while in deep sleep. Calculate the average power consumption of the device."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the duty-cycled active power but forget to include the power consumed during the much longer sleep period. While sleep power is tiny, it dominates the energy budget over long deployments and is a critical factor in battery life calculations.

  **Realistic Solution:** The correct way to calculate average power is to find the total energy consumed over one full cycle (active + sleep) and then divide by the cycle's time period.

1.  **Active Energy:** 50 mW × 2 s = 100 mJ
2.  **Sleep Power Conversion:** 10 µW = 0.01 mW
3.  **Sleep Energy:** 0.01 mW × (60 s - 2 s) = 0.01 mW × 58 s = 0.58 mJ
4.  **Total Energy:** 100 mJ + 0.58 mJ = 100.58 mJ
5.  **Average Power:** 100.58 mJ / 60 s ≈ 1.68 mW

  > **Napkin Math:** P_avg = (P_active * t_active) + (P_sleep * t_sleep) / t_period
P_avg = (50mW * 2s + 0.01mW * 58s) / 60s
P_avg = (100mJ + 0.58mJ) / 60s
P_avg = 100.58mJ / 60s
P_avg ≈ 1.68 mW

  > **Key Equation:** $$ P_{\text{avg}} = \frac{P_{\text{active}} \cdot t_{\text{active}} + P_{\text{sleep}} \cdot t_{\text{sleep}}}{t_{\text{period}}} $$

  > **Options:**
  > [ ] ~1.67 mW (Incorrect: This result comes from completely ignoring the sleep power contribution, a common oversight.)
  > [ ] ~11.33 mW (Incorrect: This result comes from misinterpreting 10µW as 10mW, a 1000x unit error.)
  > [x] ~1.68 mW (Correct: This accurately accounts for the energy consumed during both the active and sleep states over the full period.)
  > [ ] ~100.6 mW (Incorrect: This fails to divide the total energy (mJ) by the time period (s), confusing energy with average power.)

  📖 **Deep Dive:** [TinyML Systems](https://mlsysbook.ai/tinyml/)
  </details>
</details>










































#### 🟢 L3
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The FOTA Update Risk</b> · <code>deployment</code></summary>

- **Interviewer:** "You have 10,000 sensor nodes deployed in a warehouse, each running a vibration anomaly detection model on a Cortex-M4. You need to update the model. The nodes communicate via LoRaWAN (250 bytes/second effective throughput). How do you update them, and what happens if the update fails?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Send the new firmware over LoRaWAN and flash it." At 250 bytes/second, a 200 KB model takes 200,000/250 = 800 seconds = 13 minutes per device. With 10,000 devices sharing the LoRa channel: years.

  **Realistic Solution:** FOTA (Firmware Over-The-Air) for constrained networks requires a different approach:

  (1) **Delta updates** — don't send the full model. Compute a binary diff between the old and new model weights. If only 10% of weights changed (fine-tuning), the delta is ~20 KB instead of 200 KB. Transfer time: 20,000/250 = 80 seconds per device.

  (2) **Multicast** — LoRaWAN Class C supports multicast. Send the update once, all 10,000 devices receive it simultaneously. Transfer time: 80 seconds total (not per device).

  (3) **A/B flash partitioning** — the MCU's 1 MB flash is split: 500 KB for the running firmware (slot A), 500 KB for the update (slot B). The new model is written to slot B while slot A continues running. After verification (CRC check + test inference on a known input), the bootloader atomically swaps the active slot pointer.

  (4) **Failure recovery** — if the CRC check fails, the device stays on slot A and reports the failure. If the device boots from slot B and the watchdog timer fires (model crashes), the bootloader automatically reverts to slot A. The device is never bricked.

  (5) **Staged rollout** — update 100 devices first (1% of fleet). Monitor their anomaly detection accuracy for 24 hours. If no degradation, update the remaining 9,900.

  > **Napkin Math:** Full model: 200 KB. Delta: 20 KB. LoRaWAN multicast: 20 KB / 250 B/s = 80 seconds. Verification: CRC (1ms) + test inference (50ms) = 51ms. Swap: atomic pointer write (1ms). Total per device: 80s transfer + 0.05s verify + 0.001s swap. Fleet of 10,000 via multicast: **80 seconds** + staged validation (24 hours for safety). Without delta/multicast: 200 KB × 10,000 / 250 B/s = 8,000,000 seconds = **92.6 days**.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> Model Versioning on MCU</b> · <code>deployment</code></summary>

- **Interviewer:** "You manage a fleet of 500 RP2040-based (Cortex-M0+, 264 KB SRAM, 2 MB flash) environmental sensors. After three OTA updates, your support team can't tell which model version a device is running. A customer reports false alarms, and you need to know if they're on model v1.2 or v1.4. How do you track model versions on a device with no OS and no filesystem?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Store the version in a global variable in firmware." This ties the model version to the firmware version — if you update only the model (not the firmware), the version string is stale. Worse, a global variable lives in SRAM and is lost on reset.

  **Realistic Solution:** Store model metadata in a structured header prepended to the model binary in flash:

  **Model header format (64 bytes, fixed at a known flash address):**

  | Offset | Size | Field | Example |
  |--------|------|-------|---------|
  | 0x00 | 4 B | Magic number | `0x4D4C5359` ("MLSY") |
  | 0x04 | 4 B | Header version | `0x00000001` |
  | 0x08 | 4 B | Model version (semver packed) | `0x00010400` (v1.4.0) |
  | 0x0C | 4 B | Model size (bytes) | `0x0001C000` (114,688) |
  | 0x10 | 4 B | CRC-32 of model weights | `0xA3F7B2C1` |
  | 0x14 | 8 B | Build timestamp (Unix epoch) | `1710000000` |
  | 0x1C | 16 B | Model hash (first 128 bits of SHA-256) | Unique model fingerprint |
  | 0x2C | 4 B | Target hardware ID | `0x00002040` (RP2040) |
  | 0x30 | 16 B | Reserved / padding | `0x00...` |

  **At boot:** firmware reads the magic number at the known flash address. If valid, it parses the header and exposes the model version via a BLE characteristic or UART command. The support team queries any device with `AT+MODELVER` and gets back `v1.4.0, built 2025-03-10, CRC OK`.

  **During OTA:** the new model binary includes its header. After flashing, the bootloader verifies the magic number and CRC before marking the update as valid. If the header is corrupt or the CRC doesn't match, the update is rejected and the device stays on the previous version.

  **Fleet-wide:** the gateway collects model versions during daily telemetry. A dashboard shows: 480 devices on v1.4.0, 15 on v1.2.0 (failed update), 5 offline. The support team immediately knows the customer's device is on v1.2 and pushes a targeted update.

  > **Napkin Math:** Header overhead: 64 bytes per model. On 2 MB flash: 64 / 2,097,152 = 0.003% — negligible. Boot-time header validation: read 64 bytes from flash (64 / 4 bytes per read × 2 cycles = 32 cycles) + CRC check of 114 KB model (114,000 × 3 cycles = 342,000 cycles at 133 MHz = 2.6 ms). Total boot overhead: **< 3 ms**. BLE version query: 20 bytes response, single BLE packet, < 10 ms round-trip.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Model Checksum Paradox</b> · <code>deployment</code> <code>security</code></summary>

- **Interviewer:** "You deploy a `.tflite` model to an MCU via an OTA update. To ensure the file isn't corrupted, you calculate a CRC32 checksum of the file on your server, send it to the device, and the device calculates the CRC32 of the downloaded file. They match. The device reboots. The model fails to load, crashing the device. How can a file have a perfect checksum but still be completely invalid?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "CRC32 has collisions." While CRC32 isn't cryptographically secure, random collisions on OTA updates are vanishingly rare. The problem is what happened *after* the download.

  **Realistic Solution:** You checked the data in transit, but ignored **Flash Write Errors**.

  Your OTA architecture likely did this:
  1. Download chunk from network into RAM.
  2. Update running CRC32 calculation using the RAM buffer.
  3. Write the RAM buffer to Flash memory.
  4. Compare final CRC32. Match!

  The flaw is that you verified the bytes *before* they were written to the physical storage medium. If the Flash memory was worn out, if there was a voltage droop during the write, or if you forgot to erase the Flash sector before writing (Flash can only turn 1s into 0s without an erase), the physical bits on the silicon will be corrupted. Your RAM buffer was perfect, but the persistent storage is garbage.

  **The Fix:** Always calculate the checksum by reading the data **back out of the Flash memory** after the write is complete. This verifies the entire pipeline: network -> RAM -> Flash Controller -> Physical Silicon.

  > **Napkin Math:** Flash memory must be erased to `0xFF` before writing. If you write `0xAA` (10101010) over unerased data like `0x0F` (00001111), the result is the bitwise AND: `0x0A` (00001010). The file on disk is destroyed, even though the byte in RAM you checksummed was a perfect `0xAA`.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> BLE Throughput for Model Update</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "Your TinyML wearable needs an over-the-air model update via BLE 5.0. The new model is 150 KB (INT8 quantized, stored in external flash). BLE 5.0 supports 2 Mbps PHY with a maximum data throughput of ~1.4 Mbps after protocol overhead. The device has a 100 mAh battery at 3.7V. The BLE radio draws 8 mA during active transmission/reception. Estimate the update time and the battery cost of the update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "150 KB at 1.4 Mbps = 0.86 seconds. Trivial." This uses the raw PHY throughput and ignores BLE's connection-oriented protocol overhead, which dramatically reduces effective throughput.

  **Realistic Solution:** BLE data transfer is constrained by connection intervals, MTU size, and the number of packets per connection event.

  **(1) BLE throughput reality.** BLE 5.0 at 2M PHY: raw bit rate = 2 Mbps. But BLE transfers data in ATT (Attribute Protocol) notifications within connection events. Maximum ATT payload (with DLE — Data Length Extension): 244 bytes per packet. Packets per connection event: typically 4–6 (limited by connection event length). Connection interval: 7.5ms minimum (iOS enforces 15ms minimum). Effective throughput: 6 packets × 244 bytes / 7.5ms = **195 KB/s** (best case). Realistic (Android, 15ms interval, 4 packets): 4 × 244 / 15ms = **65 KB/s**.

  **(2) Update time.** At 65 KB/s: 150 KB / 65 = **2.3 seconds**. At 195 KB/s (best case): 0.77 seconds. But add protocol overhead: connection setup (200ms), service discovery (500ms), MTU negotiation (100ms), and post-transfer verification (CRC check, 200ms). Total: 2.3 + 1.0 = **3.3 seconds** (realistic). Best case: 1.77 seconds.

  **(3) Battery cost.** BLE radio at 8 mA for 3.3 seconds: 8 mA × 3.3s / 3600 = **0.0073 mAh**. Battery: 100 mAh. Cost: 0.0073% of battery. Negligible — you could do 13,700 model updates on a full charge. The BLE update cost is irrelevant to battery life.

  **(4) The real bottleneck: flash write.** Writing 150 KB to external NOR flash at ~1 MB/s (typical SPI flash write speed with 256-byte page writes): 150 KB / 1 MB/s = **150ms**. But NOR flash requires sector erase before write: 150 KB / 4 KB sectors = 38 sector erases × 50ms each = **1.9 seconds**. Total: BLE transfer (2.3s) + flash erase (1.9s) + flash write (0.15s) + verify (0.2s) = **4.55 seconds**. Flash erase is 42% of the total update time.

  > **Napkin Math:** BLE transfer: 2.3s at 65 KB/s. Flash erase: 1.9s (38 sectors × 50ms). Flash write: 0.15s. Verify: 0.2s. Total: 4.55s. Battery cost: 8 mA × 4.55s = 36.4 mA·ms = 0.01 mAh (0.01% of battery). For a 500 KB model: BLE = 7.7s, flash erase = 6.25s (125 sectors), total = 14.2s. At this size, BLE and flash erase are roughly equal bottlenecks. For models >1 MB: flash erase dominates. Optimization: use a delta update (send only changed weights) — typical delta for a fine-tuned model: 10–20% of weights change → 15–30 KB transfer instead of 150 KB.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The A/B Test Brick-pocalypse</b> · <code>ab-testing-ota-risk</code></summary>

- **Interviewer:** "You are managing a fleet of 500,000 battery-powered smart doorbells, each with a 1MB Flash budget. You want to A/B test a new person detection model. The total firmware size is 450KB. The Over-the-Air (OTA) update mechanism is reliable, with only a 0.05% failure rate (e.g., due to power loss or connectivity issues during the write process), but a failure requires a full device replacement. You plan to roll out the new model to a 20% test group. Your manager, accustomed to web-based A/B tests, sees this as a low-risk operation. How would you apply your systems knowledge to diagnose the single greatest quantitative risk to the business from this 'simple' A/B test?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers from a cloud background often perceive A/B testing as a cheap, reversible software flag change. They might focus on secondary metrics like the new model's latency, power draw, or the network cost of the deployment. They fail to appreciate that on embedded devices, a 'deployment' is a fragile firmware write operation, and a failure at scale, even with a low probability, can lead to catastrophic and irreversible hardware losses.

  **Realistic Solution:** The greatest risk is the number of permanently bricked devices, leading to direct financial loss from hardware replacement and customer support. Unlike a server that can be easily reimaged, a failed OTA update on a consumer device is often unrecoverable. An A/B test isn't just a metadata flip; it's a high-risk firmware operation performed on a massive number of devices simultaneously. The key is to calculate the expected number of failures and translate that into a dollar amount, showing that the 'low probability' event is a near certainty at scale.

  > **Napkin Math:** 1. **Calculate the size of the test group:**
   500,000 devices * 20% = 100,000 devices

2. **Calculate the expected number of bricked devices:**
   100,000 devices * 0.05% failure rate = 50 devices

3. **Estimate the financial impact:**
   Assume a replacement cost (hardware + shipping + support) of $50 per device.
   50 devices * $50/device = $2,500

4. **Diagnose the Risk:**
   The A/B test will result in an expected, immediate loss of $2,500 from bricked units. This is a direct, quantifiable cost. While other factors like battery drain exist, the irreversible hardware failure is the most significant and immediate business risk that distinguishes this from a simple web A/B test.

  > **Key Equation:** E[\text{Loss}] = \text{FleetSize} \times \text{TestGroup\%} \times \text{FailureRate} \times \text{UnitCost}

  > **Options:**
  > [ ] The network cost of sending a 450KB update to 100,000 devices will be the dominant expense.
  > [ ] The increased power consumption of the new model will drain batteries faster, leading to customer complaints.
  > [ ] The new model might have higher latency, violating the real-time processing budget of the doorbell.
  > [x] The expected number of permanently bricked devices due to OTA update failures represents the most significant, direct financial risk.

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/README.html)
  </details>
</details>



#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Offline Drift Detector</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your deployed anomaly detector starts producing false positives after 3 months. The device has no cloud connection — it operates fully offline. How do you detect and handle model drift on a device with 256 KB SRAM and no internet?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Upload data to the cloud for analysis." There's no cloud connection. You must handle this entirely on-device.

  **Realistic Solution:** On-device drift detection with minimal resources:

  (1) **Running statistics** — maintain exponential moving averages of the model's input feature statistics (mean and variance of each input channel). Storage: 2 floats × N channels × 4 bytes = ~64 bytes for a 8-channel sensor. When the running mean drifts beyond 3σ of the baseline (computed at deployment), flag a drift event. This detects covariate shift (e.g., sensor degradation, environmental change).

  (2) **Prediction distribution monitoring** — track the distribution of the model's output confidence scores. A healthy model produces mostly high-confidence predictions (normal) with occasional low-confidence ones (anomalies). If the ratio of low-confidence predictions exceeds a threshold (e.g., >30% of predictions in the last hour), the model is likely seeing OOD data.

  (3) **Self-calibration** — store a small set of "golden" reference inputs in flash (10 known-normal vibration signatures, ~5 KB). Periodically (once per hour), run inference on these references. If the model's predictions on known-normal inputs start drifting (confidence drops below 0.95), the model or the sensor has degraded.

  (4) **Graceful response** — when drift is detected: (a) increase the anomaly threshold to reduce false positives (accepting more false negatives), (b) activate an LED indicator for maintenance personnel, (c) log the drift event with timestamp to flash for later retrieval, (d) if drift exceeds a critical threshold, fall back to a simple threshold-based detector (no ML) until the device is serviced.

  > **Napkin Math:** Running statistics: 64 bytes RAM. Golden references: 5 KB flash. Hourly self-test: 10 inferences × 50ms = 500ms per hour = 0.014% CPU overhead. Drift detection latency: 1 hour (self-test interval). Storage for drift log: 20 bytes per event × 100 events = 2 KB flash. Total resource cost: 64 bytes RAM + 7 KB flash — negligible on a 256 KB SRAM / 1 MB flash device.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> FOTA Update Integrity Verification</b> · <code>deployment</code> <code>security</code></summary>

- **Interviewer:** "Your predictive maintenance sensors receive firmware updates over-the-air (FOTA) containing a new TFLite Micro model. The bootloader verifies the binary hash (SHA-256) before swapping partitions. Why is verifying the binary hash insufficient for ML models, and how do you implement functional model attestation (inference on a golden test input) to prove the model's math is intact?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "If the SHA-256 hash matches, the file hasn't been corrupted, so the model is safe to run." This assumes the model was compiled correctly and the runtime on the device is perfectly compatible with the new model's operators.

  **Realistic Solution:** A binary hash only proves the file arrived exactly as it was sent. It does *not* prove that the ML model will actually execute correctly on the device. The new model might use a TFLite operator (e.g., `RESIZE_NEAREST_NEIGHBOR`) that isn't compiled into the device's specific TFLite Micro runtime, causing a hard fault on the first real inference. Or, a quantization bug on the build server might have produced a model that hashes perfectly but outputs garbage predictions.

  To guarantee ML integrity, the bootloader (or a first-boot initialization sequence) must perform **functional model attestation**. The device stores a "golden" test input (e.g., a pre-processed vibration spectrogram) and its expected output tensor in a read-only flash sector. After the SHA-256 check passes, the device loads the new model into the tensor arena, feeds it the golden input, and runs a full forward pass. It then compares the output tensor to the expected reference. If the Mean Squared Error (MSE) is below a strict threshold, the model's *math* is proven intact, and the update is committed. If it crashes or outputs garbage, the device rolls back to the previous partition.

  > **Napkin Math:** SHA-256 on a 200 KB model takes ~15ms on a Cortex-M4. Functional attestation (running one inference) takes ~50ms. The golden input (e.g., 32×32 INT8 spectrogram) takes 1 KB of flash. The expected output (e.g., 4-class probabilities) takes 4 bytes. For a 65ms total boot-time penalty and 1 KB of flash overhead, you completely eliminate the risk of bricking a remote sensor with a mathematically broken model update.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> LoRaWAN Telemetry for ML Metrics</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "You have 3,000 soil moisture sensors deployed across farmland, each running a crop stress prediction model on an STM32WL (Cortex-M4, 48 MHz, 256 KB flash, 64 KB SRAM) with built-in LoRa radio. You want to monitor model performance remotely — inference confidence, prediction distribution, drift indicators. But LoRaWAN has strict duty cycle limits. Design the telemetry payload and transmission strategy."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Send all inference results over LoRa." At 24 inferences/day (one per hour) with even a minimal 20-byte result, that's 480 bytes/day. LoRaWAN's duty cycle limits make this surprisingly expensive in airtime.

  **Realistic Solution:** LoRaWAN operates under regional duty cycle regulations (EU868: 1% duty cycle, US915: dwell time limits). You must minimize airtime:

  **LoRaWAN constraints (EU868, SF7, 125 kHz BW):**
  - Data rate: ~5.5 kbps (SF7)
  - Max payload per uplink: 222 bytes (DR5)
  - 1% duty cycle on most sub-bands: after transmitting for 1 second, you must wait 99 seconds
  - Airtime for 50-byte payload at SF7: ~72 ms → cooldown: 7.2 seconds

  **Telemetry payload design (compact binary, not JSON):**

  | Field | Size | Encoding | Description |
  |-------|------|----------|-------------|
  | Device ID | 0 B | In LoRaWAN header (DevAddr) | Free — already in the protocol |
  | Timestamp | 2 B | Minutes since midnight | Resets daily, 0-1440 |
  | Battery voltage | 1 B | (V - 2.0) × 100, uint8 | Range 2.0-4.55V, 10 mV resolution |
  | Inference count (24h) | 2 B | uint16 | 0-65535 |
  | Anomaly count (24h) | 2 B | uint16 | Predictions above threshold |
  | Mean confidence (24h) | 1 B | uint8, 0-255 → 0.0-1.0 | Average softmax confidence |
  | Confidence histogram | 4 B | 4 bins × 1 byte (counts) | [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0] |
  | Drift indicator | 1 B | uint8 flags | Bit flags: input drift, output drift, sensor fault |
  | Model version | 2 B | uint16 (major.minor packed) | Which model is running |
  | Temperature | 1 B | int8, °C | -128 to +127°C |
  | **Total** | **16 B** | | |

  **Transmission strategy:** One uplink per day with the 16-byte summary. Airtime at SF7: ~41 ms. Duty cycle consumed: 0.041 / 86,400 = 0.000047% — well within the 1% limit. This leaves 99.99% of the duty cycle budget for emergency alerts (e.g., sudden anomaly spike → immediate uplink).

  **Fleet aggregation:** 3,000 devices × 16 bytes/day = 48 KB/day at the network server. A simple dashboard computes fleet-wide metrics: mean accuracy proxy (confidence distribution), drift prevalence, battery health histogram, model version distribution.

  > **Napkin Math:** Daily telemetry: 16 bytes × 1 uplink = 16 bytes/day. Airtime: 41 ms/day. Duty cycle: 0.000047%. Annual airtime: 41 ms × 365 = 15 seconds/year. Energy per uplink: 40 mA TX × 41 ms × 3.3V = 5.4 mJ. Annual telemetry energy: 5.4 × 365 = 1.97 J. On a 3.6V 19 Ah lithium battery (68,400 J): telemetry = 0.003% of battery — invisible. If you sent raw results (480 bytes/day): airtime = 600 ms/day, duty cycle = 0.0007%, energy = 79 mJ/day = 28.8 J/year = 0.04% of battery. Still manageable, but 15× more expensive for minimal extra insight.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> Inference Result Compression for Upload</b> · <code>deployment</code> <code>data-pipeline</code></summary>

- **Interviewer:** "Your fleet of 1,000 wildlife acoustic monitors runs a bird species classifier on an RP2040 (Cortex-M0+, 133 MHz, 264 KB SRAM, 2 MB flash). Each device classifies 10-second audio clips and detects up to 50 species. The devices upload results daily via cellular (LTE-M Cat-M1, billed at $0.50/MB). You're currently sending raw JSON results and the cellular bill is $3,000/month. Compress the upload payload."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use gzip on the JSON." Gzip on a Cortex-M0+ is expensive — the M0+ has no barrel shifter, making the bit manipulation in DEFLATE very slow. And JSON is the wrong format for constrained uploads in the first place.

  **Realistic Solution:** The problem is the data format, not the compression algorithm. Replace JSON with a compact binary protocol:

  **Current format (JSON):**
  ```
  {"ts":1710000000,"species":"AMRO","conf":0.92,"count":3}
  ```
  ~55 bytes per detection. At 100 detections/day per device: 5.5 KB/day. 1,000 devices: 5.5 MB/day = **165 MB/month = $82.50/month**. Wait — the user said $3,000/month. That means they're sending much more: probably the full 50-class softmax vector per clip.

  **Full softmax upload:** 50 classes × 4 bytes (float32) = 200 bytes per clip. At 6 clips/minute × 60 min × 12 hours of daylight = 4,320 clips/day. 4,320 × 200 = 864 KB/day. 1,000 devices: 864 MB/day = **25.9 GB/month = $12,960/month**. That explains the bill.

  **Optimized binary format:**

  (1) **Top-K only** — instead of 50 softmax values, send only the top-3 species per clip. Encoding: 3 × (1 byte species ID + 1 byte confidence as uint8 0-255) = 6 bytes per clip.

  (2) **Temporal aggregation** — instead of per-clip results, aggregate over 1-hour windows. For each hour, send: {hour_id (1B), top-5 species detected (5 × 2B = 10B), total clip count (2B), anomaly flag (1B)} = 14 bytes per hour. 12 hours/day: 168 bytes/day.

  (3) **Daily summary packet:** 168 bytes payload + 8 bytes header (device ID + date) = **176 bytes/day per device**.

  **Fleet-wide:** 1,000 × 176 = 176 KB/day = **5.28 MB/month = $2.64/month**.

  **Savings:** from $12,960/month to $2.64/month = **99.98% reduction**. Even if the original $3,000 figure was with some optimization already applied, the binary aggregation approach reduces it by 3 orders of magnitude.

  **On the M0+:** No compression algorithm needed. The "compression" is semantic — sending summaries instead of raw data. The aggregation logic (tracking top-K species per hour) requires: 50 species × 2 bytes (count) = 100 bytes of SRAM per hour window. Trivial on 264 KB.

  > **Napkin Math:** Raw softmax: 200 B/clip × 4,320 clips/day = 864 KB/day. Top-3 binary: 6 B/clip × 4,320 = 25.9 KB/day (33× reduction). Hourly aggregation: 168 B/day (5,143× reduction). Fleet monthly cost: raw = $12,960, top-3 = $388, aggregated = $2.64. Annual savings: $155,000. SRAM cost of aggregation: 100 bytes. CPU cost: 50 comparisons per clip to update top-K = 50 × 4,320 = 216,000 ops/day at 133 MHz = 1.6 ms/day. The compression is essentially free.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Bootloader A/B Firmware Partitioning</b> · <code>deployment</code> <code>fault-tolerance</code></summary>

- **Interviewer:** "Design the flash memory layout for a Cortex-M4 with 1 MB flash that supports A/B firmware partitioning with rollback. The firmware includes a bootloader, application code, and a TFLite Micro model. The device is deployed in a location where physical access costs $500 per visit."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Split flash 50/50: 512 KB for slot A, 512 KB for slot B." This wastes flash and doesn't account for the bootloader, configuration, or wear leveling.

  **Realistic Solution:** Design the flash layout with every sector accounted for:

  **Flash map (1 MB = 1024 KB):**

  | Region | Size | Address | Purpose |
  |--------|------|---------|---------|
  | Bootloader | 32 KB | 0x0800_0000 | Immutable first-stage bootloader. Never updated OTA. |
  | Boot config | 4 KB | 0x0800_8000 | Active slot pointer, boot count, rollback flag. Wear-leveled. |
  | Slot A (firmware + model) | 480 KB | 0x0800_9000 | Application code (~120 KB) + model weights (~350 KB) |
  | Slot B (firmware + model) | 480 KB | 0x0808_7000 | Mirror of Slot A for updates |
  | Persistent storage | 28 KB | 0x080F_9000 | Calibration data, drift logs, device ID. Survives updates. |

  **Boot sequence:**
  1. Bootloader reads boot config: which slot is active, boot count, rollback flag.
  2. If boot count > 3 (three consecutive failed boots): set rollback flag, switch to other slot, reset boot count.
  3. Jump to active slot. Application increments boot count at start, clears it after successful self-test (inference on golden reference input).
  4. If self-test fails: reboot (boot count increments → eventually triggers rollback).

  **Update sequence:**
  1. Download new firmware+model to inactive slot via FOTA.
  2. Verify CRC-32 of inactive slot.
  3. Write new boot config: set inactive slot as active, reset boot count.
  4. Reboot into new firmware.
  5. New firmware runs self-test. If pass: clear boot count (update confirmed). If fail: reboot → boot count increments → after 3 failures, bootloader reverts.

  **The $500 guarantee:** The device can never be bricked by a bad OTA update. The bootloader is immutable (never updated OTA). The worst case is reverting to the previous working firmware. The only way to brick it is a bootloader bug — which is why the bootloader must be minimal (~2000 lines of C), thoroughly tested, and never updated in the field.

  > **Napkin Math:** Flash overhead for A/B: 32 KB (bootloader) + 4 KB (config) + 28 KB (persistent) = 64 KB overhead. Available per slot: (1024 - 64) / 2 = 480 KB. Model budget per slot: 480 - 120 (app code) - 10 (TFLite Micro runtime) = 350 KB for model weights. At INT8: 350K parameters. Sufficient for most TinyML models (keyword spotting: ~80 KB, person detection: ~300 KB). Boot config wear: 4 KB sector, ~100K erase cycles. At 1 update/week: 100,000 / 52 = 1,923 years before wear-out.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Fleet-Wide Model Update Strategy</b> · <code>deployment</code> <code>deployment</code></summary>

- **Interviewer:** "You manage 100,000 predictive maintenance sensors across 200 factories. The fleet has 5 hardware variants: Cortex-M0+ (nRF52810, 64 KB flash), Cortex-M4 (STM32L4, 1 MB flash), Cortex-M4F (Apollo4, 2 MB flash), Cortex-M33 (nRF5340, 1 MB flash), and ESP32-S3 (8 MB flash). Connectivity is mixed: 40% BLE-only, 35% LoRaWAN, 25% cellular (LTE-M). You need to deploy a retrained anomaly detection model to the entire fleet. Design the update strategy."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Build one model, compile for each target, push to all devices." This ignores that a model fitting in 64 KB flash (nRF52810) is fundamentally different from one using 2 MB flash (Apollo4). You can't deploy the same model to all variants — you need a model family.

  **Realistic Solution:** This is a multi-dimensional logistics problem: model × hardware × connectivity.

  **Step 1: Model family.** Train one base model, then produce 5 target-specific variants:

  | Hardware | Flash budget | Model variant | Size |
  |----------|-------------|---------------|------|
  | nRF52810 (M0+) | 30 KB | INT8, 3-layer, pruned 80% | 28 KB |
  | STM32L4 (M4) | 350 KB | INT8, 8-layer, pruned 50% | 180 KB |
  | Apollo4 (M4F) | 1.5 MB | INT8, 12-layer, full | 420 KB |
  | nRF5340 (M33) | 300 KB | INT8, 8-layer, pruned 60% | 160 KB |
  | ESP32-S3 | 4 MB | INT8, 12-layer, full + ensemble | 800 KB |

  **Step 2: Delta compression.** Compute binary diffs between old and new model for each variant. Typical delta for a retrained model (same architecture, updated weights): 15-25% of full size.

  **Step 3: Connectivity-aware rollout.**

  *Cellular (25K devices):* Push delta updates directly. 25,000 devices × 50 KB avg delta / 50 KB/s LTE-M = 1 second per device. Parallelize across 100 concurrent connections: 25,000 / 100 = 250 batches × 1 s = **4.2 minutes**.

  *BLE (40K devices):* Requires gateway proximity. Each factory has 2-5 BLE gateways. Gateway downloads full delta via Ethernet, then pushes to devices via BLE mesh. 40,000 devices / 200 factories = 200 devices per factory. At 60 KB/s BLE throughput, 50 KB delta: 0.83 s per device × 200 = 166 s per factory. With 3 gateways in parallel: **55 seconds per factory**. All factories in parallel: **55 seconds**.

  *LoRaWAN (35K devices):* The bottleneck. LoRaWAN Class C multicast: 250 B/s effective. 50 KB delta: 200 seconds per multicast group. Devices are grouped by LoRa gateway (typically 500 devices per gateway). 35,000 / 500 = 70 gateways, all multicasting in parallel: **200 seconds = 3.3 minutes**.

  **Step 4: Staged rollout.** Update 1% of each variant first (1,000 devices). Monitor for 24 hours: inference latency, anomaly rate, battery drain, crash rate. If all metrics are within 10% of baseline, proceed with the remaining 99%.

  > **Napkin Math:** Total fleet update time (after staging): max(4.2 min cellular, 55 s BLE, 3.3 min LoRa) = **4.2 minutes** (cellular is the bottleneck due to sequential batching). Add 24-hour staging validation: **24 hours + 4.2 minutes**. Cost: cellular data = 25,000 × 50 KB = 1.25 GB at $0.50/MB = $625. BLE/LoRa: free (local). Total update cost: **$625 + engineering time**. Per device: $0.006. Without delta compression: 25,000 × 250 KB avg = 6.25 GB = $3,125. Delta saves **$2,500 per update cycle**.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Hardware-in-the-Loop Testing</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "Your CI pipeline tests ML models in simulation (x86 QEMU), but you've been burned twice by models that pass simulation and fail on real hardware — once due to CMSIS-NN kernel differences, once due to flash timing. You have 5 hardware variants (Cortex-M0+, M4, M4F, M7, M33). Design a hardware-in-the-loop (HIL) CI system. How many test boards do you need, and what's the test time per commit?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Buy one of each board and run tests sequentially." With 5 boards tested sequentially, each taking 2-3 minutes, your CI feedback loop is 10-15 minutes per commit. Developers will skip HIL tests or ignore failures because the feedback is too slow.

  **Realistic Solution:** Design the HIL farm for parallelism and reliability:

  **Test board fleet:**

  | Variant | Board | Unit cost | Quantity | Purpose |
  |---------|-------|-----------|----------|---------|
  | Cortex-M0+ | nRF52810-DK | $40 | 3 | Minimum viable target, tests tight flash/SRAM |
  | Cortex-M4 | STM32L4-Discovery | $20 | 3 | Primary deployment target |
  | Cortex-M4F | Apollo4 Blue EVB | $50 | 2 | FPU-enabled path, large SRAM |
  | Cortex-M7 | STM32H7-Nucleo | $25 | 2 | High-performance target, TCM testing |
  | Cortex-M33 | nRF5340-DK | $45 | 3 | TrustZone + dual-core testing |
  | **Total** | | | **13 boards** | **$435** |

  3 boards per primary target (M0+, M4, M33) for redundancy — if one board fails, tests still run on the other two. 2 boards for secondary targets (M4F, M7).

  **HIL test pipeline per commit:**

  (1) **Flash firmware (parallel across all boards):** SEGGER J-Link connected to each board via USB to a Raspberry Pi 4 test controller. Flash time: ~1 second per board. All 13 boards flash in parallel: **1 second**.

  (2) **Inference accuracy test:** Run inference on 10 golden test inputs. Compare outputs against x86 reference (bit-exact for INT8, within tolerance for FP32). Time per board: 10 inferences × 50 ms (worst case on M0+) = 500 ms. All boards in parallel: **500 ms**.

  (3) **Latency regression test:** Run 100 inferences, measure P50/P99 latency. Compare against baseline. Flag if P99 regresses by > 5%. Time: 100 × 50 ms = 5 seconds on M0+. All boards in parallel: **5 seconds**.

  (4) **Memory high-water-mark test:** Instrument the tensor arena with a canary pattern. After inference, check how much of the arena was touched. Flag if peak SRAM usage increased. Time: **500 ms** (one instrumented inference + canary check).

  (5) **Power measurement (nightly, not per-commit):** Use Nordic PPK2 on one board per variant. Run 1000 inferences, measure energy per inference. Compare against baseline. Time: 50 seconds per board. Run sequentially (one PPK2 per variant): **250 seconds = 4.2 minutes**.

  **Total per-commit HIL time:** 1 + 0.5 + 5 + 0.5 = **7 seconds** (all boards in parallel). Add CI overhead (checkout, build, flash): ~60 seconds. **Total: ~67 seconds per commit.**

  **Infrastructure:**
  - 13 dev boards: $435
  - 1 Raspberry Pi 4 per 5 boards (USB hub): 3 × $75 = $225
  - 3 SEGGER J-Link EDU: 3 × $60 = $180
  - USB hubs, cables, rack: ~$100
  - **Total: ~$940** — less than one engineer-day of debugging a hardware-specific failure.

  > **Napkin Math:** Per-commit HIL: 67 seconds. At 20 commits/day: 22 minutes of total HIL time. Board utilization: 7 s active / 67 s cycle = 10.4%. Boards are idle 90% of the time — plenty of headroom for parallel branches. Cost of one missed hardware bug (field failure on 10,000 devices): $50 per device visit × 10,000 = $500,000. HIL farm cost: $940. ROI: prevents one field failure = **531× return**. Nightly power test: 4.2 minutes × 5 variants = 21 minutes. Catches power regressions before they reach production.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Unsigned Integer Wrap</b> · <code>deployment</code> <code>adversarial</code></summary>

- **Interviewer:** "Your predictive maintenance system uses a Cortex-M0+ to monitor motor vibrations. It keeps a running tally of anomalies in a `uint16_t` counter and uploads the total to the cloud every week. After 18 months, the cloud dashboard suddenly reports that the factory had exactly 65,500 *fewer* anomalies this week than last week. The factory hasn't changed. What broke?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The ML model drifted and stopped detecting anomalies." Model drift happens slowly; an instant drop of 65,000 points to a data type failure.

  **Realistic Solution:** You suffered an **Integer Overflow (Wrap-Around)**.

  A `uint16_t` (unsigned 16-bit integer) can only hold a maximum value of `65,535`.
  Because your edge device was keeping a *running total* of anomalies over 18 months, the counter slowly crept up.
  When the counter hit `65,535`, the very next anomaly caused the integer to overflow and wrap back to `0`.

  When the device uploaded `0` to the cloud, the cloud dashboard subtracted last week's value (e.g., `65,500`) from this week's value (`0`), determining that there was a massive negative drop in anomalies.

  **The Fix:**
  1. Use wider data types for absolute accumulators (e.g., `uint32_t` holds up to 4.2 billion).
  2. The edge device should *never* send running totals. It should send the *delta* (number of anomalies since the last upload) and clear the counter to zero.

  > **Napkin Math:** 100 anomalies a day. `65535 / 100 = 655 days`. At exactly 1.8 years (18 months), the integer physically runs out of bits and wraps to zero, corrupting your MLOps dashboard.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> Multi-MCU Distributed Inference System</b> · <code>data-parallelism</code> <code>serving</code></summary>

- **Interviewer:** "You need to run a model that requires 80 KB of activation memory on a system with three Cortex-M0+ MCUs, each with only 32 KB SRAM. No single MCU can hold the full activation tensor. Design a distributed inference system that splits the model across the three MCUs, connected via SPI at 8 MHz. Specify the partitioning strategy, communication protocol, and the latency overhead of distribution."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Split the model into 3 equal parts and pipeline them." Equal splitting ignores that the activation memory peak varies by layer — the early layers (high resolution, few channels) have different memory profiles than late layers (low resolution, many channels). You need to split at points where the activation memory fits within 32 KB.

  **Realistic Solution:** Use layer-wise profiling to find optimal split points that minimize both per-MCU memory and inter-MCU communication.

  **(1) Layer-by-layer memory profiling.** Profile the model to find the activation memory at each layer boundary. For a typical CNN (e.g., MobileNetV1 at 64×64 input): Layer 0 output: 64×64×32 = 131 KB (too large for any MCU). Layer 3 output (after stride-2): 32×32×64 = 65 KB (still too large). Layer 5 output (after stride-2): 16×16×128 = 32 KB (fits!). Layer 8 output: 8×8×256 = 16 KB (fits). Layer 11 output: 4×4×512 = 8 KB (fits). Final output: 1×1×1000 = 4 KB (fits).

  **(2) Split strategy.** Split at layer boundaries where the output tensor is ≤32 KB: MCU 1: layers 0–5 (input → 16×16×128 output). Peak activation: needs input (64×64×3 = 12 KB) + largest intermediate (64×64×32 = 128 KB). Still too large! The early layers' activations exceed 32 KB.

  **(3) Revised strategy: patch-based inference on MCU 1.** Process the 64×64 input as four 32×32 patches on MCU 1. Each patch: 32×32×3 = 3 KB input. After 2 stride-2 layers: 8×8×128 = 8 KB output per patch. Peak activation per patch: ~20 KB. Fits in 32 KB. MCU 1 processes 4 patches sequentially, producing 4 × 8×8×128 = 32 KB total output (reassembled into 16×16×128). Transfer 32 KB to MCU 2 via SPI.

  **(4) MCU 2: layers 6–8.** Input: 16×16×128 = 32 KB (exactly fits). Output: 8×8×256 = 16 KB. Peak activation during computation: 32 KB input + 16 KB output = 48 KB — doesn't fit! Solution: process in two 16×16×64 channel slices. Each slice: 16 KB input, 8 KB output, 24 KB peak. Fits. Transfer 16 KB to MCU 3.

  **(5) MCU 3: layers 9–end.** Input: 8×8×256 = 16 KB. Output: 1×1×1000 = 4 KB. Peak: ~20 KB. Fits easily.

  **(6) Communication overhead.** SPI at 8 MHz, 8-bit mode = 1 MB/s. MCU 1 → MCU 2: 32 KB / 1 MB/s = **32ms**. MCU 2 → MCU 3: 16 KB / 1 MB/s = **16ms**. Total communication: **48ms**. Compute per MCU: ~5ms each. Total inference: 4×5ms (MCU 1 patches) + 32ms (transfer) + 2×5ms (MCU 2 slices) + 16ms (transfer) + 5ms (MCU 3) = **73ms**.

  > **Napkin Math:** Single MCU (if it had 80 KB SRAM): ~15ms inference. Distributed (3 MCUs): 73ms. Overhead: 4.9× (communication dominates). SPI transfer: 48ms / 73ms = 66% of total time spent on communication. Optimization: use SPI DMA to overlap transfer with computation on the next MCU. MCU 2 starts computing as soon as the first channel slice arrives (16 KB / 1 MB/s = 16ms). While MCU 2 computes slice 1 (5ms), MCU 1 sends slice 2 (16ms). Pipelined: total reduces to ~55ms. Still 3.7× slower than a single MCU, but enables models that physically cannot fit on one device. Alternative: use a single Cortex-M4 with 128 KB SRAM ($2 more) and avoid the distributed complexity entirely. The engineering cost of distributed TinyML rarely justifies the BOM savings.

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Bricked OTA Update</b> · <code>memory-hierarchy-ota</code></summary>

- **Interviewer:** "You are the firmware architect for a new battery-powered smart lock that uses a keyword spotting model to activate. Your MCU has 1MB of Flash and 256KB of SRAM. The product ships with a 350KB model. Six months post-launch, the ML team develops a new, 450KB model that is much more accurate. Your team attempts to deploy this model via an Over-the-Air (OTA) update. The update process downloads the new model, but the devices start crashing and failing the update, effectively bricking them in the field. Your bootloader, RTOS, and application logic consume 150KB of Flash. Propose a complete memory architecture and update process from scratch for V2 of this product that guarantees safe, atomic OTA updates for future model deployments. Justify your partitioning of Flash and prove your design won't exhaust SRAM during the update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Focusing only on Flash capacity, assuming SRAM is plentiful. A common mistake is to propose buffering the entire new model in SRAM before writing to Flash, which is impossible. Another is failing to account for the memory required by the *currently running* old model and RTOS during the update itself.

  **Realistic Solution:** The core problem is managing three concurrent memory demands: Flash for storage, SRAM for execution (of the old model), and SRAM for the update payload. A robust L6+ design implements A/B partitioning for the model in Flash.

1.  **Flash Partitioning:** The 1MB Flash must be divided. You can't fit Bootloader (32K) + RTOS/App (150K) + Old Model (350K) + New Model (450K) = 982KB, leaving only 42KB for anything else, which is too tight. The critical insight is that the app itself doesn't need two copies. A better layout is: Bootloader (32KB), App State/Config (32KB), RTOS/App (150KB), Model Partition A (450KB), Model Partition B (450KB). Total = 1114KB. This exceeds the 1MB Flash. The candidate must recognize the constraint is impossible as stated and propose a new MCU with 2MB flash or a smaller model. Assuming a 2MB Flash was approved:

2.  **Atomic Update Process:** The bootloader is the key. On boot, it checks a flag in a dedicated config sector to see which partition (A or B) is 'active'. It then jumps to the application, which loads the model from that active partition. The OTA update process, running in the main app, receives the new model and writes it to the *inactive* partition. It only flips the 'active' flag in the config sector *after* the new model has been fully downloaded and its integrity verified. If the device reboots at any point, the bootloader will still load the old, valid model.

3.  **SRAM Management:** This is the crucial part. You cannot buffer the entire 450KB model in the 256KB SRAM. The update must be streamed. The device receives the update in small chunks (e.g., 4KB), buffers just that single chunk in SRAM, writes it to the inactive Flash partition, verifies the write, and then requests the next chunk. This keeps the SRAM overhead for the update minimal.

  > **Napkin Math:** Let's prove the SRAM budget during an update. Total SRAM: 256KB.
- **RTOS + App:** Let's budget a generous 80KB for the RTOS kernel, application threads, and heap.
- **Old Model Execution:** The currently running 350KB model needs a tensor arena. Let's assume its peak arena size is 120KB. The device must remain functional during the download.
- **OTA Update Buffer:** We stream the update in 4KB chunks. So we need a 4KB buffer.
- **Peak SRAM Usage:** `80KB (RTOS/App) + 120KB (Old Model Arena) + 4KB (Update Chunk Buffer) = 204KB`.
- **Conclusion:** This peak usage of 204KB is well within the 256KB SRAM limit. The flawed approach of buffering the whole model would require `80KB + 120KB + 450KB = 650KB`, which would instantly crash the device.

  > **Key Equation:** $\text{SRAM}_{\text{peak}} = \text{SRAM}_{\text{RTOS}} + \text{SRAM}_{\text{Arena}(\text{old})} + \text{SRAM}_{\text{buffer}(\text{chunk})} < \text{SRAM}_{\text{total}}$

  📖 **Deep Dive:** [TinyML: Deployed Device](https://mlsysbook.ai/tinyml/03_deployed_device.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The OTA Update Brickening</b> · <code>tinyml-ota-memory-fragmentation</code></summary>

- **Interviewer:** "You are the architect for a fleet of 100,000 IoT environmental sensors. After pushing a new ML model via your Over-the-Air (OTA) update system, 10% of the devices are 'bricked' and unresponsive. The MCU is a Cortex-M4 with 1MB Flash and 256KB SRAM. Your OTA design uses a dual-partition scheme ('App A' / 'App B'). The bootloader successfully swaps to the new application, but it hangs soon after. Your investigation reveals the new model requires a 140KB tensor arena, an increase from 120KB in the previous version. The `tflite::MicroInterpreter::AllocateTensors()` call is failing. Formulate a hypothesis for why a seemingly small 20KB increase in memory requirement is causing catastrophic failure. Propose a set of architectural changes to both your OTA process and your application's memory management to ensure future updates are 100% reliable. How would you have caught this issue before it was ever deployed?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common incorrect analysis is to simply sum the memory requirements (`140KB_arena + RTOS_heap < 256KB_total`) and conclude it should fit. This ignores the critical concept of memory fragmentation. A junior engineer might suggest simply increasing the heap size, which doesn't solve the underlying problem of obtaining a large *contiguous* block of memory after the heap has been used by other parts of the system, especially the OTA download manager itself.

  **Realistic Solution:** The root cause is the failure to acquire a single, contiguous 140KB memory block from the system's general-purpose heap. While the total *free* memory might be sufficient (e.g., 150KB), the heap becomes fragmented over the device's uptime and especially during the OTA process, which allocates and frees buffers for radio packets. This leaves many small 'holes' in the heap, with no single block large enough for the new tensor arena. The `AllocateTensors` call fails, and since the return value was not checked, the program proceeds to use a null pointer, causing a hard fault.

**Architectural Hardening:**
1.  **Static Arena Allocation:** The fundamental fix is to stop using the dynamic heap (`malloc`) for the tensor arena. The arena should be declared as a large, static, compile-time array (e.g., `static uint8_t g_tensor_arena[140 * 1024] __attribute__((aligned(16)));`). This reserves the memory at link time, guaranteeing its existence and contiguity. The linker will now be the gatekeeper; if the total static memory exceeds the device's SRAM, the firmware will fail to build, preventing the faulty image from ever being created. This is the single most important best practice.
2.  **Robust Bootloader with Rollback:** A production OTA system must be fault-tolerant. The bootloader should incorporate a watchdog mechanism. After swapping to the new application, it starts a timer. The new application has a limited time (e.g., 10 seconds) to perform a self-test (including tensor allocation) and 'check in' with the bootloader to confirm it is healthy. If the check-in fails, the watchdog reboots the device, and the bootloader automatically rolls back to the previous known-good application partition. The device then reports the rollback event to the cloud for debugging.
3.  **Explicit Memory Budgeting:** The project must maintain a 'memory manifest' (e.g., a shared header or document) that explicitly budgets the SRAM usage for all components: RTOS kernel, task stacks, DMA buffers, and the ML tensor arena. Any change to this budget, like the 20KB arena increase, would trigger an architectural review. This formalizes memory as a constrained resource.

  > **Napkin Math:** This is a problem of accounting and layout, not complex formulas.

1.  **Total SRAM:** 256 KB.
2.  **System Static Allocation:**
    *   RTOS Kernel: ~16 KB
    *   Network Stack (static portion): ~24 KB
    *   Task Stacks (4 tasks × 4KB each): 16 KB
    *   **Sub-total:** 56 KB
3.  **Available for Heap/Arena:** 256 KB - 56 KB = 200 KB.
4.  **Scenario A (Old Model):** A 120KB arena allocation from a 200KB space is likely to succeed even with mild fragmentation.
5.  **Scenario B (New Model):** A 140KB arena allocation is required. During OTA, the download manager might allocate a 32KB radio buffer from the heap. Now the largest possible block is `200 - 32 = 168KB`. After the download is done and the buffer is freed, the heap is fragmented. If another small allocation occurs, the 140KB request can easily fail.

**Catching it Pre-Deployment:** The static allocation approach catches this at compile time via a linker error. A memory budget review process would have flagged the `120KB -> 140KB` change, forcing an analysis that would show the remaining `256 - 56 - 140 = 60KB` for all other dynamic needs is a high risk.

  > **Key Equation:** \sum(\text{Static Alloc}) + \max(\text{Peak Dynamic Alloc}) < \text{Total SRAM}

  📖 **Deep Dive:** [TinyML](https://mlsysbook.ai/tinyml/03_deployed_device.html)
  </details>
</details>




---


### Networking & Connectivity


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cellular NAT Timeout</b> · <code>interconnect</code> <code>deployment</code></summary>

- **Interviewer:** "Your IoT device connects to AWS IoT Core via MQTT over an LTE-M cellular connection. It sends an ML telemetry payload perfectly upon booting. It then sits idle for 30 minutes. When the next anomaly occurs, the `mqtt_publish()` function claims success, but the message never arrives at AWS. The device didn't sleep and the cellular signal is perfect. Why did the network silently swallow the message?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The MQTT broker crashed." AWS IoT Core doesn't crash; the path to it was severed.

  **Realistic Solution:** You hit the **Cellular Carrier NAT (Network Address Translation) Timeout**.

  Cellular networks do not give your device a public IP address. They put you behind a massive Carrier-Grade NAT. The NAT router maintains a state table linking your device's internal IP to the external AWS server's IP.

  To save memory on their expensive routers, cellular carriers aggressively purge idle connections. If no data flows through the TCP socket for a short period (often as little as 2 to 5 minutes), the carrier silently deletes the routing table entry.

  Your device thinks the TCP socket is still perfectly open. AWS thinks the device gracefully disconnected. When your device finally sends data 30 minutes later, the packet hits the carrier's NAT router, the router has no idea where it belongs, and silently drops it.

  **The Fix:** You must configure the MQTT **Keep-Alive Interval** to be strictly shorter than the carrier's NAT timeout (e.g., set MQTT Keep-Alive to 60 or 120 seconds). The device will periodically send a tiny 2-byte PINGREQ packet, which resets the carrier's NAT timer and keeps the TCP tunnel physically open.

  > **Napkin Math:** LTE-M NAT Timeout = ~300 seconds. ML Anomaly Interval = 1800 seconds. 1800 > 300. The TCP connection is guaranteed to be dead every single time the device tries to use it.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
  </details>
</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The OTA Bandwidth Congestion</b> · <code>interconnect</code> <code>deployment</code></summary>

- **Interviewer:** "You have a fleet of 5,000 smart factory sensors connected via a shared LoRaWAN gateway. You push a 100 KB model update to the fleet simultaneously. The OTA update process stalls, taking days to complete, and normal sensor telemetry stops functioning entirely. What network characteristic of LoRaWAN did you violate?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "100 KB is too large for the gateway." While true, it's not just the size; it's the collision domain and the protocol duty cycle.

  **Realistic Solution:** You violated the **Duty Cycle Limits and the ALOHA MAC protocol**.

  LoRaWAN operates in unlicensed sub-GHz bands (like 868 MHz or 915 MHz). By law in many regions, a device can only transmit for 1% of the time (the duty cycle limit).

  Furthermore, LoRa uses a modified ALOHA protocol. Devices just "shout" their data into the air. If 5,000 devices are all trying to send acknowledgment packets (ACKs) for the OTA chunks they are receiving at the exact same time, the radio waves collide in the air. The gateway receives garbage. The devices wait, timeout, and retry... causing even more collisions. This is a **Broadcast Storm**.

  Your OTA update effectively DDOS'd your own factory network.

  **The Fix:**
  1. Use **Multicast OTA (FUOTA - Firmware Update Over The Air)**. The gateway broadcasts the firmware chunks once, and all 5,000 devices listen simultaneously without sending individual ACKs for every packet. They only request missing packets at the very end.
  2. If Multicast isn't available, you must strictly stagger the updates (e.g., updating only 10 devices an hour) to prevent airwave congestion.

  > **Napkin Math:** In LoRa SF12, a 51-byte payload takes ~2.5 seconds of airtime. A 1% duty cycle means the device must remain completely silent for the next 247 seconds before it can send an ACK. Sending 100 KB point-to-point to 5,000 devices is mathematically impossible under these physics.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>

---


### Security & Privacy


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The MCU Model Extraction Attack</b> · <code>security</code></summary>

- **Interviewer:** "Your company deploys a proprietary defect detection model on an STM32F4 MCU inside an industrial inspection camera. A competitor buys your product, connects a JTAG debugger to the exposed debug header, and dumps the entire Flash memory — including your model weights — in under 60 seconds. How do you protect the model on a $3 MCU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Encrypt the model in Flash and decrypt at runtime." On an MCU with 256 KB SRAM and a 500 KB model in Flash, you can't decrypt the entire model into SRAM — it doesn't fit. Decrypting layer-by-layer adds latency and the decryption key must be stored *somewhere* on the same chip.

  **Realistic Solution:** Defense-in-depth using the MCU's hardware security features:

  (1) **Read-out protection (RDP)** — the STM32F4 has three RDP levels. RDP Level 1: JTAG/SWD can connect but cannot read Flash. RDP Level 2: JTAG/SWD is permanently disabled — the debug port is fused off. Level 2 is irreversible (hardware fuse). This blocks the trivial JTAG dump attack. Cost: $0 (just set an option byte).

  (2) **Proprietary code readout protection (PCROP)** — STM32F4 supports PCROP on specific Flash sectors. Mark the sectors containing model weights as PCROP-protected. Even if an attacker downgrades from RDP Level 2 (impossible, but hypothetically), PCROP sectors return zeros on read. The CPU can *execute* from these sectors but cannot *read* them as data — but model weights are data, not code. Solution: store weights in PCROP sectors and use a small trusted loader that copies weights to SRAM sector-by-sector during inference, erasing each SRAM sector after use.

  (3) **Physical attack mitigation** — a determined attacker can decap the chip and use a focused ion beam (FIB) to read Flash cells directly. Defense: use the STM32's hardware AES-256 engine to encrypt model weights in Flash with a key derived from the device's unique ID (96-bit UID). Each chip has a different key. Decapping one chip doesn't help with another. The AES engine decrypts at hardware speed (~1 cycle/byte at 168 MHz) with negligible latency impact.

  (4) **Accept the economics** — a FIB attack costs $50,000-$100,000 per chip. If your model's competitive advantage is worth less than this, RDP Level 2 + AES encryption is sufficient. If it's worth more, consider a secure element (e.g., ATECC608B, $0.50) to store the decryption key in tamper-resistant silicon.

  > **Napkin Math:** JTAG dump without protection: 60 seconds, $0 cost (just a $20 ST-Link). RDP Level 2: blocks JTAG entirely. Decap + FIB: $50K-$100K, 2-4 weeks. AES decryption overhead: 500 KB model / 168 MB/s = 3 ms (one-time at boot). SRAM budget for layer-by-layer decryption: largest layer = 40 KB. Fits in 256 KB SRAM with room for inference. Secure element cost: $0.50 per unit × 100K units = $50K — same as one FIB attack.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Extraction Attack</b> · <code>security</code></summary>

- **Interviewer:** "An attacker has physical access to your deployed MCU. They want to extract your proprietary model weights from flash memory. How can power side-channel analysis extract model weights by correlating power traces with MAC operations, and why does the model's arithmetic structure make this ML-specific attack possible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Flash memory is internal to the MCU — it can't be read externally." Or "Just enable Read-Out Protection (RDP)." RDP blocks JTAG debuggers, but side-channel attacks bypass digital locks entirely.

  **Realistic Solution:** The attacker doesn't need to read the flash directly. They connect a high-resolution current probe to the MCU's power rail and record the power consumption while feeding known inputs to the ML model.

  This attack exploits the fundamental arithmetic structure of neural networks. During inference, the MCU executes millions of Multiply-Accumulate (MAC) operations. The power consumed by the ALU during a multiplication $w \times x$ depends on the Hamming weight (number of '1' bits) of the operands. Because the attacker knows the input $x$, they can use statistical methods (like Correlation Power Analysis) across thousands of inference traces to guess the weight $w$. They hypothesize a weight value, simulate the expected power draw for the known inputs, and correlate it with the measured power trace. The value with the highest correlation is the true weight. They repeat this layer by layer.

  **Defense:** (1) **Weight Masking:** XOR the weights with a random mask before storage, and unmask them dynamically during inference using a hardware random number generator (TRNG). (2) **Dummy Operations:** Insert random dummy MAC operations into the inference loop to desynchronize the power trace. (3) **Execution Jitter:** Randomly vary the MCU clock speed or insert random delays between layers to misalign the attacker's traces.

  > **Napkin Math:** A Cortex-M4 executing a `SMLAD` instruction draws ~20 mA. The difference in current between multiplying 0x0000 and 0xFFFF might be just 50 µA. A 1 GS/s oscilloscope can capture this micro-variation. With ~10,000 inference traces (which takes just a few minutes to collect at 10 Hz inference rate), the signal-to-noise ratio is high enough to extract an entire 8-bit weight matrix with >99% accuracy. Defenses like dummy operations add ~10-20% performance overhead but reduce the SNR so severely that the attacker would need millions of traces, making the attack economically unviable.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Secure Boot Chain for ML Models</b> · <code>security</code> <code>deployment</code></summary>

- **Interviewer:** "Your company ships a medical wearable running a cardiac arrhythmia detection model on an STM32U5 (Cortex-M33 with TrustZone, 160 MHz, 2 MB flash, 786 KB SRAM). Regulatory compliance (IEC 62443) requires that only authenticated firmware and models can execute on the device. An attacker who gains physical access must not be able to replace the model with a malicious one. Design the secure boot chain."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Enable Secure Boot and sign the firmware." This protects the firmware but not the model weights, which are stored separately in flash. An attacker could replace the model weights (e.g., a model that always reports "normal rhythm") without touching the signed firmware, and the device would boot successfully with a compromised model.

  **Realistic Solution:** The secure boot chain must cover every executable and data component:

  **Boot chain (4 stages):**

  (1) **ROM bootloader (immutable, in silicon)** — STM32U5's built-in secure boot ROM. Verifies the hash of the first-stage bootloader against a value burned into OTP (One-Time Programmable) fuses. Cannot be modified by software. If verification fails: device halts (no boot).

  (2) **First-stage bootloader (32 KB, in secure flash)** — runs in TrustZone Secure World. Holds the RSA-2048 public key (256 bytes). Verifies the signature of the application firmware: computes SHA-256 hash of the firmware region, then verifies the RSA-2048 signature (stored in the last 256 bytes of the firmware slot) against the public key.

  (3) **Application firmware (verified)** — before loading the model, computes SHA-256 of the model weights region and verifies against a signed model manifest (hash + RSA signature, stored in a protected flash page). This ensures the model hasn't been tampered with independently of the firmware.

  (4) **Runtime integrity** — periodically (every 1000 inferences), re-hash a random 4 KB page of the model weights and compare against the stored hash. This detects runtime flash corruption or fault-injection attacks that modify weights after boot.

  **RSA-2048 verification cost on Cortex-M33 at 160 MHz:**
  RSA-2048 signature verification (modular exponentiation with e=65537) requires ~30 million cycles on a Cortex-M33 without hardware crypto. At 160 MHz: 30M / 160M = **187 ms**. With STM32U5's PKA (Public Key Accelerator): ~5 million cycles = **31 ms**.

  SHA-256 of 500 KB firmware+model: 500,000 × 15 cycles/byte = 7.5M cycles = **47 ms** (software) or **12 ms** (with HASH peripheral).

  **Total secure boot time:** SHA-256 (12 ms) + RSA verify (31 ms) + model hash (8 ms) + model RSA verify (31 ms) = **82 ms** with hardware acceleration, **280 ms** without. Acceptable for a device that boots once and runs for months.

  > **Napkin Math:** Boot time budget: 82 ms (with HW crypto) or 280 ms (SW only). Flash overhead: 256 B RSA signature per firmware slot + 256 B per model slot + 32 B SHA-256 hash per model = 544 bytes. Key storage: 256 B public key in secure OTP. Runtime integrity check: SHA-256 of 4 KB page = 4,096 × 15 / 160M = 0.38 ms every 1000 inferences. At 1 inference/second: 0.38 ms / 1000 s = 0.00004% CPU overhead. Attack cost to bypass: requires extracting the private key (stored only on the signing server, never on the device) or finding a SHA-256 collision (2¹²⁸ operations — infeasible).

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


---


### Additional Topics


#### 🟢 L1/L2

#### 🟢 L3
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The MRAM Wear Illusion</b> · <code>persistent-storage</code> <code>deployment</code></summary>

- **Interviewer:** "You switch from traditional SPI Flash to an external MRAM (Magnetoresistive RAM) chip to store telemetry. MRAM is famous for essentially infinite write endurance. You write your ML logs continuously in a tight loop. A year later, the MRAM chip starts returning corrupted bits. If MRAM doesn't wear out like Flash, why did it fail?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "MRAM has a limited cycle count just like Flash." MRAM cycle counts are practically infinite (10^14 cycles), it shouldn't wear out in a year.

  **Realistic Solution:** You fell victim to **Write Endurance vs. Data Retention Trade-offs**.

  While MRAM does not suffer from dielectric breakdown like Flash memory, it is heavily susceptible to thermal energy and continuous magnetic disturbance.

  If you write to the exact same memory cells millions of times a second without pause, the localized heating and continuous magnetic tunneling currents can physically destabilize the magnetic orientation of adjacent cells (similar to Rowhammer in DRAM, but magnetic).

  Furthermore, many cheaper MRAM chips promise "infinite" endurance only if the ambient temperature is tightly controlled. In a hot industrial environment, the magnetic states naturally degrade over time (Data Retention failure), which is accelerated by constant writing.

  **The Fix:** Even with MRAM, you should still implement basic wear-leveling (ring buffers) to distribute the localized thermal and magnetic stress across the entire silicon die, rather than hammering address `0x00` infinitely.

  > **Napkin Math:** 10^14 writes is massive, but at 1 million writes per second (tight C loop), you hit 10^14 in about 3.1 years. If you hammer one address, you can absolutely wear out MRAM.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
  </details>
</details>


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Wear-Leveling Blindspot</b> · <code>persistent-storage</code> <code>deployment</code></summary>

- **Interviewer:** "Your edge sensors log anomaly data to internal Flash. To prevent wearing out the Flash (which has a 10,000 cycle limit), you write a script to always save logs starting at memory address 0x08000000, and sequentially move forward to 0x08040000 before looping back. After a year, the system crashes because the flash sector at 0x08000000 is physically destroyed. Why didn't your sequential logging work as wear-leveling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "You didn't make the loop big enough." The size of the loop isn't the primary failure mode; it's how Flash physics requires data to be updated.

  **Realistic Solution:** You ignored **Flash Page Erase Granularity**.

  You can write (program) bits in Flash from 1 to 0 sequentially. But you cannot flip a 0 back to a 1 without **erasing an entire sector/page** at once.
  If your microcontroller's flash sector size is 16 KB, and you write 100 bytes of logs sequentially into that sector, you eventually fill the 16 KB. To write the 16,001st byte, you must erase the *entire* 16 KB sector.

  Your script looped through the memory, but every time it looped back to 0x08000000, it had to issue an Erase command on Sector 0. If you log frequently, Sector 0 absorbs massive amounts of Erase cycles (which is what physically destroys the silicon) while the rest of the memory space might remain lightly used.

  **The Fix:** Never write raw Flash management code yourself. Use a proper **Flash Translation Layer (FTL)** or an embedded filesystem designed for flash (like LittleFS or SPIFFS). These libraries abstract the physical addresses and automatically map logical writes to different physical sectors to ensure perfect, even wear-leveling across the entire chip.

  > **Napkin Math:** If you log 64 bytes a minute, a 16 KB sector fills in 256 minutes (~4.2 hours). You are erasing that sector 5.6 times a day. 5.6 erases * 365 days = 2,044 erase cycles per year. The flash will die in roughly 4.8 years.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Logging Flash Death</b> · <code>persistent-storage</code> <code>deployment</code></summary>

- **Interviewer:** "Your smart thermostat logs the room temperature and the ML model's occupancy prediction to internal Flash memory every 5 minutes for user analytics. You are using a standard SPIFFS filesystem. The internal flash has a 10,000 cycle erase limit. A year later, 15% of the devices are permanently bricked. How did logging 20 bytes every 5 minutes destroy the Flash?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "20 bytes * 105,000 logs a year is only 2 MB. It didn't fill up the drive." It didn't fill up the drive, it destroyed the silicon via Write Amplification.

  **Realistic Solution:** You fell victim to **Write Amplification and Erase Granularity**.

  Flash memory cannot overwrite a 0 to a 1 without erasing an entire "Sector" (usually 4 KB or 16 KB).
  When you append 20 bytes to a log file, the filesystem (even flash-aware ones like SPIFFS) eventually fills up a sector. To write the next 20 bytes, it must find a new sector, or erase an old one.

  Because the filesystem also has to update metadata (the file size, the directory index, the wear-leveling headers) for *every single 20-byte write*, it is constantly erasing and rewriting the metadata sectors. You are effectively performing a 4 KB erase cycle just to save 20 bytes of data.

  **The Fix:** Never log high-frequency telemetry synchronously to internal Flash.
  1. Buffer the 20-byte logs in SRAM (RAM has infinite endurance).
  2. Only write to Flash once a day, or when the SRAM buffer hits 4 KB, ensuring you only trigger one Erase cycle per 4 KB of actual payload data.

  > **Napkin Math:** 1 log every 5 mins = 288 logs/day. If each log updates a metadata sector, that sector is erased 288 times a day. 288 * 365 = 105,120 erase cycles per year. The Flash is rated for 10,000 cycles. You physically destroyed the silicon in exactly 34 days.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Federated Learning on Constrained Devices</b> · <code>data-parallelism</code> <code>deployment</code></summary>

- **Interviewer:** "You manage 10,000 vibration sensors on factory equipment. Each sensor runs anomaly detection on a Cortex-M4. After 6 months, the model drifts because equipment ages and vibration patterns change. You want to update the model using data from the fleet — but you can't upload raw sensor data (proprietary manufacturing data, 100 TB total). Can you do federated learning on MCUs?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run standard federated averaging — each device trains locally and sends gradients." Standard FL requires backpropagation, which needs: (1) storing all activations for the backward pass (2× the forward-pass memory), (2) float32 arithmetic for stable gradients, (3) an optimizer state (momentum, Adam state). None of this fits on a Cortex-M4 with 256 KB SRAM.

  **Realistic Solution:** Standard federated learning is infeasible on MCUs. But there are MCU-compatible alternatives:

  **(1) Federated fine-tuning of the last layer only.** Freeze all layers except the final classification head. The head is a small fully-connected layer (e.g., 64 → 4 classes = 256 weights). Fine-tuning only the head requires: storing 64 activations from the penultimate layer (64 bytes INT8), computing gradients for 256 weights (1 KB in FP16), and running a simple SGD update (no momentum, no Adam). Total memory: ~2 KB. This fits easily on any MCU. Each device fine-tunes its head on local data, then sends the 256 updated weights (512 bytes in FP16) to the server via BLE/LoRa. The server averages the weights from all devices and broadcasts the updated head.

  **(2) Federated feature statistics.** Instead of training on-device, each device computes running statistics of its penultimate-layer activations (mean and covariance per class). These statistics (64-dim mean + 64×64 covariance = ~8 KB per class) are uploaded to the server. The server uses these to retrain the classification head in the cloud, then pushes the updated head to all devices via FOTA. No on-device training required.

  **(3) Federated distillation.** Each device runs inference on its local data and uploads the model's soft predictions (probability distributions, not raw data). The server trains a new model using these soft labels as supervision (knowledge distillation). The new model is pushed to devices via FOTA. Privacy-preserving: soft predictions leak less information than raw data.

  **Communication budget:** 10,000 devices × 512 bytes (head weights) = 5 MB per round. Via BLE to gateways: 5 MB / 60 KB/s per gateway / 5 gateways (2000 devices per gateway) = 17 seconds per round. Via LoRaWAN: 5 MB / 250 B/s multicast = 20,000 seconds ≈ 5.5 hours per round (LoRa is the bottleneck).

  > **Napkin Math:** Full FL (infeasible): backward pass memory = 2× forward = 400 KB. Doesn't fit in 256 KB. Optimizer state (Adam): 2× model size = 400 KB. Doesn't fit. Head-only FL: head weights = 256 × 2 bytes = 512 bytes. Gradient computation: 64 activations × 4 classes × 2 bytes = 512 bytes. SGD update: 512 bytes. Total: ~1.5 KB. Fits in 256 KB with 254.5 KB to spare. Accuracy recovery: head-only fine-tuning typically recovers 60-80% of the drift-induced accuracy loss. Full model retraining (in the cloud, using federated statistics) recovers 90-95%.

  📖 **Deep Dive:** [Volume I: Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> TinyML Federated Learning System</b> · <code>data-parallelism</code> <code>deployment</code></summary>

- **Interviewer:** "You have a fleet of 10,000 smart electricity meters, each with a Cortex-M4F (256 KB SRAM, 1 MB flash, 168 MHz) and a LoRaWAN radio (250 bps effective throughput after duty cycle limits). Each meter runs a load forecasting model (2-layer LSTM, 15 KB weights) that predicts next-hour consumption. After 1 year, the model has drifted because consumer behavior changed (more EVs, more solar panels). Design a federated learning system that retrains the model across the fleet using LoRaWAN's extreme bandwidth constraints."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run standard FedAvg — upload gradients via LoRaWAN." At 250 bps, uploading 15 KB of gradients takes 15,000 × 8 / 250 = 480 seconds = 8 minutes per device. With LoRaWAN duty cycle limits (1% in EU868): actual airtime = 8 min / 0.01 = 800 minutes = **13.3 hours** per device per round. For 10,000 devices: the gateway can handle ~100 devices per hour. One round takes 100 hours = 4.2 days. 20 rounds to convergence: **84 days**. Unacceptable.

  **Realistic Solution:** Extreme compression is mandatory. The communication constraint drives every design decision.

  **(1) On-device training.** The M4F can train a 2-layer LSTM (15 KB weights) using backpropagation through time (BPTT). Training data: last 7 days of hourly readings = 168 samples × 10 features × 4 bytes = 6.7 KB. Forward pass: 50K MACs × 168 time steps = 8.4M MACs → 25ms on M4F with CMSIS-NN. Backward pass: ~3× forward = 75ms. Per epoch: 100ms. 10 epochs: 1 second. Memory: weights (15 KB) + gradients (15 KB) + activations (5 KB) + training data (6.7 KB) + optimizer state (15 KB for Adam) = **57 KB**. Fits in 256 KB SRAM with 199 KB to spare.

  **(2) Extreme gradient compression.** After local training, compute weight delta (new − old). Apply: (a) Top-k sparsification: keep only top 0.1% of deltas = 15 values (indices + values). (b) Quantize delta values to INT8: 15 × (2 bytes index + 1 byte value) = **45 bytes**. (c) LoRaWAN transmission: 45 bytes at 250 bps = 1.44 seconds airtime. With 1% duty cycle: 144 seconds = 2.4 minutes per device. Gateway capacity: ~25 devices per hour. 10,000 devices / 25 = 400 hours per round? No — use multiple gateways (typical LoRaWAN deployment: 1 gateway per 1,000 devices). 10 gateways: 40 hours per round.

  **(3) Aggregation.** The server receives 45-byte sparse deltas from each device. Reconstruct the full gradient by accumulating sparse updates. With 10,000 devices each contributing 15 non-zero deltas: 150,000 updates across 15,000 parameters → each parameter gets ~10 updates on average. Aggregate via weighted mean. Download the updated model: 15 KB, but only send the delta (also sparse): ~45 bytes. Round-trip per device: 90 bytes.

  **(4) Convergence.** With 0.1% sparsification, convergence is slower: ~50 rounds (vs 20 for full gradients). But each round is 40 hours. Total: 50 × 40 = 2,000 hours = **83 days**. Still slow, but this is a background process — the existing model continues serving predictions while retraining happens. After 83 days, the fleet has a model adapted to the new consumption patterns.

  **(5) Practical optimization.** Don't train all 10,000 devices every round. Sample 500 devices (5%) per round. Each round: 500 / 25 per gateway per hour / 10 gateways = 2 hours. 50 rounds × 2 hours = **100 hours = 4.2 days**. Much more practical. The sampling introduces variance but with 500 devices, the gradient estimate is statistically robust.

  > **Napkin Math:** Per-device upload: 45 bytes (0.1% sparse INT8 delta). LoRaWAN airtime: 1.44s (within single-packet limit of 51 bytes for SF7). Duty cycle: 1.44s / 0.01 = 144s between transmissions. Training compute: 1 second on M4F. Training energy: 1s × 30 mW = 30 mJ. LoRa TX energy: 1.44s × 100 mW = 144 mJ. Total per round: 174 mJ. Battery impact (if battery-powered, 3,000 mAh × 3.3V = 9.9 Wh = 35,640 J): 50 rounds × 174 mJ = 8.7 J = 0.024% of battery. Negligible. The extreme compression makes federated learning viable even on LoRaWAN.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>
