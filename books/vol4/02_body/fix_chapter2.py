import re

with open('/private/tmp/MLSysBook-dev-notation/books/vol4/02_body/02_body.qmd', 'r') as f:
    text = f.read()

# 1. DC -> direct current (DC)
text = text.replace('DC bus voltages never sag', 'direct current (DC) bus voltages never sag')

# 2. back-EMF -> back-EMF (electromotive force, the voltage opposing current change)
text = text.replace('against motor back-EMF', 'against motor back-EMF (electromotive force, the voltage opposing current change)')

# 3. \mathbf{\Momentum} -> \mathbf{x}
text = text.replace(r'\mathbf{\Momentum}(t)', r'\mathbf{x}(t)')
text = text.replace(r'\Delta \mathbf{\Momentum}', r'\Delta \mathbf{x}')

# 4. Remove margin pointer block at line 262
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ The same deadline applied to a stored belief, which lapses once its region falls out of view, is developed in the memory stage\.\n:::\n\n?', '', text)

# 5. pre-brake delay travel inline
text = text.replace('Observation age then converts directly into unguided vehicular travel before braking begins', 'Observation age then converts directly into unguided vehicular travel (pre-brake delay travel) before braking begins')

# 6. Remove margin pointer block at line 286
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ Rolling-shutter distortion on the navigation camera, and its compensation before images enter the state estimator, is developed in the perception stage\.\n:::\n\n?', '', text)

# 7. PCIe -> Peripheral Component Interconnect Express (PCIe)
text = text.replace('PCIe transport', 'Peripheral Component Interconnect Express (PCIe) transport')
text = text.replace('PCIe DMA transfer', 'Peripheral Component Interconnect Express (PCIe) DMA transfer')

# 8. @Tbl-04-event-ledger
text = text.replace('@Tbl-04-event-ledger defines these four milestones', 'A formal event ledger defines these four milestones')

# 9. DDR -> double data rate (DDR)
text = text.replace('DDR memory controller', 'double data rate (DDR) memory controller')

# 10. @tbl-04-event-ledger
text = text.replace('Whereas @tbl-04-event-ledger formally budgets', 'Whereas a formal event ledger budgets')

# 11. blind pre-brake delay travel
text = text.replace('paid as blind travel $v\\,\\PrebrakeDelay$', 'paid as blind pre-brake delay travel $v\\,\\PrebrakeDelay$')

# 12. Ethernet PHY -> Ethernet physical layer transceiver
text = text.replace('Ethernet PHY timestamps', 'Ethernet physical layer transceiver (PHY) timestamps')

# 13. Remove margin pointer block at line 739
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ The localization bound \$\\LocalizationBound\$ and its dependence on map corrections are developed in the memory stage\.\n:::\n\n?', '', text)

# 14. tbl-04-event-ledger in fig-02-stopping-distance caption
text = text.replace('Case 2 adds the full `{python} BodyStoppingBudget.tau_full_str` pre-brake delay that @tbl-04-event-ledger assembles from later chapters; at `{python} BodyStoppingBudget.case_full_str` it still fits, although later chapters add a stop profile and a tracking bound.', 'Case 2 adds the full `{python} BodyStoppingBudget.tau_full_str` pre-brake delay that the event ledger assembles; at `{python} BodyStoppingBudget.case_full_str` it still fits, although a stop profile and a tracking bound will be added.')

# 15. Remove margin pointer block at line 909
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ The chunk lease, the time an admitted trajectory segment may keep executing without a replacement, is one of the delay terms that spend this headroom\. Its sizing from this stopping distance is developed in the nervous stage\.\n:::\n\n?', '', text)

# 16. enforcement stage -> safety enforcement
text = text.replace('That per-tick validation belongs to the permission path; the enforcement stage formalizes it around the jerk-limited deceleration profiles executed on the physical platform. What the physical body contributes is the ceiling itself', 'That per-tick validation belongs to the permission path; safety enforcement formalizes it around the jerk-limited deceleration profiles executed on the physical platform. What the physical body contributes is the ceiling itself')

# 17. i_q definition
text = text.replace('where $\\PhaseInductanceAndResistance$ is phase inductance, $\\PhaseInductanceAndResistanceTwo$ is phase resistance, $v_q(t)$ is the applied quadrature voltage from the inverter bridge, and $\\BackemfVoltageBemf(t) = \\MotorBackEmfConstant \\omega(t)$ is the back-EMF.', 'where $\\PhaseInductanceAndResistance$ is phase inductance, $\\PhaseInductanceAndResistanceTwo$ is phase resistance, $v_q(t)$ is the applied quadrature voltage from the inverter bridge, $i_q(t)$ is the quadrature-axis current (the torque-producing component of the phase current), and $\\BackemfVoltageBemf(t) = \\MotorBackEmfConstant \\omega(t)$ is the back-EMF.')

# 18. MOSFET
text = text.replace('switching MOSFETs', 'switching metal-oxide-semiconductor field-effect transistors (MOSFETs)')

# 19. Remove margin pointer block at line 1152
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ Trajectory timing under joint torque and jerk bounds is developed in the planning stage\.\n:::\n\n?', '', text)

# 20. nervous stage
text = text.replace('The impedance command interface, through which a learned policy proposes these gains and the permission path validates them before admission, is defined in the nervous stage.', 'The impedance command interface defines how a learned policy proposes these gains and the permission path validates them before admission.')

# 21. Remove margin pointer block at line 1853
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ How actuator dynamics mismatch is measured and closed in the training simulator is developed in the training stage\.\n:::\n\n?', '', text)

# 22. tbl-12-iec60204-stopping-categories
text = text.replace('and @tbl-12-iec60204-stopping-categories maps each one to the physical evidence it still requires. Beneath every row of that table lies the body\'s fact:', 'mapping each one to the physical evidence it still requires. Beneath every fallback sequence lies the body\'s fact:')

# 23. Remove margin pointer block at line 2115
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ The placement of Safe Torque Off in the machine\'s hardwired interlock is developed in the nervous stage\.\n:::\n\n?', '', text)

# 24. eq-eval-rule-of-three
text = text.replace('by the Rule of Three[^fn-rule-three-origin] (@eq-eval-rule-of-three), about one stop', 'by the Rule of Three[^fn-rule-three-origin], about one stop')

# 25. Remove margin pointer block at line 2543
text = re.sub(r'::: \{\.column-margin \.margin-pointer\}\n↪ The handoff record that carries each limit record to the permission path is developed in the nervous stage\.\n:::\n\n?', '', text)

# 26. enforcement stage
text = text.replace('While hardware qualification must account for component tolerances, conversion losses, and actual operating current, this hold-up window comfortably covers the multi-stage fallback ladder developed in the enforcement stage.', 'While hardware qualification must account for component tolerances, conversion losses, and actual operating current, this hold-up window comfortably covers the necessary multi-stage fallback ladder.')

# 27. Sec-brain & nervous stage
text = text.replace('@Sec-brain analyzes model generation capacity, trajectory chunk execution times, and observation freshness across learned proposers. These proposer dynamics meet the body\'s physical limits in the nervous stage, where the runtime permission path converts handoff records and timing guarantees into bounded action leases.', 'Learned proposers exhibit generation capacity, trajectory chunk execution times, and observation freshness constraints. These proposer dynamics meet the body\'s physical limits when the runtime permission path converts handoff records and timing guarantees into bounded action leases.')

with open('/private/tmp/MLSysBook-dev-notation/books/vol4/02_body/02_body.qmd', 'w') as f:
    f.write(text)

