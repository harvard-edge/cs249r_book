#!/usr/bin/env python3
"""Figures for Lecture 10b (Model Compression, Part 2: How Few Bits?).

Every value is sourced. "Chapter" means Model Compression in Volume I.
Outside papers were read on 2026-10-17.

  - Krishnamoorthi 2018 (arXiv 1806.08342). Table 3, post-training INT8
    weights and activations: MobileNetV2 1.0 224 asymmetric per-layer 0.001,
    symmetric per-channel 0.698, float 0.719; ResNet-v1-50 0.75, 0.751, 0.752;
    MobileNet-v1 symmetric per-channel 0.591, asymmetric per-channel 0.703,
    float 0.709. Table 4 (QAT, symmetric per-channel): MobileNetV2 0.711,
    ResNet-v1-50 0.75. Table 5 (4-bit weights): ResNet-v1-50 per-layer PTQ
    0.002, per-channel PTQ 0.54, QAT 0.732; MobileNetV2 0.001, 0.001, 0.62.
    Table 7 (Pixel 2): MobileNetV2 105 ms float CPU, 63 ms fixed point CPU,
    15.5 ms fixed point on the Hexagon DSP (NN-API).
  - Nagel et al. 2019 (arXiv 1906.04721), Figure 2: per-output-channel weight
    ranges of the first depthwise-separable layer of MobileNetV2, 32 channels,
    "strong differences between channel weight ranges". The redraw below is a
    schematic of that finding; no channel value is read from the plot. The
    paper's own example (section 3.1): one channel in [-128, 128], another in
    (-0.5, 0.5); at 8 bits the second rounds entirely to 0.
  - Dettmers et al. 2022 (arXiv 2208.07339), Table 1: C4 perplexity of fairseq
    dense transformers, 125M to 13B. 32-bit float 25.65 15.91 14.43 13.30
    12.45; INT8 absmax 87.76 16.55 15.11 14.59 19.08; absmax LLM.int8() 25.83
    15.93 14.44 13.24 12.45.
  - Frantar et al. 2023, GPTQ (arXiv 2210.17323). Table 5, OPT-175B WikiText2:
    FP16 8.34, RTN 4-bit 10.54, GPTQ 4-bit 8.37, RTN 3-bit 7.3e3, GPTQ 3-bit
    8.68. Table 6: per-token latency, FP16 on five A100s 230 ms, 3-bit on one
    A100 71 ms (3.24x).
  - Lin et al. 2024, AWQ (arXiv 2306.00978), Table 1: OPT-6.7B INT3 group 128,
    WikiText perplexity: FP16 10.86, RTN 23.54, 1 percent kept in FP16 chosen
    by activation 11.39, by weight 22.37.
  - Micikevicius et al. 2022 (arXiv 2209.05433), Table 1: E4M3 max normal 448,
    min subnormal 2^-9; E5M2 max 57,344. The E4M3 value grid below is derived
    from that definition (one NaN pattern, 127 non-negative finite values).
  - Chapter: FP16 max 65,504 (Sec. 10.3.1); storage 2x / 4x / 8x (table 7);
    Horowitz energy (table 8); Napkin Math 10.1 and 10.4; listing 2 weights;
    blur waterfall (fig-compression-waterfall, illustrative); GPT-2 XL
    streaming times (table 1).
  - Every value now comes from MLSysIM through calcs.py (imported; the
    printed strings through deckvals.val and num), so figures, slides, and
    the chapter share one computation. The citations above record where the
    MLSysIM entries come from.
  - Published figures shown as images (not generated here): Wu et al. 2020
    Fig. 2, Nagel et al. 2019 Fig. 2, Dettmers et al. 2022 Fig. 3(a); see
    SOURCES.yml.

Run from this folder: python3 make_figs.py
"""
import mlsysim
_VALS = {'r50_float': '75.2', 'r50_tensor': '75.0', 'r50_tensor_loss': '0.2', 'mbv2_float': '71.9', 'mbv2_tensor': '0.1', 'chance_pct': '0.1', 'imagenet_classes': '1,000', 'nagel_float': '71.72', 'nagel_tensor': '0.12', 'r50_chapter_top1': '76.1', 'gpt2_fp16_gb': '3', 'gpt2_read_ms': '59', 'phone_bw': '51.2', 'blur_cpu_ms': '125', 'blur_pruned_ms': '87', 'blur_int8_ms': '20', 'frame_ms': '33', 'mnv2_read_ms': '0.27', 'mnv2_read_share_pct': '1', 'r50_read_ms': '2', 'blur_fp32_mb': '14.0', 'blur_pruned_mb': '10.4', 'blur_int8_mb': '2.6', 'fp32_exp_bits': '8', 'fp32_frac_bits': '23', 'bf16_exp_bits': '8', 'fp16_max': '65,504', 'e4m3_exp_bits': '4', 'e4m3_frac_bits': '3', 'e4m3_max': '448', 'e5m2_max': '57,344', 'int8_levels': '256', 'int4_levels': '16', 'int8_smaller': '4', 'int4_smaller': '8', 'gguf_q2_k_bits': '2', 'gguf_q2_k_gb': '2.83', 'gguf_q2_k_bpw': '3.35', 'gguf_q3_k_m_bits': '3', 'gguf_q3_k_m_gb': '3.30', 'gguf_q3_k_m_bpw': '3.91', 'gguf_q4_0_bits': '4', 'gguf_q4_0_gb': '3.83', 'gguf_q4_0_bpw': '4.54', 'gguf_q4_k_m_bits': '4', 'gguf_q4_k_m_gb': '4.08', 'gguf_q4_k_m_bpw': '4.84', 'gguf_q6_k_bits': '6', 'gguf_q6_k_gb': '5.53', 'gguf_q6_k_bpw': '6.56', 'gguf_q8_0_bits': '8', 'gguf_q8_0_gb': '7.16', 'gguf_q8_0_bpw': '8.50', 'llama_params_b': '6.74', 'q4_block': '32', 'q4_scale_bytes': '2', 'q4_scale_bits': '16', 'q4_code_bytes': '16', 'q4_block_bytes': '18', 'q4_bpw': '4.5', 'q4_overhead_pct': '12.5', 'q2k_block_bpw': '2.5625', 'e4m3_pos': '126', 'e4m3_below1': '55', 'e4m3_top': '7', 'e4m3_top_binade': '256', 'e4m3_min': '0.002', 'e4m3_min_exp': '-9', 'e4m3_decades_word': 'five', 'int8_pos': '127', 'int8_step448': '3.53', 'int8_zero_below': '1.76', 'e_add_fp32': '0.90', 'e_add_int8': '0.03', 'e_add_ratio': '30', 'e_mul_ratio': '18.5', 'e_mac_ratio': '15.3', 'e_dram': '640', 'e_dram_mac': '2,100', 'spotter_fp32_kb': '800', 'spotter_int8_kb': '200', 'mcu_sram_kb': '512', 'toy_alpha': '-1', 'toy_neg_alpha': '1', 'toy_beta': '2.5', 'toy_bits': '3', 'toy_bits_word': 'three', 'toy_outside_word': 'two', 'toy_levels': '8', 'toy_steps': '7', 'toy_range': '3.5', 'toy_s': '0.5', 'toy_z': '2', 'toy_half': '0.25', 'toy_x0': '-1.4', 'toy_q0': '0', 'toy_r0': '-1', 'toy_x1': '-0.3', 'toy_q1': '1', 'toy_r1': '-0.5', 'toy_x2': '0.8', 'toy_q2': '4', 'toy_r2': '1.0', 'toy_x3': '1.6', 'toy_q3': '5', 'toy_r3': '1.5', 'toy_x4': '2.9', 'toy_q4': '7', 'toy_r4': '2.5', 'toy_x5': '2.2', 'toy_q5': '6', 'toy_r5': '2.0', 'toy_err5': '0.2', 'nm_alpha': '-1', 'nm_neg_alpha': '1', 'nm_beta': '3', 'nm_bits': '8', 'nm_range': '4', 'nm_steps': '255', 'nm_s': '0.0157', 'nm_z_exact': '63.75', 'nm_z': '64', 'nm_nudge': '0.25', 'nm_x0': '-1', 'nm_q0': '0', 'nm_x1': '1', 'nm_q1': '128', 'nm_x2': '0', 'nm_q2': '64', 'nm_x3': '3', 'nm_q3': '255', 'nm_r0': '-1.004', 'nm_r0_torch': '-1.0039', 'l2_max': '0.392', 'l2_int_max': '127', 'l2_s': '0.00309', 'l2_half': '0.00154', 'l2_outlier': '0.784', 'l2_w0': '0.127', 'l2_q0': '41', 'l2_e0': '0.00045', 'l2_w1': '-0.084', 'l2_q1': '-27', 'l2_e1': '0.00066', 'l2_w2': '0.392', 'l2_q2': '127', 'l2_e2': '0', 'l2_w3': '-0.203', 'l2_q3': '-66', 'l2_e3': '0.00072', 'zp_sym_lo': '-1', 'zp_sym_hi': '1', 'zp_sym_qmin': '-127', 'zp_sym_qmax': '127', 'zp_asym_lo': '-0.5', 'zp_asym_hi': '1.5', 'zp_asym_qmin': '-128', 'zp_asym_qmax': '127', 'zp_asym_range': '2', 'zp_asym_steps': '255', 'zp_asym_z': '-64', 'int8_shift': '128', 'relu_hi': '6', 'relu_sym_div': '127', 'relu_asym_div': '255', 'relu_sym_s': '0.047', 'relu_asym_s': '0.024', 'mbv1_float': '70.9', 'mbv1_sym': '59.1', 'mbv1_asym': '70.3', 'mbv1_sym_cost': '11', 'il_w_max': '0.75', 'il_x_max': '2', 'il_int_max': '127', 'il_w0': '0.5', 'il_x0': '1.0', 'il_wq0': '85', 'il_xq0': '64', 'il_p0': '5,440', 'il_w1': '-0.25', 'il_x1': '2.0', 'il_wq1': '-42', 'il_xq1': '127', 'il_p1': '-5,334', 'il_w2': '0.75', 'il_x2': '0.5', 'il_wq2': '127', 'il_xq2': '32', 'il_p2': '4,064', 'il_w0_ratio': '84.7', 'il_x0_ratio': '63.5', 'il_sum': '4,170', 'il_readback': '0.388', 'il_exact': '0.375', 'il_m': '0.0118', 'il_y': '49', 'il_sy_div': '127', 'il_headroom': '133,000', 'il_acc_exp': '31', 'clip_full': '2.5', 'clip_cut': '1.1', 'clip_steps': '255', 'clip_full_s': '0.0098', 'clip_cut_s': '0.0043', 'clip_finer': '2.3', 'wu_resnet50_fp32': '76.16', 'wu_resnet50_max': '75.82', 'wu_resnet50_entropy': '76.05', 'wu_resnet50_pct_99_99': '75.98', 'wu_mobilenet_fp32': '71.88', 'wu_mobilenet_max': '69.41', 'wu_mobilenet_entropy': '70.28', 'wu_mobilenet_pct_99_99': '71.14', 'wu_inception_fp32': '79.71', 'wu_inception_max': '0.12', 'wu_inception_entropy': '79.60', 'wu_inception_pct_99_99': '79.63', 'wu_efficientnet_fp32': '76.85', 'wu_efficientnet_max': '22.3', 'wu_efficientnet_entropy': '72.06', 'wu_efficientnet_pct_99_99': '68.33', 'wu_bert_fp32': '91.01', 'wu_bert_max': '85.92', 'wu_bert_entropy': '37.40', 'wu_bert_pct_99_99': '89.59', 'wu_bert_best': '90.20', 'mbv2_channel': '69.8', 'mbv2_qat': '71.1', 'mbv2_qat_gain': '1.3', 'r50_channel': '75.1', 'r50_qat': '75.0', 'nagel_wide': '128', 'nagel_narrow': '0.5', 'nagel_step': '1', 'nagel_channels': '32', 'awq_group': '128', 'det_small_m': '125', 'det_mid_b': '1.3', 'det_large_b': '13', 'det_fourth_b': '6.7', 'det_absmax_small': '87.76', 'det_absmax_large': '19.08', 'det_fp32_large': '12.45', 'out_b': '6.7', 'out_floor_b': '6', 'out_tokens_pct': '75', 'out_mult': '20', 'out_dims': '6', 'out_threshold': '6', 'cliff_8bit_gap': '2', 'int4_bits': '4', 'int8_bits': '8', 'r50_w4_tensor': '0.2', 'mbv2_w4_tensor': '0.1', 'r50_w4_channel': '54.0', 'mbv2_w4_channel': '0.1', 'r50_w4_qat': '73.2', 'mbv2_w4_qat': '62.0', 'r50_w4_qat_gap': '2', 'mbv2_w4_qat_gap': '10', 'gptq_fp16': '8.34', 'gptq_rtn4': '10.54', 'gptq_4': '8.37', 'gptq_rtn3_k': '7.3', 'gptq_3': '8.68', 'awq_fp16': '10.86', 'awq_rtn': '23.54', 'awq_by_weight': '22.37', 'awq_by_act': '11.39', 'awq_kept_pct': '1', 'dec_w16_gb': '3', 'dec_w4_gb': '0.75', 'dec_kv_gb': '0.31', 'dec_t16_gb': '3.31', 'dec_t4_gb': '1.06', 'dec_t16_ms': '65', 'dec_t4_ms': '21', 'dec_speedup': '3.1', 'dec_byte_ratio': '4', 'h100_int8': '1,979', 'h100_fp16': '989', 'h100_int8_x': '2', 'h100_int8_x_word': 'two', 'gptq_fp16_ms': '230', 'gptq_3bit_ms': '71', 'gptq_fp16_gpus_word': 'five', 'gptq_3bit_gpus_word': 'one', 'gptq_x': '3.24', 'lb_ultra_bw': '800', 'lb_tg4_x': '2.30', 'lb_tg8_x': '1.62', 'lb_pp4_x': '0.88', 'lb_file_x': '3.53', 'lb_tg16': '41', 'lb_pp16': '1,402', 'lb_tg8': '67', 'lb_pp8': '1,249', 'lb_tg4': '94', 'lb_pp4': '1,238', 'lb_other_tg_lo': '2.63', 'lb_other_tg_hi': '2.86', 'lb_other_pp_lo': '0.88', 'lb_other_pp_hi': '0.98', 'lb_stream16': '553', 'lb_stream4': '360', 'dq_sep_gb': '7.06', 'dq_sep_ms': '138', 'dq_sep_x': '2.13', 'px_fp32': '105', 'px_cpu': '63', 'px_dsp': '15.5', 'px_cpu_x': '1.7', 'px_dsp_x': '6.8', 'simd_bits': '512', 'simd_fp32': '16', 'simd_int8': '64', 'pf_fp32_top1': '76.1', 'pf_int8_top1': '75.8', 'pf_top1_limit': '75.5', 'pf_top1_drop': '0.3', 'pf_fp32_cal': '2.1', 'pf_int8_cal': '3.4', 'pf_cal_limit': '3.0', 'pf_latency_x': '1.3', 'pf_throughput_x': '5.9', 'pf_energy_x': '5.9', 'pf_tdp_w': '700', 'p7b_b': '7', 'p7b_fp16_gb': '14', 'p7b_int4_gb': '3.5', 'p7b_q4_gb': '3.9', 'int4_bytes': '0.5', 'phone_ram_gb': '8'}
_NUMS = {'r50_float': 75.2, 'r50_tensor': 75.0, 'r50_tensor_loss': 0.2, 'mbv2_float': 71.9, 'mbv2_tensor': 0.1, 'chance_pct': 0.1, 'imagenet_classes': 1000.0, 'nagel_float': 71.72, 'nagel_tensor': 0.12, 'r50_chapter_top1': 76.1, 'gpt2_fp16_gb': 3.0, 'gpt2_read_ms': 59.0, 'phone_bw': 51.2, 'blur_cpu_ms': 125.0, 'blur_pruned_ms': 87.0, 'blur_int8_ms': 20.0, 'frame_ms': 33.0, 'mnv2_read_ms': 0.27, 'mnv2_read_share_pct': 1.0, 'r50_read_ms': 2.0, 'blur_fp32_mb': 14.0, 'blur_pruned_mb': 10.4, 'blur_int8_mb': 2.6, 'fp32_exp_bits': 8.0, 'fp32_frac_bits': 23.0, 'bf16_exp_bits': 8.0, 'fp16_max': 65504.0, 'e4m3_exp_bits': 4.0, 'e4m3_frac_bits': 3.0, 'e4m3_max': 448.0, 'e5m2_max': 57344.0, 'int8_levels': 256.0, 'int4_levels': 16.0, 'int8_smaller': 4.0, 'int4_smaller': 8.0, 'gguf_q2_k_bits': 2.0, 'gguf_q2_k_gb': 2.83, 'gguf_q2_k_bpw': 3.35, 'gguf_q3_k_m_bits': 3.0, 'gguf_q3_k_m_gb': 3.3, 'gguf_q3_k_m_bpw': 3.91, 'gguf_q4_0_bits': 4.0, 'gguf_q4_0_gb': 3.83, 'gguf_q4_0_bpw': 4.54, 'gguf_q4_k_m_bits': 4.0, 'gguf_q4_k_m_gb': 4.08, 'gguf_q4_k_m_bpw': 4.84, 'gguf_q6_k_bits': 6.0, 'gguf_q6_k_gb': 5.53, 'gguf_q6_k_bpw': 6.56, 'gguf_q8_0_bits': 8.0, 'gguf_q8_0_gb': 7.16, 'gguf_q8_0_bpw': 8.5, 'llama_params_b': 6.74, 'q4_block': 32.0, 'q4_scale_bytes': 2.0, 'q4_scale_bits': 16.0, 'q4_code_bytes': 16.0, 'q4_block_bytes': 18.0, 'q4_bpw': 4.5, 'q4_overhead_pct': 12.5, 'q2k_block_bpw': 2.5625, 'e4m3_pos': 126.0, 'e4m3_below1': 55.0, 'e4m3_top': 7.0, 'e4m3_top_binade': 256.0, 'e4m3_min': 0.002, 'e4m3_min_exp': -9.0, 'int8_pos': 127.0, 'int8_step448': 3.53, 'int8_zero_below': 1.76, 'e_add_fp32': 0.9, 'e_add_int8': 0.03, 'e_add_ratio': 30.0, 'e_mul_ratio': 18.5, 'e_mac_ratio': 15.3, 'e_dram': 640.0, 'e_dram_mac': 2100.0, 'spotter_fp32_kb': 800.0, 'spotter_int8_kb': 200.0, 'mcu_sram_kb': 512.0, 'toy_alpha': -1.0, 'toy_neg_alpha': 1.0, 'toy_beta': 2.5, 'toy_bits': 3.0, 'toy_levels': 8.0, 'toy_steps': 7.0, 'toy_range': 3.5, 'toy_s': 0.5, 'toy_z': 2.0, 'toy_half': 0.25, 'toy_x0': -1.4, 'toy_q0': 0.0, 'toy_r0': -1.0, 'toy_x1': -0.3, 'toy_q1': 1.0, 'toy_r1': -0.5, 'toy_x2': 0.8, 'toy_q2': 4.0, 'toy_r2': 1.0, 'toy_x3': 1.6, 'toy_q3': 5.0, 'toy_r3': 1.5, 'toy_x4': 2.9, 'toy_q4': 7.0, 'toy_r4': 2.5, 'toy_x5': 2.2, 'toy_q5': 6.0, 'toy_r5': 2.0, 'toy_err5': 0.2, 'nm_alpha': -1.0, 'nm_neg_alpha': 1.0, 'nm_beta': 3.0, 'nm_bits': 8.0, 'nm_range': 4.0, 'nm_steps': 255.0, 'nm_s': 0.0157, 'nm_z_exact': 63.75, 'nm_z': 64.0, 'nm_nudge': 0.25, 'nm_x0': -1.0, 'nm_q0': 0.0, 'nm_x1': 1.0, 'nm_q1': 128.0, 'nm_x2': 0.0, 'nm_q2': 64.0, 'nm_x3': 3.0, 'nm_q3': 255.0, 'nm_r0': -1.004, 'nm_r0_torch': -1.0039, 'l2_max': 0.392, 'l2_int_max': 127.0, 'l2_s': 0.00309, 'l2_half': 0.00154, 'l2_outlier': 0.784, 'l2_w0': 0.127, 'l2_q0': 41.0, 'l2_e0': 0.00045, 'l2_w1': -0.084, 'l2_q1': -27.0, 'l2_e1': 0.00066, 'l2_w2': 0.392, 'l2_q2': 127.0, 'l2_e2': 0.0, 'l2_w3': -0.203, 'l2_q3': -66.0, 'l2_e3': 0.00072, 'zp_sym_lo': -1.0, 'zp_sym_hi': 1.0, 'zp_sym_qmin': -127.0, 'zp_sym_qmax': 127.0, 'zp_asym_lo': -0.5, 'zp_asym_hi': 1.5, 'zp_asym_qmin': -128.0, 'zp_asym_qmax': 127.0, 'zp_asym_range': 2.0, 'zp_asym_steps': 255.0, 'zp_asym_z': -64.0, 'int8_shift': 128.0, 'relu_hi': 6.0, 'relu_sym_div': 127.0, 'relu_asym_div': 255.0, 'relu_sym_s': 0.047, 'relu_asym_s': 0.024, 'mbv1_float': 70.9, 'mbv1_sym': 59.1, 'mbv1_asym': 70.3, 'mbv1_sym_cost': 11.0, 'il_w_max': 0.75, 'il_x_max': 2.0, 'il_int_max': 127.0, 'il_w0': 0.5, 'il_x0': 1.0, 'il_wq0': 85.0, 'il_xq0': 64.0, 'il_p0': 5440.0, 'il_w1': -0.25, 'il_x1': 2.0, 'il_wq1': -42.0, 'il_xq1': 127.0, 'il_p1': -5334.0, 'il_w2': 0.75, 'il_x2': 0.5, 'il_wq2': 127.0, 'il_xq2': 32.0, 'il_p2': 4064.0, 'il_w0_ratio': 84.7, 'il_x0_ratio': 63.5, 'il_sum': 4170.0, 'il_readback': 0.388, 'il_exact': 0.375, 'il_m': 0.0118, 'il_y': 49.0, 'il_sy_div': 127.0, 'il_headroom': 133000.0, 'il_acc_exp': 31.0, 'clip_full': 2.5, 'clip_cut': 1.1, 'clip_steps': 255.0, 'clip_full_s': 0.0098, 'clip_cut_s': 0.0043, 'clip_finer': 2.3, 'wu_resnet50_fp32': 76.16, 'wu_resnet50_max': 75.82, 'wu_resnet50_entropy': 76.05, 'wu_resnet50_pct_99_99': 75.98, 'wu_mobilenet_fp32': 71.88, 'wu_mobilenet_max': 69.41, 'wu_mobilenet_entropy': 70.28, 'wu_mobilenet_pct_99_99': 71.14, 'wu_inception_fp32': 79.71, 'wu_inception_max': 0.12, 'wu_inception_entropy': 79.6, 'wu_inception_pct_99_99': 79.63, 'wu_efficientnet_fp32': 76.85, 'wu_efficientnet_max': 22.3, 'wu_efficientnet_entropy': 72.06, 'wu_efficientnet_pct_99_99': 68.33, 'wu_bert_fp32': 91.01, 'wu_bert_max': 85.92, 'wu_bert_entropy': 37.4, 'wu_bert_pct_99_99': 89.59, 'wu_bert_best': 90.2, 'mbv2_channel': 69.8, 'mbv2_qat': 71.1, 'mbv2_qat_gain': 1.3, 'r50_channel': 75.1, 'r50_qat': 75.0, 'nagel_wide': 128.0, 'nagel_narrow': 0.5, 'nagel_step': 1.0, 'nagel_channels': 32.0, 'awq_group': 128.0, 'det_small_m': 125.0, 'det_mid_b': 1.3, 'det_large_b': 13.0, 'det_fourth_b': 6.7, 'det_absmax_small': 87.76, 'det_absmax_large': 19.08, 'det_fp32_large': 12.45, 'out_b': 6.7, 'out_floor_b': 6.0, 'out_tokens_pct': 75.0, 'out_mult': 20.0, 'out_dims': 6.0, 'out_threshold': 6.0, 'cliff_8bit_gap': 2.0, 'int4_bits': 4.0, 'int8_bits': 8.0, 'r50_w4_tensor': 0.2, 'mbv2_w4_tensor': 0.1, 'r50_w4_channel': 54.0, 'mbv2_w4_channel': 0.1, 'r50_w4_qat': 73.2, 'mbv2_w4_qat': 62.0, 'r50_w4_qat_gap': 2.0, 'mbv2_w4_qat_gap': 10.0, 'gptq_fp16': 8.34, 'gptq_rtn4': 10.54, 'gptq_4': 8.37, 'gptq_rtn3_k': 7.3, 'gptq_3': 8.68, 'awq_fp16': 10.86, 'awq_rtn': 23.54, 'awq_by_weight': 22.37, 'awq_by_act': 11.39, 'awq_kept_pct': 1.0, 'dec_w16_gb': 3.0, 'dec_w4_gb': 0.75, 'dec_kv_gb': 0.31, 'dec_t16_gb': 3.31, 'dec_t4_gb': 1.06, 'dec_t16_ms': 65.0, 'dec_t4_ms': 21.0, 'dec_speedup': 3.1, 'dec_byte_ratio': 4.0, 'h100_int8': 1979.0, 'h100_fp16': 989.0, 'h100_int8_x': 2.0, 'gptq_fp16_ms': 230.0, 'gptq_3bit_ms': 71.0, 'gptq_x': 3.24, 'lb_ultra_bw': 800.0, 'lb_tg4_x': 2.3, 'lb_tg8_x': 1.62, 'lb_pp4_x': 0.88, 'lb_file_x': 3.53, 'lb_tg16': 41.0, 'lb_pp16': 1402.0, 'lb_tg8': 67.0, 'lb_pp8': 1249.0, 'lb_tg4': 94.0, 'lb_pp4': 1238.0, 'lb_other_tg_lo': 2.63, 'lb_other_tg_hi': 2.86, 'lb_other_pp_lo': 0.88, 'lb_other_pp_hi': 0.98, 'lb_stream16': 553.0, 'lb_stream4': 360.0, 'dq_sep_gb': 7.06, 'dq_sep_ms': 138.0, 'dq_sep_x': 2.13, 'px_fp32': 105.0, 'px_cpu': 63.0, 'px_dsp': 15.5, 'px_cpu_x': 1.7, 'px_dsp_x': 6.8, 'simd_bits': 512.0, 'simd_fp32': 16.0, 'simd_int8': 64.0, 'pf_fp32_top1': 76.1, 'pf_int8_top1': 75.8, 'pf_top1_limit': 75.5, 'pf_top1_drop': 0.3, 'pf_fp32_cal': 2.1, 'pf_int8_cal': 3.4, 'pf_cal_limit': 3.0, 'pf_latency_x': 1.3, 'pf_throughput_x': 5.9, 'pf_energy_x': 5.9, 'pf_tdp_w': 700.0, 'p7b_b': 7.0, 'p7b_fp16_gb': 14.0, 'p7b_int4_gb': 3.5, 'p7b_q4_gb': 3.9, 'int4_bytes': 0.5, 'phone_ram_gb': 8.0}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.units import FLOAT_FORMAT_BITS, FP8_FORMAT_BITS, FP32_BITS, INT4_BITS, INT8_BITS
from mlsysim.physics import calc_dequantize, calc_quantize
plt = sp.plt
C = sp.C
out = Path(__file__).parent
FONT = 'Helvetica, Arial, sans-serif'

def bars(path, rows, shown, log=False, xlim=None, xlabel=None, size=sp.SIZE, vlines=(), xticks=None, note=None, label_size=13, boxed=False, text_size=12, axis_size=None):
    """axis_size: tick, axis-label, and reference-line label size; raise it
    with label_size and text_size when the plot is drawn for a column slot."""
    'Horizontal bars with a show mask; hidden rows keep their slot.'
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        vis = shown[i]
        ax.barh(y, r['value'], color=C[r['color']], height=0.62, alpha=1.0 if vis else 0.0)
        ax.text(-0.01, y, r['label'], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=label_size, color=C['text'], fontweight='bold', alpha=1.0 if vis else 0.0)
        tx = r['value'] * 1.25 if log else r['value'] + (xlim[1] if xlim else 1) * 0.012
        bb = dict(facecolor='white', edgecolor='none', pad=1.0, alpha=1.0 if vis else 0.0) if boxed else None
        ax.text(r.get('tx', tx), y, r['text'], va='center', fontsize=text_size, bbox=bb, zorder=4, color=C[r.get('tcolor', r['color'])], fontweight='bold', alpha=1.0 if vis else 0.0)
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls='--', zorder=1)
        ax.text(x, n - 0.45, ' ' + label, color=C[ck], fontsize=axis_size or 10, fontweight='bold', va='bottom', ha='left')
    if log:
        ax.set_xscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axis_size)
    if axis_size:
        ax.tick_params(labelsize=axis_size)
    if note:
        fig.text(0.99, 0.01, note, ha='right', va='bottom', fontsize=9, color=C['neutral'])
    return (fig, ax)

def bars_save(path, *a, **k):
    fig, _ = bars(path, *a, **k)
    sp.save(fig, path)

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')

def text(x, y, s, size=22, color=C['text'], weight='normal', anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n'

def box(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n'

def vbars(ax, groups, series, shown, width=0.2, fmt='{:.1f}', ymax=85, inside=None):
    """Grouped vertical bars. groups: names; series: list of (name, color,
    values per group). shown[s][g] masks a bar (hidden bars keep their slot).
    """
    ns = len(series)
    for si, (sname, ck, vals) in enumerate(series):
        for gi, v in enumerate(vals):
            x = gi + (si - (ns - 1) / 2) * width
            vis = shown[si][gi]
            if v is None:
                continue
            ax.bar(x, v, width=width * 0.92, color=C[ck], alpha=1.0 if vis else 0.0)
            ax.text(x, v + ymax * 0.015, fmt.format(v), ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=C[ck], alpha=1.0 if vis else 0.0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=12, fontweight='bold')
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    groups = ['ResNet-50', 'MobileNetV2']
    flt = [75.2, 71.9]
    i8 = [75.0, 0.1]
    w = 0.36
    for gi in range(2):
        show_f = gi == 0 or k >= 2
        show_q = gi == 0 or k >= 3
        a = 1.0 if show_f else 0.0
        ax.bar(gi - w / 2, flt[gi], width=w * 0.92, color=C['neutral'], alpha=a)
        ax.text(gi - w / 2, flt[gi] + 1.5, f'{flt[gi]:.1f}', ha='center', fontsize=12, fontweight='bold', color=C['neutral'], alpha=a)
        aq = 1.0 if show_q else 0.0
        ax.bar(gi + w / 2, max(i8[gi], 0.6), width=w * 0.92, color=C['compute'], alpha=aq)
        ax.text(gi + w / 2, i8[gi] + 1.5, f'{i8[gi]:.1f}', ha='center', fontsize=12, fontweight='bold', color=C['danger'] if i8[gi] < 1 else C['compute'], alpha=aq)
        if gi == 1 and k == 2:
            ax.add_patch(plt.Rectangle((gi + w / 2 - w * 0.46, 0), w * 0.92, 40, fill=False, ls='--', lw=1.5, ec=C['compute']))
            ax.text(gi + w / 2, 18, '?', ha='center', va='center', fontsize=26, fontweight='bold', color=C['compute'])
    ax.text(0.02, 0.98, 'float', transform=ax.transAxes, color=C['neutral'], fontsize=12, fontweight='bold', va='top')
    ax.text(0.22, 0.98, 'INT8, one scale per tensor', transform=ax.transAxes, color=C['compute'], fontsize=12, fontweight='bold', va='top')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(groups, fontsize=13, fontweight='bold')
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('ImageNet top-1 (%)')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'hook_step{k}.svg')

def smaller(e, f):
    return f'{FP32_BITS // (1 + e + f)}× smaller'
F = {**FLOAT_FORMAT_BITS, **FP8_FORMAT_BITS}
formats = [('FP32', 1, *F['fp32'], 0, 'baseline', 1), ('BF16', 1, *F['bf16'], 0, f"FP32's range, {smaller(*F['bf16'])}", 1), ('FP16', 1, *F['fp16'], 0, f"max 65,504, {smaller(*F['fp16'])}", 1), ('FP8 E4M3', 1, *F['e4m3'], 0, f"max 448, {smaller(*F['e4m3'])}", 2), ('FP8 E5M2', 1, *F['e5m2'], 0, f"max 57,344, {smaller(*F['e5m2'])}", 2), ('INT8', 0, 0, 0, INT8_BITS, '256 levels, 4× smaller', 3), ('INT4', 0, 0, 0, INT4_BITS, '16 levels, 8× smaller', 3)]
BW = 15
X0 = 150
for k in (1, 2, 3):
    body = ''
    lx = X0
    for name, fill, stroke in (('sign', C['neutral_l'], C['neutral']), ('exponent: range', C['memory_l'], C['memory']), ('mantissa: precision', C['data_l'], C['data']), ('integer levels', C['compute_l'], C['compute'])):
        body += box(lx, 8, 22, 22, fill, stroke, 2, 2)
        body += text(lx + 30, 26, name, 20, C['text'])
        lx += 46 + len(name) * 10.5
    for i, (name, s, e, m, b, note, step) in enumerate(formats):
        y = 38 + i * 44
        op = 1.0 if step <= k else 0.0
        if step == 1 and k > 1:
            op = 0.55
        body += f'<g opacity="{op}">\n'
        body += text(X0 - 12, y + 27, name, 22, C['text'], 'bold', 'end')
        x = X0
        for nbits, fill, stroke in ((s, C['neutral_l'], C['neutral']), (e, C['memory_l'], C['memory']), (m, C['data_l'], C['data']), (b, C['compute_l'], C['compute'])):
            if nbits:
                body += box(x, y + 6, nbits * BW, 30, fill, stroke, 2, 2)
                if nbits * BW >= 30:
                    body += text(x + nbits * BW / 2, y + 28, str(nbits), 20, stroke, 'bold', 'middle')
                x += nbits * BW
        body += text(X0 + 32 * BW + 18, y + 28, note, 21, C['text'])
        body += '</g>\n'
    body += text(X0 + 32 * BW + 18, 30, '', 10)
    svg(out / f'formats_step{k}.svg', 900, 352, body)
pos = np.array([0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0])
s_int = 3.527559055
int_levels = s_int * np.arange(1, 127 + 1)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.vlines(pos, 1.0, 1.6, color=C['memory'], lw=1.1)
    ax.text(0.0013, 1.95, 'FP8 E4M3: 126 positive values, 55 of them below 1', fontsize=12, fontweight='bold', color=C['memory'])
    a = 1.0 if k == 2 else 0.0
    ax.vlines(int_levels, 0.0, 0.6, color=C['compute'], lw=1.1, alpha=a)
    ax.axvspan(0.001, s_int / 2, ymin=0.233, ymax=0.433, color=C['danger_l'], alpha=a)
    ax.text(0.0013, -0.5, 'INT8 scaled to 448: one level every 3.53; all below 1.76 round to 0', fontsize=12, fontweight='bold', color=C['compute'], alpha=a)
    ax.set_xscale('log')
    ax.set_xlim(0.001, 1000)
    ax.set_ylim(-0.7, 2.3)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100', '1000'])
    ax.minorticks_off()
    ax.set_xlabel('Magnitude of a stored value (log scale)')
    sp.save(fig, out / f'grids_step{k}.svg')
ops = [('Add', 0.9, 0.03, '30×'), ('Multiply', 3.7, 0.2, '18.5×'), ('Multiply-accumulate', 4.6, 0.3, '15.3×')]
DRAM_32 = 640.0
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ylabels = []
    for i, (name, f32, i8, ratio) in enumerate(ops):
        y = 3 - i
        ax.barh(y + 0.19, f32, height=0.34, color=C['compute'])
        ax.barh(y - 0.19, i8, height=0.34, color=C['neutral'])
        ax.text(f32 * 1.2, y + 0.19, f'FP32 {f32:.2f} pJ', va='center', fontsize=11, color=C['compute'], fontweight='bold')
        ax.text(i8 * 1.2, y - 0.19, f'INT8 {i8:.2f} pJ: {ratio} less', va='center', fontsize=11, color=C['neutral'], fontweight='bold')
        ylabels.append((y, name))
    vis = 1.0 if k == 2 else 0.0
    ax.barh(-0.05, DRAM_32, height=0.42, color=C['memory'], alpha=vis)
    ax.text(DRAM_32 * 1.2, -0.05, '640 pJ', va='center', fontsize=11, color=C['memory'], fontweight='bold', alpha=vis)
    ylabels.append((-0.05, '32-bit DRAM read'))
    for y, name in ylabels:
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'], alpha=vis if name.startswith('32') else 1.0)
    ax.set_xscale('log')
    ax.set_xlim(0.01, 20000)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Energy (picojoules, log scale; 45 nm)')
    sp.save(fig, out / f'energy_step{k}.svg')
ALPHA, BETA = (-1.0, 2.5)
S_TOY, Z_TOY = (0.5, 2)
levels = S_TOY * (np.arange(0, 7 + 1) - Z_TOY)
samples = [-1.4, -0.3, 0.8, 1.6, 2.9]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    ax.plot([-1.75, 3.25], [1, 1], color=C['text'], lw=1.5)
    ax.axvspan(ALPHA, BETA, ymin=0.52, ymax=0.7, color=C['memory_l'])
    ax.text(ALPHA, 1.58, 'α = -1'.replace('-', '−'), ha='center', fontsize=13, fontweight='bold', color=C['memory'])
    ax.text(BETA, 1.58, 'β = 2.5', ha='center', fontsize=13, fontweight='bold', color=C['memory'])
    ax.text(-1.9, 1, 'real', ha='right', va='center', fontsize=13, fontweight='bold')
    for x in samples:
        inside = ALPHA <= x <= BETA
        ax.plot([x], [1], 'o', ms=9, color=C['data'] if inside else C['danger'], zorder=5)
        ax.text(x, 1.18, f'{x:g}', ha='center', fontsize=12, fontweight='bold', color=C['data'] if inside else C['danger'])
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot([-1.75, 3.25], [0, 0], color=C['neutral'], lw=1.5, alpha=a2)
    ax.text(-1.9, 0, '3-bit\ncode', ha='right', va='center', fontsize=12, fontweight='bold', color=C['neutral'], alpha=a2)
    for q, lv in enumerate(levels):
        hl = q == Z_TOY
        ax.plot([lv, lv], [-0.12, 0.12], color=C['compute'], lw=3 if hl else 2, alpha=a2)
        ax.plot([lv, lv], [0.12, 0.95], color=C['compute'], lw=0.8, ls=':', alpha=a2)
        ax.text(lv, -0.42, str(q), ha='center', fontsize=13, fontweight='bold', color=C['danger'] if hl else C['compute'], alpha=a2)
    ax.annotate('', (levels[5], -0.66), (levels[4], -0.66), arrowprops=dict(arrowstyle='<->', color=C['text'], lw=1.3, alpha=a2))
    ax.text(levels[4] + S_TOY / 2, -1.02, 's = 0.5', ha='center', fontsize=12, fontweight='bold', alpha=a2)
    ax.text(levels[Z_TOY], -1.02, 'z = 2', ha='center', fontsize=12, fontweight='bold', color=C['danger'], alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    for x in samples:
        q, xr = {-1.4: (0, -1.0), -0.3: (1, -0.5), 0.8: (4, 1.0), 1.6: (5, 1.5), 2.9: (7, 2.5)}[x]
        inside = ALPHA <= x <= BETA
        ck = C['data'] if inside else C['danger']
        ax.annotate('', (xr, 0.14), (x, 0.9), arrowprops=dict(arrowstyle='-|>', color=ck, lw=1.6, alpha=a3))
    ax.text(3.25, 0.5, 'outside:\nclamped', ha='left', va='center', fontsize=11, fontweight='bold', color=C['danger'], alpha=a3)
    ax.set_xlim(-2.6, 3.9)
    ax.set_ylim(-1.2, 1.85)
    ax.axis('off')
    sp.save(fig, out / f'affgrid_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 1.7))
    ax.plot([-1.25, 3.25], [1, 1], color=C['text'], lw=1.5)
    ax.plot([-1.25, 3.25], [0, 0], color=C['neutral'], lw=1.5)
    ax.text(-1.4, 1, 'real', ha='right', va='center', fontsize=15, fontweight='bold')
    ax.text(-1.4, 0, 'UINT8', ha='right', va='center', fontsize=15, fontweight='bold', color=C['neutral'])
    for real, q in zip([-1.0, 1.0, 0.0, 3.0], [0, 128, 64, 255]):
        xq = -1.0 + q * 0.01568627451
        ax.plot([real], [1], 'o', ms=9, color=C['memory'])
        ax.text(real, 1.25, f'{real:g}', ha='center', fontsize=15, fontweight='bold', color=C['memory'])
        a = 1.0 if k == 2 else 0.0
        ax.annotate('', (xq, 0.06), (real, 0.92), alpha=a, arrowprops=dict(arrowstyle='-|>', color=C['compute'], lw=1.6, alpha=a))
        ax.plot([xq], [0], 's', ms=8, color=C['compute'], alpha=a)
        ax.text(xq, -0.5, f'{q}', ha='center', fontsize=15, fontweight='bold', color=C['compute'], alpha=a)
    ax.set_xlim(-2.2, 3.4)
    ax.set_ylim(-0.6, 1.5)
    ax.axis('off')
    sp.save(fig, out / f'numline_step{k}.svg')
xs = np.linspace(-1.75, 3.25, 4001)
err = np.array([0.75, 0.74875, 0.7475, 0.74625, 0.745, 0.74375, 0.7425, 0.74125, 0.74, 0.73875, 0.7375, 0.73625, 0.735, 0.73375, 0.7325, 0.73125, 0.73, 0.72875, 0.7275, 0.72625, 0.725, 0.72375, 0.7225, 0.72125, 0.72, 0.71875, 0.7175, 0.71625, 0.715, 0.71375, 0.7125, 0.71125, 0.71, 0.70875, 0.7075, 0.70625, 0.705, 0.70375, 0.7025, 0.70125, 0.7, 0.69875, 0.6975, 0.69625, 0.695, 0.69375, 0.6925, 0.69125, 0.69, 0.68875, 0.6875, 0.68625, 0.685, 0.68375, 0.6825, 0.68125, 0.68, 0.67875, 0.6775, 0.67625, 0.675, 0.67375, 0.6725, 0.67125, 0.67, 0.66875, 0.6675, 0.66625, 0.665, 0.66375, 0.6625, 0.66125, 0.66, 0.65875, 0.6575, 0.65625, 0.655, 0.65375, 0.6525, 0.65125, 0.65, 0.64875, 0.6475, 0.64625, 0.645, 0.64375, 0.6425, 0.64125, 0.64, 0.63875, 0.6375, 0.63625, 0.635, 0.63375, 0.6325, 0.63125, 0.63, 0.62875, 0.6275, 0.62625, 0.625, 0.62375, 0.6225, 0.62125, 0.62, 0.61875, 0.6175, 0.61625, 0.615, 0.61375, 0.6125, 0.61125, 0.61, 0.60875, 0.6075, 0.60625, 0.605, 0.60375, 0.6025, 0.60125, 0.6, 0.59875, 0.5975, 0.59625, 0.595, 0.59375, 0.5925, 0.59125, 0.59, 0.58875, 0.5875, 0.58625, 0.585, 0.58375, 0.5825, 0.58125, 0.58, 0.57875, 0.5775, 0.57625, 0.575, 0.57375, 0.5725, 0.57125, 0.57, 0.56875, 0.5675, 0.56625, 0.565, 0.56375, 0.5625, 0.56125, 0.56, 0.55875, 0.5575, 0.55625, 0.555, 0.55375, 0.5525, 0.55125, 0.55, 0.54875, 0.5475, 0.54625, 0.545, 0.54375, 0.5425, 0.54125, 0.54, 0.53875, 0.5375, 0.53625, 0.535, 0.53375, 0.5325, 0.53125, 0.53, 0.52875, 0.5275, 0.52625, 0.525, 0.52375, 0.5225, 0.52125, 0.52, 0.51875, 0.5175, 0.51625, 0.515, 0.51375, 0.5125, 0.51125, 0.51, 0.50875, 0.5075, 0.50625, 0.505, 0.50375, 0.5025, 0.50125, 0.5, 0.49875, 0.4975, 0.49625, 0.495, 0.49375, 0.4925, 0.49125, 0.49, 0.48875, 0.4875, 0.48625, 0.485, 0.48375, 0.4825, 0.48125, 0.48, 0.47875, 0.4775, 0.47625, 0.475, 0.47375, 0.4725, 0.47125, 0.47, 0.46875, 0.4675, 0.46625, 0.465, 0.46375, 0.4625, 0.46125, 0.46, 0.45875, 0.4575, 0.45625, 0.455, 0.45375, 0.4525, 0.45125, 0.45, 0.44875, 0.4475, 0.44625, 0.445, 0.44375, 0.4425, 0.44125, 0.44, 0.43875, 0.4375, 0.43625, 0.435, 0.43375, 0.4325, 0.43125, 0.43, 0.42875, 0.4275, 0.42625, 0.425, 0.42375, 0.4225, 0.42125, 0.42, 0.41875, 0.4175, 0.41625, 0.415, 0.41375, 0.4125, 0.41125, 0.41, 0.40875, 0.4075, 0.40625, 0.405, 0.40375, 0.4025, 0.40125, 0.4, 0.39875, 0.3975, 0.39625, 0.395, 0.39375, 0.3925, 0.39125, 0.39, 0.38875, 0.3875, 0.38625, 0.385, 0.38375, 0.3825, 0.38125, 0.38, 0.37875, 0.3775, 0.37625, 0.375, 0.37375, 0.3725, 0.37125, 0.37, 0.36875, 0.3675, 0.36625, 0.365, 0.36375, 0.3625, 0.36125, 0.36, 0.35875, 0.3575, 0.35625, 0.355, 0.35375, 0.3525, 0.35125, 0.35, 0.34875, 0.3475, 0.34625, 0.345, 0.34375, 0.3425, 0.34125, 0.34, 0.33875, 0.3375, 0.33625, 0.335, 0.33375, 0.3325, 0.33125, 0.33, 0.32875, 0.3275, 0.32625, 0.325, 0.32375, 0.3225, 0.32125, 0.32, 0.31875, 0.3175, 0.31625, 0.315, 0.31375, 0.3125, 0.31125, 0.31, 0.30875, 0.3075, 0.30625, 0.305, 0.30375, 0.3025, 0.30125, 0.3, 0.29875, 0.2975, 0.29625, 0.295, 0.29375, 0.2925, 0.29125, 0.29, 0.28875, 0.2875, 0.28625, 0.285, 0.28375, 0.2825, 0.28125, 0.28, 0.27875, 0.2775, 0.27625, 0.275, 0.27375, 0.2725, 0.27125, 0.27, 0.26875, 0.2675, 0.26625, 0.265, 0.26375, 0.2625, 0.26125, 0.26, 0.25875, 0.2575, 0.25625, 0.255, 0.25375, 0.2525, 0.25125, 0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, -0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, 0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, -0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, 0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, -0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, 0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, -0.25, 0.24875, 0.2475, 0.24625, 0.245, 0.24375, 0.2425, 0.24125, 0.24, 0.23875, 0.2375, 0.23625, 0.235, 0.23375, 0.2325, 0.23125, 0.23, 0.22875, 0.2275, 0.22625, 0.225, 0.22375, 0.2225, 0.22125, 0.22, 0.21875, 0.2175, 0.21625, 0.215, 0.21375, 0.2125, 0.21125, 0.21, 0.20875, 0.2075, 0.20625, 0.205, 0.20375, 0.2025, 0.20125, 0.2, 0.19875, 0.1975, 0.19625, 0.195, 0.19375, 0.1925, 0.19125, 0.19, 0.18875, 0.1875, 0.18625, 0.185, 0.18375, 0.1825, 0.18125, 0.18, 0.17875, 0.1775, 0.17625, 0.175, 0.17375, 0.1725, 0.17125, 0.17, 0.16875, 0.1675, 0.16625, 0.165, 0.16375, 0.1625, 0.16125, 0.16, 0.15875, 0.1575, 0.15625, 0.155, 0.15375, 0.1525, 0.15125, 0.15, 0.14875, 0.1475, 0.14625, 0.145, 0.14375, 0.1425, 0.14125, 0.14, 0.13875, 0.1375, 0.13625, 0.135, 0.13375, 0.1325, 0.13125, 0.13, 0.12875, 0.1275, 0.12625, 0.125, 0.12375, 0.1225, 0.12125, 0.12, 0.11875, 0.1175, 0.11625, 0.115, 0.11375, 0.1125, 0.11125, 0.11, 0.10875, 0.1075, 0.10625, 0.105, 0.10375, 0.1025, 0.10125, 0.1, 0.09875, 0.0975, 0.09625, 0.095, 0.09375, 0.0925, 0.09125, 0.09, 0.08875, 0.0875, 0.08625, 0.085, 0.08375, 0.0825, 0.08125, 0.08, 0.07875, 0.0775, 0.07625, 0.075, 0.07375, 0.0725, 0.07125, 0.07, 0.06875, 0.0675, 0.06625, 0.065, 0.06375, 0.0625, 0.06125, 0.06, 0.05875, 0.0575, 0.05625, 0.055, 0.05375, 0.0525, 0.05125, 0.05, 0.04875, 0.0475, 0.04625, 0.045, 0.04375, 0.0425, 0.04125, 0.04, 0.03875, 0.0375, 0.03625, 0.035, 0.03375, 0.0325, 0.03125, 0.03, 0.02875, 0.0275, 0.02625, 0.025, 0.02375, 0.0225, 0.02125, 0.02, 0.01875, 0.0175, 0.01625, 0.015, 0.01375, 0.0125, 0.01125, 0.01, 0.00875, 0.0075, 0.00625, 0.005, 0.00375, 0.0025, 0.00125, 0.0, -0.00125, -0.0025, -0.00375, -0.005, -0.00625, -0.0075, -0.00875, -0.01, -0.01125, -0.0125, -0.01375, -0.015, -0.01625, -0.0175, -0.01875, -0.02, -0.02125, -0.0225, -0.02375, -0.025, -0.02625, -0.0275, -0.02875, -0.03, -0.03125, -0.0325, -0.03375, -0.035, -0.03625, -0.0375, -0.03875, -0.04, -0.04125, -0.0425, -0.04375, -0.045, -0.04625, -0.0475, -0.04875, -0.05, -0.05125, -0.0525, -0.05375, -0.055, -0.05625, -0.0575, -0.05875, -0.06, -0.06125, -0.0625, -0.06375, -0.065, -0.06625, -0.0675, -0.06875, -0.07, -0.07125, -0.0725, -0.07375, -0.075, -0.07625, -0.0775, -0.07875, -0.08, -0.08125, -0.0825, -0.08375, -0.085, -0.08625, -0.0875, -0.08875, -0.09, -0.09125, -0.0925, -0.09375, -0.095, -0.09625, -0.0975, -0.09875, -0.1, -0.10125, -0.1025, -0.10375, -0.105, -0.10625, -0.1075, -0.10875, -0.11, -0.11125, -0.1125, -0.11375, -0.115, -0.11625, -0.1175, -0.11875, -0.12, -0.12125, -0.1225, -0.12375, -0.125, -0.12625, -0.1275, -0.12875, -0.13, -0.13125, -0.1325, -0.13375, -0.135, -0.13625, -0.1375, -0.13875, -0.14, -0.14125, -0.1425, -0.14375, -0.145, -0.14625, -0.1475, -0.14875, -0.15, -0.15125, -0.1525, -0.15375, -0.155, -0.15625, -0.1575, -0.15875, -0.16, -0.16125, -0.1625, -0.16375, -0.165, -0.16625, -0.1675, -0.16875, -0.17, -0.17125, -0.1725, -0.17375, -0.175, -0.17625, -0.1775, -0.17875, -0.18, -0.18125, -0.1825, -0.18375, -0.185, -0.18625, -0.1875, -0.18875, -0.19, -0.19125, -0.1925, -0.19375, -0.195, -0.19625, -0.1975, -0.19875, -0.2, -0.20125, -0.2025, -0.20375, -0.205, -0.20625, -0.2075, -0.20875, -0.21, -0.21125, -0.2125, -0.21375, -0.215, -0.21625, -0.2175, -0.21875, -0.22, -0.22125, -0.2225, -0.22375, -0.225, -0.22625, -0.2275, -0.22875, -0.23, -0.23125, -0.2325, -0.23375, -0.235, -0.23625, -0.2375, -0.23875, -0.24, -0.24125, -0.2425, -0.24375, -0.245, -0.24625, -0.2475, -0.24875, -0.25, -0.25125, -0.2525, -0.25375, -0.255, -0.25625, -0.2575, -0.25875, -0.26, -0.26125, -0.2625, -0.26375, -0.265, -0.26625, -0.2675, -0.26875, -0.27, -0.27125, -0.2725, -0.27375, -0.275, -0.27625, -0.2775, -0.27875, -0.28, -0.28125, -0.2825, -0.28375, -0.285, -0.28625, -0.2875, -0.28875, -0.29, -0.29125, -0.2925, -0.29375, -0.295, -0.29625, -0.2975, -0.29875, -0.3, -0.30125, -0.3025, -0.30375, -0.305, -0.30625, -0.3075, -0.30875, -0.31, -0.31125, -0.3125, -0.31375, -0.315, -0.31625, -0.3175, -0.31875, -0.32, -0.32125, -0.3225, -0.32375, -0.325, -0.32625, -0.3275, -0.32875, -0.33, -0.33125, -0.3325, -0.33375, -0.335, -0.33625, -0.3375, -0.33875, -0.34, -0.34125, -0.3425, -0.34375, -0.345, -0.34625, -0.3475, -0.34875, -0.35, -0.35125, -0.3525, -0.35375, -0.355, -0.35625, -0.3575, -0.35875, -0.36, -0.36125, -0.3625, -0.36375, -0.365, -0.36625, -0.3675, -0.36875, -0.37, -0.37125, -0.3725, -0.37375, -0.375, -0.37625, -0.3775, -0.37875, -0.38, -0.38125, -0.3825, -0.38375, -0.385, -0.38625, -0.3875, -0.38875, -0.39, -0.39125, -0.3925, -0.39375, -0.395, -0.39625, -0.3975, -0.39875, -0.4, -0.40125, -0.4025, -0.40375, -0.405, -0.40625, -0.4075, -0.40875, -0.41, -0.41125, -0.4125, -0.41375, -0.415, -0.41625, -0.4175, -0.41875, -0.42, -0.42125, -0.4225, -0.42375, -0.425, -0.42625, -0.4275, -0.42875, -0.43, -0.43125, -0.4325, -0.43375, -0.435, -0.43625, -0.4375, -0.43875, -0.44, -0.44125, -0.4425, -0.44375, -0.445, -0.44625, -0.4475, -0.44875, -0.45, -0.45125, -0.4525, -0.45375, -0.455, -0.45625, -0.4575, -0.45875, -0.46, -0.46125, -0.4625, -0.46375, -0.465, -0.46625, -0.4675, -0.46875, -0.47, -0.47125, -0.4725, -0.47375, -0.475, -0.47625, -0.4775, -0.47875, -0.48, -0.48125, -0.4825, -0.48375, -0.485, -0.48625, -0.4875, -0.48875, -0.49, -0.49125, -0.4925, -0.49375, -0.495, -0.49625, -0.4975, -0.49875, -0.5, -0.50125, -0.5025, -0.50375, -0.505, -0.50625, -0.5075, -0.50875, -0.51, -0.51125, -0.5125, -0.51375, -0.515, -0.51625, -0.5175, -0.51875, -0.52, -0.52125, -0.5225, -0.52375, -0.525, -0.52625, -0.5275, -0.52875, -0.53, -0.53125, -0.5325, -0.53375, -0.535, -0.53625, -0.5375, -0.53875, -0.54, -0.54125, -0.5425, -0.54375, -0.545, -0.54625, -0.5475, -0.54875, -0.55, -0.55125, -0.5525, -0.55375, -0.555, -0.55625, -0.5575, -0.55875, -0.56, -0.56125, -0.5625, -0.56375, -0.565, -0.56625, -0.5675, -0.56875, -0.57, -0.57125, -0.5725, -0.57375, -0.575, -0.57625, -0.5775, -0.57875, -0.58, -0.58125, -0.5825, -0.58375, -0.585, -0.58625, -0.5875, -0.58875, -0.59, -0.59125, -0.5925, -0.59375, -0.595, -0.59625, -0.5975, -0.59875, -0.6, -0.60125, -0.6025, -0.60375, -0.605, -0.60625, -0.6075, -0.60875, -0.61, -0.61125, -0.6125, -0.61375, -0.615, -0.61625, -0.6175, -0.61875, -0.62, -0.62125, -0.6225, -0.62375, -0.625, -0.62625, -0.6275, -0.62875, -0.63, -0.63125, -0.6325, -0.63375, -0.635, -0.63625, -0.6375, -0.63875, -0.64, -0.64125, -0.6425, -0.64375, -0.645, -0.64625, -0.6475, -0.64875, -0.65, -0.65125, -0.6525, -0.65375, -0.655, -0.65625, -0.6575, -0.65875, -0.66, -0.66125, -0.6625, -0.66375, -0.665, -0.66625, -0.6675, -0.66875, -0.67, -0.67125, -0.6725, -0.67375, -0.675, -0.67625, -0.6775, -0.67875, -0.68, -0.68125, -0.6825, -0.68375, -0.685, -0.68625, -0.6875, -0.68875, -0.69, -0.69125, -0.6925, -0.69375, -0.695, -0.69625, -0.6975, -0.69875, -0.7, -0.70125, -0.7025, -0.70375, -0.705, -0.70625, -0.7075, -0.70875, -0.71, -0.71125, -0.7125, -0.71375, -0.715, -0.71625, -0.7175, -0.71875, -0.72, -0.72125, -0.7225, -0.72375, -0.725, -0.72625, -0.7275, -0.72875, -0.73, -0.73125, -0.7325, -0.73375, -0.735, -0.73625, -0.7375, -0.73875, -0.74, -0.74125, -0.7425, -0.74375, -0.745, -0.74625, -0.7475, -0.74875, -0.75])
inside = (xs >= ALPHA) & (xs <= BETA)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.axhspan(-S_TOY / 2, S_TOY / 2, color=C['data_l'], zorder=0)
    ax.plot(xs[inside], err[inside], color=C['data'], lw=2)
    ax.text(1.0, 0.33, 'inside: at most s/2 = 0.25', ha='center', fontsize=12, fontweight='bold', color=C['data'])
    a = 1.0 if k == 2 else 0.0
    ax.plot(xs[xs < ALPHA], err[xs < ALPHA], color=C['danger'], lw=2.4, alpha=a)
    ax.plot(xs[xs > BETA], err[xs > BETA], color=C['danger'], lw=2.4, alpha=a)
    ax.annotate('clipped: grows', (2.95, -0.47), (0.55, -0.66), fontsize=12, fontweight='bold', color=C['danger'], alpha=a, va='center', arrowprops=dict(arrowstyle='-|>', color=C['danger'], lw=1.4, alpha=a))
    for x in (ALPHA, BETA):
        ax.axvline(x, color=C['memory'], lw=1.2, ls='--')
    ax.text(ALPHA, 0.72, 'α', ha='center', fontsize=13, fontweight='bold', color=C['memory'])
    ax.text(BETA, 0.72, 'β', ha='center', fontsize=13, fontweight='bold', color=C['memory'])
    ax.axhline(0, color=C['neutral'], lw=0.8)
    ax.set_xlim(-1.75, 3.25)
    ax.set_ylim(-0.85, 0.85)
    ax.set_xlabel('Real value x (3-bit grid, s = 0.5)')
    ax.set_ylabel('Error after round trip')
    sp.save(fig, out / f'rounderr_step{k}.svg')
zp_rows = [dict(label='Symmetric weights (z = 0)', value=59.1, color='danger', text='59.1'), dict(label='Asymmetric weights (z ≠ 0)', value=70.3, color='data', text='70.3'), dict(label='Float', value=70.9, color='neutral', text='70.9')]
bars_save(out / 'zeropoint.svg', zp_rows, [1, 1, 1], xlim=(0, 82), size=(6.6, 1.25), label_size=15, text_size=15, axis_size=14, xlabel='MobileNet-v1 top-1 (%), INT8 post-training, per-channel weights')
fig, axes = plt.subplots(1, 2, figsize=(5.9, 1.75))
for ax, (lo, hi, qlo, qhi, zq, title, ck) in zip(axes, ((-1.0, 1.0, -127, 127, 0, f'Symmetric: z = {0}', 'memory'), (-0.5, 1.5, -128, 127, -64, 'Asymmetric: z = -64'.replace('-', '−'), 'data'))):
    ax.plot([0, 1], [1, 1], color=C['text'], lw=1.5)
    ax.plot([0, 1], [0, 0], color=C['neutral'], lw=1.5)
    xr0 = (0 - lo) / (hi - lo)
    xq0 = (zq - qlo) / (qhi - qlo)
    for xr, xq, lab, qlab, hl in ((0, 0, f'α = {lo:g}', f'{qlo}', False), (xr0, xq0, '0', f'{zq}', True), (1, 1, f'β = {hi:g}', f'{qhi}', False)):
        col = C['danger'] if hl else C[ck]
        ax.annotate('', (xq, 0.06), (xr, 0.94), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6))
        ax.plot([xr], [1], 'o', ms=7, color=col)
        ax.plot([xq], [0], 's', ms=6, color=col)
        ax.text(xr, 1.18, lab.replace('-', '−'), ha='center', fontsize=12, fontweight='bold', color=col)
        ax.text(xq, -0.42, qlab.replace('-', '−'), ha='center', fontsize=12, fontweight='bold', color=col)
    ax.text(0.5, 1.62, title, ha='center', fontsize=13, fontweight='bold', color=C[ck])
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.55, 1.85)
    ax.axis('off')
axes[0].text(-0.16, 1, 'real', ha='right', va='center', fontsize=12, fontweight='bold')
axes[0].text(-0.16, 0, 'INT8', ha='right', va='center', fontsize=12, fontweight='bold', color=C['neutral'])
fig.tight_layout(w_pad=2.0)
sp.save(fig, out / 'calranges.svg')
hist_px = [238, 228, 218, 200, 175, 145, 115, 90, 70, 52, 40, 30, 23, 18, 14, 11, 9, 7, 6, 4, 3]
hist = np.array(hist_px) / 240
for k in (1,):
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    for i, h in enumerate(hist):
        tail = 0.1 * i >= 1.1 - 1e-09
        ax.bar(0.1 * i + 0.05, h, width=0.092, color=C['danger_l'] if tail else C['memory_l'], edgecolor=C['danger'] if tail else C['memory'], lw=0.8)
    if k == 1:
        cut, full = (1.1, 2.5)
        ax.axvline(cut, color=C['data'], lw=1.8, ls='--')
        ax.text(cut + 0.03, 0.9, 'clip at 1.1?', fontsize=12, fontweight='bold', color=C['data'])
        ax.axvline(full, color=C['neutral'], lw=1.8, ls='--')
        ax.text(full - 0.03, 0.9, 'cover all: 2.5', fontsize=12, fontweight='bold', color=C['neutral'], ha='right')
        ax.annotate('rare tail', (1.65, 0.07), (1.75, 0.45), fontsize=12, fontweight='bold', color=C['danger'], arrowprops=dict(arrowstyle='-|>', color=C['danger'], lw=1.4))
    ax.set_xlim(0, 2.55)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel('Activation value, one ResNet-50 layer (illustrative)')
    sp.save(fig, out / f'calhist_step{k}.svg')
pc_series = [('Float', 'neutral', [71.9, 75.2]), ('INT8 per-tensor', 'danger', [0.1, 75.0]), ('INT8 per-channel', 'data', [69.8, 75.1]), ('INT8 per-channel + QAT', 'memory', [71.1, 75.0])]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    shown = [[1, 1], [1, 1], [1, 1], [k >= 2, k >= 2]]
    vbars(ax, ['MobileNetV2', 'ResNet-50'], pc_series, shown, width=0.19, ymax=92)
    for si, (sname, ck, _) in enumerate(pc_series):
        vis = all(shown[si])
        ax.text(1.02, 0.92 - si * 0.16, sname, transform=ax.transAxes, fontsize=11, fontweight='bold', color=C[ck], alpha=1.0 if vis else 0.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('ImageNet top-1 (%)')
    sp.save(fig, out / f'perchannel_step{k}.svg')
syn_rows = [dict(label='Float', value=71.9, color='neutral', text='71.9'), dict(label='INT8 per-tensor', value=0.1, color='danger', text='0.1'), dict(label='INT8 per-channel', value=69.8, color='data', text='69.8'), dict(label='+ QAT', value=71.1, color='memory', text='71.1')]
bars_save(out / 'synth_acc.svg', syn_rows, [1, 1, 1, 1], xlim=(0, 85), size=(4.0, 2.3), xlabel='MobileNetV2 top-1 (%)')
size_b = [0.125, 1.3, 2.7, 6.7, 13.0]
f32 = [25.65, 15.91, 14.43, 13.3, 12.45]
absmax = [87.76, 16.55, 15.11, 14.59, 19.08]
llmint8 = [25.83, 15.93, 14.44, 13.24, 12.45]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.axvspan(6.0, 15, color=C['danger_l'], alpha=0.6 if k == 2 else 0.0, zorder=0)
    ax.plot(size_b, f32, 'o-', color=C['neutral'], lw=2.2, ms=7)
    ax.plot(size_b, absmax, 'o-', color=C['danger'], lw=2.2, ms=7)
    a = 1.0 if k == 2 else 0.0
    ax.plot(size_b, llmint8, 's--', color=C['data'], lw=2.0, ms=6, alpha=a)
    ax.text(19, absmax[-1], 'INT8, one scale\nper tensor: 19.08', fontsize=11, fontweight='bold', color=C['danger'], va='center')
    ax.text(19, 12.2, '32-bit float: 12.45' if k == 1 else '32-bit float and\nLLM.int8(): 12.45', fontsize=11, fontweight='bold', color=C['neutral'] if k == 1 else C['data'], va='center')
    ax.text(6.3, 60, 'outlier features\nin every layer', fontsize=10.5, fontweight='bold', color=C['danger'], alpha=a)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.09, 15)
    ax.set_ylim(10, 100)
    ax.set_xticks(size_b)
    ax.set_xticklabels([f'{x * 10 ** 3:g}M' if x < 1 else f'{x:g}B' for x in size_b])
    ax.set_yticks([10, 15, 20, 30, 50, 100])
    ax.set_yticklabels(['10', '15', '20', '30', '50', '100'])
    ax.minorticks_off()
    ax.set_xlabel('Parameters')
    ax.set_ylabel('C4 perplexity (lower\nis better, log scale)')
    ax.grid(color=C['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'llmint8_step{k}.svg')
cliff_series = [('Float', 'neutral', [75.2, 71.9]), ('INT8 per-channel PTQ', 'data', [75.1, 69.8]), ('INT4 per-tensor PTQ', 'danger', [0.2, 0.1]), ('INT4 per-channel PTQ', 'compute', [54.0, 0.1]), ('INT4 per-channel QAT', 'memory', [73.2, 62.0])]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    vis = [k >= 1, k >= 1, k >= 2, k >= 2, k >= 3]
    shown = [[v, v] for v in vis]
    vbars(ax, ['ResNet-50', 'MobileNetV2'], cliff_series, shown, width=0.155, ymax=92)
    for si, (sname, ck, _) in enumerate(cliff_series):
        ax.text(1.02, 0.95 - si * 0.15, sname, transform=ax.transAxes, fontsize=10.5, fontweight='bold', color=C[ck], alpha=1.0 if vis[si] else 0.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('ImageNet top-1 (%)')
    sp.save(fig, out / f'cliff_step{k}.svg')
gptq_rows = [dict(label='FP16', value=8.34, color='neutral', text='8.34'), dict(label='4-bit, round to nearest', value=10.54, color='danger', text='10.54'), dict(label='4-bit, GPTQ', value=8.37, color='data', text='8.37'), dict(label='3-bit, round to nearest', value=16, color='danger', text='', tx=10.6), dict(label='3-bit, GPTQ', value=8.68, color='data', text='8.68')]
fig, ax = bars(None, gptq_rows, [1] * 5, xlim=(0, 16), size=(5.8, 2.5), xlabel='OPT-175B WikiText2 perplexity (lower is better)', boxed=True)
for dx in (13.6, 14.0):
    ax.plot([dx - 0.15, dx + 0.15], [0.69, 1.31], color='white', lw=3, zorder=5)
ax.text(0.3, 1, '7.3 thousand: bar cut off', va='center', fontsize=12, fontweight='bold', color='white', zorder=6)
sp.save(fig, out / 'gptq.svg')
awq_rows = [dict(label='FP16', value=10.86, color='neutral', text='10.86'), dict(label='INT3, round to nearest', value=23.54, color='danger', text='23.54'), dict(label='Keep 1% by weight size', value=22.37, color='compute', text='22.37'), dict(label='Keep 1% by activation size', value=11.39, color='data', text='11.39')]
for k in (1, 2):
    bars_save(out / f'awq_step{k}.svg', awq_rows, [1, 1, k >= 2, k >= 2], xlim=(0, 30), size=(5.8, 2.1), xlabel='OPT-6.7B WikiText perplexity (lower is better)')
RIDGE, ROOF = (100.0, 1.0)
xs = np.logspace(-1.5, 4.5, 300)
MS16, MS4 = ('65', '21')
SPEEDUP = 3.1
INT8_X = '2'
dec16, dec4 = (6.0, 6.0 * SPEEDUP)
mlp16, mlp8 = (3000.0, 6000.0)
FS = 10.5
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    roof = np.minimum(ROOF, xs / RIDGE * ROOF)
    ax.plot(xs[xs <= RIDGE], roof[xs <= RIDGE], color=C['memory'], lw=3)
    ax.plot(xs[xs >= RIDGE], roof[xs >= RIDGE], color=C['compute'], lw=3)
    ax.text(0.75, 0.028, 'memory slope', color=C['memory'], fontsize=FS, fontweight='bold', ha='center', va='center', rotation=44)
    ax.plot([dec16], [dec16 / RIDGE], 'o', ms=10, color=C['memory'], mec='white', mew=1.5, zorder=5, alpha=0.45 if k >= 2 else 1.0)
    ax.annotate(f'Decode, FP16:\n{MS16} ms per token', (dec16, dec16 / RIDGE), (8, -6), textcoords='offset points', ha='left', va='top', fontsize=FS, fontweight='bold', color=C['memory'])
    a2 = 1.0 if k >= 2 else 0.0
    if True:
        ax.annotate('', (dec4, dec4 / RIDGE), (dec16 * 1.08, dec16 / RIDGE * 1.08), arrowprops=dict(arrowstyle='-|>', color=C['data'], lw=2.2, alpha=a2))
        ax.plot([dec4], [dec4 / RIDGE], 'o', ms=10, color=C['data'], mec='white', mew=1.5, zorder=5, alpha=a2)
        ax.annotate(f'INT4 weights: {MS4} ms\n(3.1× fewer bytes)', (dec4, dec4 / RIDGE), (-8, 6), textcoords='offset points', ha='right', va='bottom', fontsize=FS, fontweight='bold', color=C['data'], alpha=a2)
    ax.plot([mlp16], [ROOF], 'o', ms=10, color=C['compute'], mec='white', mew=1.5, zorder=5, alpha=0.45 if k >= 3 else 1.0)
    ax.annotate('MLP, large batch:\nFP16 roof, 989', (mlp16, ROOF), (0, -12), textcoords='offset points', ha='center', va='top', fontsize=FS, fontweight='bold', color=C['compute'])
    a3 = 1.0 if k >= 3 else 0.0
    if True:
        xi = xs[(xs >= RIDGE) & (xs <= 2 * RIDGE)]
        ax.plot(xi, xi / RIDGE * ROOF, color=C['memory'], lw=2, ls='--', alpha=a3)
        xr = xs[xs >= 2 * RIDGE]
        ax.plot(xr, np.full_like(xr, 2 * ROOF), color=C['danger'], lw=2.4, ls='--', alpha=a3)
        ax.text(230, 2 * ROOF * 1.25, f'INT8 roof: 1,979 ({INT8_X}×)', color=C['danger'], fontsize=FS, fontweight='bold', va='bottom', alpha=a3)
        ax.annotate('', (mlp8, 2 * ROOF * 0.95), (mlp16 * 1.06, ROOF * 1.04), arrowprops=dict(arrowstyle='-|>', color=C['danger'], lw=2.0, alpha=a3))
        ax.plot([mlp8], [2 * ROOF], 'o', ms=10, color=C['danger'], mec='white', mew=1.5, zorder=5, alpha=a3)
        ax.text(500, 0.008, 'Decode does not move:\nits math stays 16-bit', fontsize=FS - 0.5, color=C['neutral'], fontweight='bold', va='center', alpha=a3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.035, 30000.0)
    ax.set_ylim(0.003, 6)
    ax.minorticks_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Operations per byte moved')
    ax.set_ylabel('Operations per second')
    ax.text(0.01, 0.99, 'Schematic; positions illustrative', transform=ax.transAxes, ha='left', va='top', fontsize=9, color=C['neutral'])
    sp.save(fig, out / f'roofint_step{k}.svg')
W16, W4, KV = (3.0, 0.75, 0.31)

def tot(gb, ms, approx=False):
    return f"{_VALS[gb]} GB, {('≈' if approx else '')}{_VALS[ms]} ms"
segs = [('FP16', [(W16, 'memory', 'FP16 weights, 3 GB'), (KV, 'neutral', '')], '3.31 GB, 65 ms'), ('INT4, fused', [(W4, 'data', ''), (KV, 'neutral', '')], '1.06 GB, 21 ms'), ('INT4, separate\ndequantize pass', [(W4, 'data', ''), (W16, 'danger', 'write FP16, 3 GB'), (W16, 'memory', 'read FP16, 3 GB'), (KV, 'neutral', '')], '7.06 GB, ≈138 ms')]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    for i, (name, parts, tot) in enumerate(segs):
        y = 2 - i
        vis = 1.0 if i < 2 or k == 2 else 0.0
        x = 0
        for v, ck, lab in parts:
            ax.barh(y, v, left=x, color=C[ck], height=0.6, alpha=vis, edgecolor='white')
            if lab:
                ax.text(x + v / 2, y, lab, ha='center', va='center', fontsize=10, color='white', fontweight='bold', alpha=vis)
            x += v
        ax.text(x + 0.1, y, tot, va='center', fontsize=12, fontweight='bold', color=C['danger'] if i == 2 else C['text'], alpha=vis)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'], alpha=vis)
    ax.text(0.0, -0.75, 'green: packed INT4 weights, 0.75 GB;  gray: FP16 KV cache, 0.31 GB', fontsize=10, color=C['neutral'])
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('GB read and written per token at 51.2 GB/s')
    sp.save(fig, out / f'dequant_step{k}.svg')
FRAME_MS = 33.33333333
FRAME_LABEL = '33 ms frame'
px_rows = [dict(label='FP32, CPU', value=105.0, color='neutral', text='105 ms'), dict(label='INT8, CPU', value=63.0, color='compute', text='63 ms (1.7×)'), dict(label='INT8, DSP', value=15.5, color='data', text='15.5 ms (6.8×)')]
bars_save(out / 'pixel2.svg', px_rows, [1, 1, 1], xlim=(0, 150), size=(4.0, 1.5), label_size=15, text_size=15, axis_size=14, vlines=[(FRAME_MS, FRAME_LABEL, 'danger')], boxed=True, xlabel='MobileNetV2 on a Pixel 2, one image (ms)')
stream_rows = [dict(label='MobileNetV2, FP32', value=0.27, color='memory', text='0.27 ms: 1% of a frame'), dict(label='ResNet-50, FP32', value=1.996643125, color='memory', text='2 ms'), dict(label='GPT-2 XL, FP16', value=59.0, color='danger', text='59 ms per token')]
bars_save(out / 'stream.svg', stream_rows, [1, 1, 1], xlim=(0, 80), size=(3.6, 1.6), label_size=15, text_size=15, axis_size=14, vlines=[(FRAME_MS, FRAME_LABEL, 'compute')], boxed=True, xlabel='Read every weight once at 51.2 GB/s (ms)')
stages = ['FP32, CPU', 'Pruned, CPU', 'Pruned INT8,\naccelerator']
memv = [14.0, 10.4, 2.6]
timev = [125.0, 87.0, 20.0]
for k in (2,):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.4, 1.55), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, vals, ck, unit, ttl in ((a1, memv, 'memory', 'MB', 'Weight memory (MB)'), (a2, timev, 'compute', 'ms', 'Frame time (ms)')):
        for i, v in enumerate(vals):
            a = 1.0 if i < k else 0.0
            y = 2 - i
            ax.barh(y, v, color=C[ck], height=0.62, alpha=a)
            ax.text(v + max(vals) * 0.04, y, f'{v:g} {unit}', ha='left', va='center', fontsize=13, fontweight='bold', color=C[ck], alpha=a)
        ax.set_xlim(0, max(vals) * 1.6)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color(C['neutral'])
        ax.set_title(ttl, fontsize=13, fontweight='bold')
    a1.set_yticks(range(3))
    a1.set_yticklabels(stages[::-1], fontsize=13)
    a1.tick_params(axis='y', length=0)
    a2.tick_params(axis='y', length=0)
    a1.set_ylim(-0.5, 2.5)
    a2.axvline(FRAME_MS, color=C['danger'], lw=1.5, ls='--')
    a2.text(FRAME_MS + 3, -0.45, '33 ms budget', color=C['danger'], fontsize=12.5, fontweight='bold', ha='left', va='bottom')
    fig.text(0.01, -0.08, 'Illustrative scenario (Example 10.1)', fontsize=12.5, color=C['neutral'])
    sp.save(fig, out / f'waterfall_step{k}.svg')
fig, axes = plt.subplots(1, 2, figsize=(5.8, 1.55))
for ax, key, title, ck in ((axes[0], 'tg', 'Generate (tg 128), tokens/s', 'memory'), (axes[1], 'pp', 'Prompt (pp 512), tokens/s', 'compute')):
    rows = [('F16', f'lb_{key}16'), ('Q8_0', f'lb_{key}8'), ('Q4_0', f'lb_{key}4')]
    top = max((_NUMS[k] for _, k in rows))
    for i, (lab, k) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, _NUMS[k], color=C[ck] if lab == 'Q4_0' else C['neutral_l'], height=0.62)
        ax.text(_NUMS[k] + top * 0.02, y, _VALS[k], va='center', fontsize=13, fontweight='bold', color=C[ck] if lab == 'Q4_0' else C['text'])
        ax.text(-top * 0.03, y, lab, va='center', ha='right', fontsize=13, fontweight='bold', color=C['text'])
    ax.set_xlim(0, top * 1.32)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', color=C[ck])
    ax.set_xticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.6)
sp.save(fig, out / 'llamabench.svg')
print('figures written')
