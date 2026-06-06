import re
from pathlib import Path
from mlsysim import Models, Hardware
from mlsysim.core.units import *
from mlsysim.physics import calc_activation_memory, model_memory
from mlsysim.fmt import fmt

batch_size = 8
seq_len = 1024
bytes_per_val = 2
act_fp16_gb = calc_activation_memory(
    n_layers=Models.Language.GPT2.layers,
    seq_len=seq_len,
    batch_size=batch_size,
    hidden_dim=Models.Language.GPT2.hidden_dim,
    precision_bytes=bytes_per_val,
    strategy="none",
).m_as(GB)
act_fp32_gb = act_fp16_gb * 2
checkpoint_factor = 4
params_fp32_gb = model_memory(Models.Language.GPT2.parameters, BYTES_FP32, GB)
grads_fp32_gb = params_fp32_gb
adam_fp32_gb = 2 * params_fp32_gb
total_fp32_gb = params_fp32_gb + grads_fp32_gb + adam_fp32_gb + act_fp32_gb
params_fp16_gb = model_memory(Models.Language.GPT2.parameters, BYTES_FP16, GB)
grads_fp16_gb = params_fp16_gb
master_weights_gb = params_fp32_gb
adam_amp_gb = adam_fp32_gb
act_fp16_gb = act_fp32_gb * (BYTES_FP16.m_as(byte) / BYTES_FP32.m_as(byte))
total_amp_gb = params_fp16_gb + grads_fp16_gb + master_weights_gb + adam_amp_gb + act_fp16_gb
static_mem_gb = params_fp16_gb + grads_fp16_gb + master_weights_gb + adam_amp_gb
act_ckpt_gb = act_fp16_gb / checkpoint_factor
total_ckpt_gb = static_mem_gb + act_ckpt_gb
v100_mem_gb = Hardware.Cloud.V100.memory.capacity.m_as(GB)
print(f"fp32={total_fp32_gb:.1f} amp={total_amp_gb:.1f} ckpt={total_ckpt_gb:.1f} v100={v100_mem_gb:.1f}")
