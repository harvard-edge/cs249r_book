#!/usr/bin/env python3
"""Canonical inputs, matching KvBudget and original plot in 13_model_serving.qmd.
Run PYTHONPATH=mlsysim /tmp/venv/bin/python <this-file>.
"""
import json
from mlsysim import Hardware,Models
from mlsysim.core.units import BYTES_FP16,BYTES_INT4,byte,GB
model=Models.Language.Llama3_70B;device=Hardware.Cloud.H100
payload=(2*model.layers*model.kv_heads*(model.hidden_dim//model.heads)*BYTES_FP16).to(byte).magnitude
budget=(device.memory.capacity-model.size_in_bytes(BYTES_INT4)).to(GB).magnitude
assert payload==327680
print(json.dumps({'bytes_per_token_per_request':payload,'cache_budget_gb':budget,'decimal_gb_bytes':(1*GB).to(byte).magnitude}))
