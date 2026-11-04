"""Named serving scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ... import scenario_types as _st


Llama3_70B_DecodeH100 = _st.DecodeScenario(
    name="Llama-3 70B batch-1 decode on one H100",
    description="FP16 weights streamed once per token on one H100-equivalent path "
    "(capacity assumed, to isolate arithmetic from weight transfer).",
    workload=_Models.Language.Llama3_70B,
    hardware=_Hardware.Cloud.H100,
)


Llama2_70B_DecodeH100 = Llama3_70B_DecodeH100.variant(
    name="Llama-2 70B batch-1 decode on one H100",
    description="The batch-1 decode scenario for the Llama-2 70B serving replica (FP16 weights).",
    workload=_Models.Language.Llama2_70B,
)


Llama3_70B_DecodeH100Batch64 = _st.DecodeScenario(
    name="Llama-3 70B decode at batch 64 on one H100",
    description="The batch-1 decode scenario with 64 requests sharing each weight read, "
    "which raises the weight-only arithmetic intensity 64-fold.",
    workload=_Models.Language.Llama3_70B,
    hardware=_Hardware.Cloud.H100,
    batch_size=64,  # scenario assumption: concurrent decode requests
)


Llama3_70B_ServingH100 = _st.ServingScenario(
    name="Llama-3 70B single-request serving on one H100",
    description="ServingModel latency (weights, KV-cache reads, launch overhead) for one "
    "request at a 2,048-token context; .variant(hardware=...) compares parts.",
    workload=_Models.Language.Llama3_70B,
    hardware=_Hardware.Cloud.H100,
    seq_len=2048,  # scenario assumption: serving context length
    batch_size=1,  # scenario assumption: single-request decode
)
