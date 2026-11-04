"""Named optimization scenarios for deployment.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....reference_stats.registry import ReferenceStats as _ReferenceStats
from ....systems.registry import Fabrics as _Fabrics, NetworkEnergy as _NetworkEnergy
from ... import scenario_types as _st
from ....systems.registry import Pods as _Pods
from ....core.units import (
    VIDEO_1080P_WIDTH as _VIDEO_1080P_WIDTH,
    VIDEO_1080P_HEIGHT as _VIDEO_1080P_HEIGHT,
    VIDEO_4K_WIDTH as _VIDEO_4K_WIDTH,
    VIDEO_4K_HEIGHT as _VIDEO_4K_HEIGHT,
    VIDEO_BYTES_PER_PIXEL_RGB as _VIDEO_BYTES_PER_PIXEL_RGB,
    VIDEO_FPS_STANDARD as _VIDEO_FPS_STANDARD,
)
from ..base.applications import SmartDoorbell


RoboticArmRegionalCloud = _st.PropagationScenario(
    name="A robot arm's safety stop served from a regional cloud",
    description="The model runs 1,500 km away; the stop must happen within 10 ms of seeing a person.",
    distance=Q_(1500, "km"),  # scenario assumption: distance to the cloud region
    budget=Q_(10, "ms"),  # scenario assumption: safety response budget
)


CrossCountryFiberPath = _st.PropagationScenario(
    name="A sensor packet from California to a data center in Virginia",
    description="Straight-line coast-to-coast distance against a sub-10 ms control budget.",
    distance=Q_(3600, "km"),  # scenario assumption: California to Virginia, straight line
    budget=Q_(10, "ms"),  # scenario assumption: real-time control budget
)


FactoryCameras1080p = _st.SensorUplinkScenario(
    name="100 factory cameras at 1080p on a 10 Gb/s link",
    description="Raw RGB video from a quality-control floor against the site link, and defect records "
    "an edge server sends instead.",
    frame_width=_VIDEO_1080P_WIDTH,
    frame_height=_VIDEO_1080P_HEIGHT,
    bytes_per_pixel=_VIDEO_BYTES_PER_PIXEL_RGB,
    frame_rate=_VIDEO_FPS_STANDARD,
    streams=100,  # scenario assumption: cameras on the floor
    link_bandwidth=_Fabrics.Ethernet_10G.bandwidth,
    event_size=Q_(1, "KB"),  # scenario assumption: one defect record
    event_rate=Q_(20, "1/second"),  # scenario assumption: detections per second across the floor
)


Drone4KObstacleOffload = _st.OffloadDeadlineScenario(
    name="A drone offloading 4K obstacle avoidance over cellular",
    description="One uncompressed 4K RGB frame per decision over a 100 Mb/s uplink, against the 60 FPS frame interval.",
    payload=_VIDEO_4K_WIDTH * _VIDEO_4K_HEIGHT * _VIDEO_BYTES_PER_PIXEL_RGB,
    uplink_bandwidth=Q_(100, "Mbit/s"),  # scenario assumption: cellular uplink
    network_latency=Q_(100, "ms"),  # scenario assumption: cloud round trip
    remote_compute_time=Q_(10, "ms"),  # scenario assumption: remote inference
    frame_rate=Q_(60, "Hz"),  # scenario assumption: camera frame rate
)


ResNet50A100Batch1 = _st.SingleNodeInferenceScenario(
    name="ResNet-50, one image, on an A100 at FP16 and peak rates",
    description="Batch-1 inference with weights and activations in 16 bits; efficiency 1, so both "
    "times are lower bounds.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.A100,
    precision="fp16",
    efficiency=1.0,  # scenario assumption: peak rates (lower bounds)
)


ResNet50PhoneNpuBatch1 = _st.SingleNodeInferenceScenario(
    name="ResNet-50, one image, on a flagship phone NPU at INT8 and peak rates",
    description="Batch-1 inference on the iPhone 15 Pro's neural engine; efficiency 1.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Mobile.iPhone15Pro,
    precision="int8",
    efficiency=1.0,  # scenario assumption: peak rates (lower bounds)
)


TPUv4Pod4096 = _st.AcceleratorPodScenario(
    name="A TPU v4 pod of 4,096 chips",
    description="Aggregate BF16 peak and HBM of the pod envelope's chips.",
    pod=_Pods.TPUv4_4096,
    accelerator=_Hardware.Cloud.TPUv4,
)


IllustrativeMemoryWall = _st.GrowthGapScenario(
    name="Compute against memory bandwidth, illustrative rates",
    description="Compute doubling every 18 months against bandwidth growing 20 percent a year, over a decade.",
    compute_doubling_period=Q_(18, "month"),  # scenario assumption: illustrative rate
    bandwidth_growth=0.20,  # scenario assumption: illustrative rate
    horizon=Q_(10, "year"),
)


AlwaysOnObjectDetectorPhone = _st.BatteryBudgetScenario(
    name="An always-on object detector on a phone",
    description="A background detector drawing 2 W continuously against a 15 Wh battery and a day.",
    battery_energy=_Hardware.Mobile.iPhone15Pro.battery_capacity,
    power_draw=_ReferenceStats.MobilePower.ObjectDetector,
    required_runtime=Q_(24, "hour"),  # scenario assumption: a full day of use
)


UnoptimizedVisionModelPhone = _st.ThermalCeilingScenario(
    name="An unoptimized vision model on a passively cooled phone",
    description="A 12 W model heats the SoC until it throttles; a 4x lighter redesign lands at the sustained ceiling.",
    power=Q_(12, "watt"),  # scenario assumption: unoptimized model power
    reduction_factor=4.0,  # scenario assumption: lighter INT8 redesign
    sustained_limit=_ReferenceStats.MobilePower.MobileNpuTypical,
    heating_rate=Q_(1, "delta_degC/second"),  # scenario assumption: representative trace
    trip_time=Q_(60, "second"),  # scenario assumption: time to the trip point
    trip_temperature=Q_(80, "degC"),  # scenario assumption: thermal trip point
    burst_rate=100.0,  # scenario assumption: frames per second before
    throttled_rate=30.0,  # scenario assumption: frames per second after
)


MobileNetV2PhoneBatch1 = _st.SingleNodeInferenceScenario(
    name="MobileNetV2, one image, on a flagship phone",
    description="Whole-phone energy over the batch-1 window (solver defaults).",
    workload=_Models.Vision.MobileNetV2,
    hardware=_Hardware.Mobile.iPhone15Pro,
)


EnergyPerInferenceTiers = _st.EnergyPerInferenceScenario(
    name="Energy per inference across deployment tiers",
    description="Device-level estimates for three vision runs, between a TinyML keyword-spotting "
    "anchor and a cloud LLM-query anchor; the tasks and boundaries differ.",
    anchors={
        "kws": Q_(10, "uJ"),  # scenario assumption: KWS on dedicated low-power audio silicon
        "llm_query": Q_(1, "kJ"),  # scenario assumption: cloud LLM query incl. data-center overhead
    },
    runs={
        "mobilenet_phone": MobileNetV2PhoneBatch1,
        "resnet_edge": _st.SingleNodeInferenceScenario(
            name="ResNet-50, one image, on a Jetson Orin NX",
            description="Solver defaults.",
            workload=_Models.Vision.ResNet50,
            hardware=_Hardware.Edge.JetsonOrinNX,
        ),
        "resnet_cloud": _st.SingleNodeInferenceScenario(
            name="ResNet-50, one image, on an A100",
            description="Solver defaults.",
            workload=_Models.Vision.ResNet50,
            hardware=_Hardware.Cloud.A100,
        ),
    },
)


RadioUploadVsMobileNetV2 = _st.OffloadEnergyScenario(
    name="Sending 1 MB by radio against one MobileNetV2 inference on the phone",
    description="Wi-Fi/LTE transmission energy against the NPU alone and against the whole phone.",
    payload=Q_(1, "MB"),  # scenario assumption: one raw sensor payload
    transmit_energy_per_byte=_NetworkEnergy.Per5gMb,
    local_run=MobileNetV2PhoneBatch1,
)


SmartDoorbellFrame = _st.SingleNodeInferenceScenario(
    name="One smart-doorbell frame on the ESP32-S3",
    description="The doorbell's wake-vision model at 16 bits, decomposed into compute and overhead.",
    workload=SmartDoorbell.workload,
    hardware=SmartDoorbell.system,
    precision="fp16",
)


SmartDoorbellResponsiveChime = SmartDoorbell.model_copy(
    update={
        "name": "Smart Doorbell, responsive chime",
        "sla_latency": Q_(50, "ms"),  # scenario assumption: tightened chime budget
    }
)


EmergencyBraking100kmh = _st.TravelDuringLatencyScenario(
    name="Emergency braking at 100 km/h under a 100 ms budget",
    description="Distance covered before braking begins, against a cloud round trip of 50 to 150 ms.",
    speed=Q_(100, "km/hour"),  # scenario assumption
    latency=Q_(100, "ms"),  # scenario assumption: end-to-end budget
    remote_delay_low=Q_(50, "ms"),  # scenario assumption: cloud round trip, best case
    remote_delay_high=Q_(150, "ms"),  # scenario assumption: cloud round trip, worst case
)


VoiceAssistantBillionDevices = _st.WakeWordFleetScenario(
    name="Wake-word detection for one billion voice assistants",
    description="Streaming every microphone to cloud detection against detecting on device and "
    "serving only post-wake queries.",
    devices=1e9,  # scenario assumption
    audio_sample_rate=Q_(16, "kHz"),  # scenario assumption: speech audio
    audio_sample_bits=Q_(16, "bit"),  # scenario assumption
    queries_per_day=20.0,  # scenario assumption: queries per device
    query_audio=Q_(5, "second"),  # scenario assumption: audio uploaded per query
    accelerator_time_per_query=Q_(200, "ms"),  # scenario assumption: GPU time per query
    accelerators_per_datacenter=10_000,  # scenario assumption
    waking_hours=Q_(16, "hour"),  # scenario assumption
    peak_multiplier=3.0,  # scenario assumption: peak over waking average
    cloud_detection_cost=Q_(0.50, "USD/year"),  # scenario assumption: per device
    local_power_low=Q_(0.1, "mW"),  # scenario assumption: on-device detector
    local_power_high=Q_(1, "mW"),  # scenario assumption
    local_detection_cost=Q_(0.01, "USD/year"),  # scenario assumption: per device
)


PhoneCameraPipeline = _st.PipelineSpeedupScenario(
    name="A phone camera pipeline with a 10x faster ML stage",
    description="Image processing, ML scene classification, and tone mapping in series.",
    stages={
        "isp": Q_(100, "ms"),  # scenario assumption: ISP and auto-exposure
        "ml": Q_(60, "ms"),  # scenario assumption: ML scene classification
        "post": Q_(40, "ms"),  # scenario assumption: tone mapping and HDR merge
    },
    accelerated="ml",
    speedup=10.0,  # scenario assumption: 10x faster model
)
