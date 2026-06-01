# LEGO MLSysIM Source-of-Truth Audit

This is an advisory work queue for values that likely define
hardware, systems, infrastructure, pricing, models, datasets,
storage subsystems, or scenarios in LEGO LOAD stages.

Findings: 1047
By target:
   285  Scenarios.* or Ops.*
   216  Models.* or Scenarios.TrainingRuns
   170  Scenarios.*
   116  Infrastructure.Pricing.* or Scenarios.*
   111  Datasets.* or Scenarios.DataWorkloads
    49  Hardware.*
    31  Infrastructure.*
    21  Systems.*
    13  Hardware.* or Scenarios.*
    11  Systems.Storage or Datasets.*
     5  Systems.Clusters or Systems.Nodes
     5  Hardware.* or Hardware.Tech.*
     4  Systems.Storage or Scenarios.*
     3  Infrastructure.Pricing.Labeling or Scenarios.*
     3  Infrastructure.* or Scenarios.Sustainability
     2  Scenarios.TrainingRuns or Systems.Reliability
     1  Systems.Storage
     1  Infrastructure.* or Scenarios.*
Top chapters:
   151  vol1/model_serving
    83  vol1/data_selection
    72  vol1/training
    57  vol1/data_engineering
    53  vol1/nn_computation
    50  vol1/benchmarking
    50  vol1/ml_systems
    38  vol1/nn_architectures
    36  vol1/ml_ops
    36  vol2/inference
    33  vol2/data_storage
    33  vol2/sustainable_ai
    32  vol2/ops_scale
    30  vol1/hw_acceleration
    29  vol1/ml_workflow
    27  vol1/responsible_engr
    22  vol2/distributed_training
    21  vol2/security_privacy
    18  vol2/fleet_orchestration
    18  vol2/performance_engineering
Top reasons:
   282  scenario/workload policy
   211  model/workload specification
   170  unit-bearing scenario input
   115  economic input or scenario price
   107  dataset/workload specification
    49  hardware-related quantitative input
    31  infrastructure input
    21  system-level fact
    16  scenario/profile input
    11  storage/data fact
     5  fleet/topology fact
     5  hardware specification
     5  workload compute requirement
     4  dataset/workload size
     3  storage observation or utilization scenario
     3  human-labeling price input
     3  infrastructure/sustainability fact
     3  storage-related scenario input
     2  checkpoint policy/workload cadence
     1  storage subsystem fact

## vol1/backmatter/appendix_algorithm.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd:136` | `SparseEmbeddingExample` | `vocab_size` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100 * THOUSAND` |
| `book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd:363` | `GPT2TrainingMem` | `batch_small` | `Scenarios.* or Ops.*` | scenario/workload policy | `8` |
| `book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd:364` | `GPT2TrainingMem` | `batch_large` | `Scenarios.* or Ops.*` | scenario/workload policy | `64` |
| `book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd:365` | `GPT2TrainingMem` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |

## vol1/backmatter/appendix_assumptions.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd:60` | `NapkinMath` | `model_params_b` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `7` |
| `book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd:61` | `NapkinMath` | `bytes_per_param_q` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `BYTES_FP16 # BF16 weights + BYTES_FP16 # BF16 gradients + 3 * BYTES_FP32` |

## vol1/backmatter/appendix_data.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:56` | `DataGravity` | `tb_bytes` | `Scenarios.*` | unit-bearing scenario input | `(1 * TB).to(byte).magnitude` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:57` | `DataGravity` | `pb_bytes` | `Scenarios.*` | unit-bearing scenario input | `(1 * PB).to(byte).magnitude` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:58` | `DataGravity` | `bw_1g` | `Scenarios.*` | unit-bearing scenario input | `(1 * Gbps).to(bit / second).magnitude` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:61` | `DataGravity` | `truck_load_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `8` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:62` | `DataGravity` | `truck_transit_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:63` | `DataGravity` | `truck_unload_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `8` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:160` | `SerializationCost` | `csv_speed` | `Scenarios.*` | unit-bearing scenario input | `100 * (MB / second)` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:161` | `SerializationCost` | `parquet_speed` | `Scenarios.*` | unit-bearing scenario input | `1000 * (MB / second)` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:164` | `SerializationCost` | `proto_speed` | `Scenarios.*` | unit-bearing scenario input | `300 * (MB / second)` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:331` | `DataAlgebra` | `row_size_q` | `Scenarios.*` | unit-bearing scenario input | `1 * KB` |
| `book/quarto/contents/vol1/backmatter/appendix_data.qmd:333` | `DataAlgebra` | `join_table_q` | `Scenarios.*` | unit-bearing scenario input | `1 * TB` |

## vol1/backmatter/appendix_machine.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:376` | `A100RooflineExample` | `relu_intensity` | `Hardware.*` | hardware-related quantitative input | `0.25 * (flop / byte)` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:648` | `TrainingTimeRef` | `p_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1 * BILLION` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:756` | `LittlesLawExample` | `lambda_qps_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:757` | `LittlesLawExample` | `w_latency` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:759` | `LittlesLawExample` | `mem_per_req` | `Scenarios.*` | unit-bearing scenario input | `1 * GB` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:760` | `LittlesLawExample` | `gpu_mem` | `Hardware.*` | hardware-related quantitative input | `24 * GB` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1062` | `BandwidthLatencySetup` | `ping_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1063` | `BandwidthLatencySetup` | `data` | `Scenarios.*` | unit-bearing scenario input | `1 * KB` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1102` | `BandwidthLatencyExample` | `packet_data` | `Scenarios.*` | unit-bearing scenario input | `1 * KB` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1104` | `BandwidthLatencyExample` | `large_data` | `Scenarios.*` | unit-bearing scenario input | `1 * GB` |
| `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1105` | `BandwidthLatencyExample` | `ping_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |

## vol1/benchmarking

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:176` | `ComponentLatencyExample` | `model_latency_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:378` | `StatisticalConfidenceTrap` | `baseline_accuracy` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.95` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:379` | `StatisticalConfidenceTrap` | `compressed_accuracy` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.94` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:670` | `RooflineExamples` | `resnet_ai` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `300.0` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:671` | `RooflineExamples` | `resnet_util_min` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `85` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:672` | `RooflineExamples` | `resnet_util_max` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `90` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:678` | `RooflineExamples` | `bert_util_peak` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.85` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:775` | `BertRoofline` | `batch_1` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:776` | `BertRoofline` | `batch_32` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:1836` | `CompressionModelSpecs` | `_mobilenet_top1` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `72` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:1837` | `CompressionModelSpecs` | `_resnet50_top1` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `76` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:1976` | `InferenceEnergy` | `_latency_fast` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:1977` | `InferenceEnergy` | `_latency_slow` | `Scenarios.*` | unit-bearing scenario input | `100 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:2335` | `ScalingEfficiencyCalc` | `t1_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:2337` | `ScalingEfficiencyCalc` | `tn_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3071` | `ColdStartTransferCalc` | `pcie_effective_bw` | `Hardware.* or Scenarios.*` | scenario/profile input | `25 * (GB / second)` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3247` | `EdgeTPUSpeedupCalc` | `edgetpu_latency` | `Hardware.*` | hardware-related quantitative input | `2 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3248` | `EdgeTPUSpeedupCalc` | `cpu_latency` | `Hardware.*` | hardware-related quantitative input | `15 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3249` | `EdgeTPUSpeedupCalc` | `edgetpu_e2e` | `Hardware.*` | hardware-related quantitative input | `6 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3250` | `EdgeTPUSpeedupCalc` | `cpu_e2e` | `Hardware.*` | hardware-related quantitative input | `18 * ms` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3251` | `EdgeTPUSpeedupCalc` | `edgetpu_power` | `Infrastructure.*` | infrastructure input | `500 * milliwatt` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:3252` | `EdgeTPUSpeedupCalc` | `cpu_power` | `Infrastructure.*` | infrastructure input | `120 * milliwatt` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4524` | `MobileNetINT8Calc` | `ece_example_confidence_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `90` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4529` | `MobileNetINT8Calc` | `confidence_threshold_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `85` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4645` | `LLMThroughputCalc` | `response_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `750` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4646` | `LLMThroughputCalc` | `slow_toks` | `Scenarios.* or Ops.*` | scenario/workload policy | `25` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4647` | `LLMThroughputCalc` | `fast_toks` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5098` | `FallaciesPitfallsSetup` | `benchmark_accuracy_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `92` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5099` | `FallaciesPitfallsSetup` | `production_accuracy_low_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `78` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5100` | `FallaciesPitfallsSetup` | `production_accuracy_high_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `82` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5109` | `FallaciesPitfallsSetup` | `energy_increase_pct` | `Infrastructure.*` | infrastructure input | `40` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5111` | `FallaciesPitfallsSetup` | `accuracy_improvement_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `2.1` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5113` | `FallaciesPitfallsSetup` | `rec_accuracy_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `94` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5116` | `FallaciesPitfallsSetup` | `high_throughput_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1200` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5117` | `FallaciesPitfallsSetup` | `low_throughput_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5118` | `FallaciesPitfallsSetup` | `high_power` | `Infrastructure.*` | infrastructure input | `420 * watt` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5119` | `FallaciesPitfallsSetup` | `low_power` | `Infrastructure.*` | infrastructure input | `180 * watt` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5122` | `FallaciesPitfallsSetup` | `imagenet_error_2010_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `28.2` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5123` | `FallaciesPitfallsSetup` | `imagenet_error_2015_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `3.57` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5124` | `FallaciesPitfallsSetup` | `imagenet_competition_end_year` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `2017` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5125` | `FallaciesPitfallsSetup` | `imagenet_teams_above_95` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `29` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5126` | `FallaciesPitfallsSetup` | `imagenet_total_teams` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `38` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5127` | `FallaciesPitfallsSetup` | `mnist_accuracy_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `99.8` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5129` | `FallaciesPitfallsSetup` | `edge_power_constraint_x` | `Infrastructure.*` | infrastructure input | `100` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5131` | `FallaciesPitfallsSetup` | `isolated_throughput_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `800` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5132` | `FallaciesPitfallsSetup` | `production_throughput_low_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `400` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5133` | `FallaciesPitfallsSetup` | `production_throughput_high_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `500` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5135` | `FallaciesPitfallsSetup` | `model_inference_low_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5136` | `FallaciesPitfallsSetup` | `model_inference_high_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10` |
| `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:5140` | `FallaciesPitfallsSetup` | `downtime_minutes_month` | `Scenarios.* or Ops.*` | scenario/workload policy | `43` |

## vol1/data_engineering

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:148` | `DataGravity` | `dataset_pb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:240` | `FeedingProblem` | `img_size_bytes` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload size | `IMAGE_DIM_RESNET * IMAGE_DIM_RESNET * IMAGE_CHANNELS_RGB * 4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:838` | `KWSProblemTargets` | `kws_accuracy_target` | `Scenarios.* or Ops.*` | scenario/workload policy | `98` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:891` | `FalsePositiveTarget` | `duty_cycle_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:892` | `FalsePositiveTarget` | `window_sec` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:990` | `BudgetAllocation` | `storage_pct` | `Scenarios.* or Ops.*` | storage observation or utilization scenario | `0.25` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:993` | `BudgetAllocation` | `review_overhead` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.20` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:1048` | `KWSDesignConstraints` | `kws_accuracy_target` | `Scenarios.* or Ops.*` | scenario/workload policy | `98` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:1054` | `KWSDesignConstraints` | `kws_base_model_acc` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `90` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:1305` | `LabelingComputeRatio` | `train_hours_low` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:1306` | `LabelingComputeRatio` | `train_hours_high` | `Scenarios.* or Ops.*` | scenario/workload policy | `6` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2012` | `KWSAcquisitionScale` | `dataset_size_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23.4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2458` | `DataloaderStats` | `gpu_limit_img_sec` | `Systems.*` | system-level fact | `3000` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2586` | `RealtimeCost` | `stream_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2587` | `RealtimeCost` | `stream_cost_per_hr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.05` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2588` | `RealtimeCost` | `batch_cores` | `Scenarios.* or Ops.*` | scenario/workload policy | `200` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2589` | `RealtimeCost` | `batch_window_min` | `Scenarios.* or Ops.*` | scenario/workload policy | `10` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2916` | `EtlEltCost` | `s3_per_tb_mo` | `Systems.Storage or Datasets.*` | storage/data fact | `23` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2919` | `EtlEltCost` | `retention_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `30` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2920` | `EtlEltCost` | `query_cost_per` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2921` | `EtlEltCost` | `etl_datasets` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `3` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2923` | `EtlEltCost` | `schema_change_etl_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:2924` | `EtlEltCost` | `schema_change_elt_minutes` | `Scenarios.* or Ops.*` | scenario/workload policy | `30` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3110` | `KWSQualityThresholds` | `snr_threshold_db` | `Scenarios.* or Ops.*` | scenario/workload policy | `20` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3112` | `KWSQualityThresholds` | `sample_duration_ms_low` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `500` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3113` | `KWSQualityThresholds` | `sample_duration_ms_high` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `800` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3235` | `KWSScalabilityTargets` | `dataset_size_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23.4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3237` | `KWSScalabilityTargets` | `model_size_limit_kb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `64` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3506` | `ActiveLearningBudget` | `dataset_images` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10 * MILLION` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3509` | `ActiveLearningBudget` | `inference_cost_per_image` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.01` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3612` | `KWSAutomatedLabelingScale` | `dataset_size_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23.4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3615` | `KWSAutomatedLabelingScale` | `accuracy_target` | `Scenarios.* or Ops.*` | scenario/workload policy | `98` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3616` | `KWSAutomatedLabelingScale` | `sample_duration_ms_low` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `500` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3617` | `KWSAutomatedLabelingScale` | `sample_duration_ms_high` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `800` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3746` | `KWSStorageEconomics` | `rounded_sample_count_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3747` | `KWSStorageEconomics` | `sample_duration_s` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3748` | `KWSStorageEconomics` | `sample_rate_hz` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `16_000` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3749` | `KWSStorageEconomics` | `bytes_per_sample` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `2` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3818` | `StorageLoading` | `rounded_sample_count_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3819` | `StorageLoading` | `sample_duration_s` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3820` | `StorageLoading` | `sample_rate_hz` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `16_000` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3821` | `StorageLoading` | `bytes_per_sample` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `2` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:3980` | `StorageBandwidth` | `image_size_kb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `150` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4162` | `CompressionTradeoff` | `dataset_gb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4236` | `LifecycleStorageRequirements` | `image_size_kb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `150` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4241` | `LifecycleStorageRequirements` | `feature_reads_per_request` | `Scenarios.* or Ops.*` | scenario/workload policy | `10` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4728` | `DataEngineeringSummaryRecap` | `train_hours_low` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4729` | `DataEngineeringSummaryRecap` | `train_hours_high` | `Scenarios.* or Ops.*` | scenario/workload policy | `6` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4732` | `DataEngineeringSummaryRecap` | `kws_dataset_size_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23.4` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4733` | `DataEngineeringSummaryRecap` | `rounded_sample_count_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4734` | `DataEngineeringSummaryRecap` | `sample_duration_s` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4735` | `DataEngineeringSummaryRecap` | `sample_rate_hz` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `16_000` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4736` | `DataEngineeringSummaryRecap` | `bytes_per_sample` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `2` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4784` | `StorageLoadingRecap` | `rounded_sample_count_m` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `23` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4785` | `StorageLoadingRecap` | `sample_duration_s` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4786` | `StorageLoadingRecap` | `sample_rate_hz` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `16_000` |
| `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:4787` | `StorageLoadingRecap` | `bytes_per_sample` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `2` |

## vol1/data_selection

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:98` | `ScalingAsymmetry` | `gpu_growth_factor` | `Systems.*` | system-level fact | `10.0` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:99` | `ScalingAsymmetry` | `gpu_period_years` | `Systems.*` | system-level fact | `3.0` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:420` | `IronLawSavings` | `factor_model` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1120` | `QualityMultiplier` | `epsilon` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.01` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1323` | `CoresetPractice` | `train_image_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_000_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1325` | `CoresetPractice` | `proxy_epoch_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `5` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1507` | `CurriculumBenchmarks` | `cifar10_baseline_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `150` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1508` | `CurriculumBenchmarks` | `cifar10_curriculum_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `115` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1510` | `CurriculumBenchmarks` | `cifar100_baseline_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `220` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1511` | `CurriculumBenchmarks` | `cifar100_curriculum_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `180` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1513` | `CurriculumBenchmarks` | `imagenet_baseline_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `90` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1514` | `CurriculumBenchmarks` | `imagenet_curriculum_epochs` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `80` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1516` | `CurriculumBenchmarks` | `mentornet_baseline_epochs` | `Scenarios.* or Ops.*` | scenario/workload policy | `90` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1517` | `CurriculumBenchmarks` | `mentornet_curriculum_epochs` | `Scenarios.* or Ops.*` | scenario/workload policy | `70` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1862` | `ActiveLearningRoi` | `cost_per_label` | `Infrastructure.Pricing.Labeling or Scenarios.*` | human-labeling price input | `5.00 * (USD / count)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1863` | `ActiveLearningRoi` | `budget` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500_000 * USD` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:1864` | `ActiveLearningRoi` | `deadline` | `Scenarios.* or Ops.*` | scenario/workload policy | `1 * ureg.month` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2029` | `ClinicalLabelingEconomics` | `review_rate_low` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2030` | `ClinicalLabelingEconomics` | `review_rate_high` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `80` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2031` | `ClinicalLabelingEconomics` | `hourly_rate_low` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2032` | `ClinicalLabelingEconomics` | `hourly_rate_high` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `300` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2033` | `ClinicalLabelingEconomics` | `workweek_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `40` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2146` | `FixmatchLabelEfficiency` | `cifar10_full_acc` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `96.1` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2149` | `FixmatchLabelEfficiency` | `cifar10_fixmatch_4k_acc` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `95.7` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2152` | `FixmatchLabelEfficiency` | `cifar10_fixmatch_250_acc` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `94.9` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2155` | `FixmatchLabelEfficiency` | `cifar10_fixmatch_40_acc` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `88.6` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2162` | `FixmatchLabelEfficiency` | `supervised_compute_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2165` | `FixmatchLabelEfficiency` | `fixmatch_compute_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `250` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2302` | `FoundationCostAmortization` | `scratch_compute_per_task` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2304` | `FoundationCostAmortization` | `pretrain_compute` | `Scenarios.* or Ops.*` | scenario/workload policy | `10000 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2305` | `FoundationCostAmortization` | `finetune_compute_per_task` | `Scenarios.* or Ops.*` | scenario/workload policy | `50 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2307` | `FoundationCostAmortization` | `cost_per_label` | `Infrastructure.Pricing.Labeling or Scenarios.*` | human-labeling price input | `1 * (USD / count)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2622` | `KwsDataSelectionCalc` | `target_samples` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10_000` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2623` | `KwsDataSelectionCalc` | `cost_per_sample_low` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2624` | `KwsDataSelectionCalc` | `cost_per_sample_high` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2869` | `SelectionBottleneckAnchor` | `dataset_image_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_000_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2870` | `SelectionBottleneckAnchor` | `scoring_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `2.8 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2917` | `SelectionInequalityCalc` | `image_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_000_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2918` | `SelectionInequalityCalc` | `coreset_image_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2919` | `SelectionInequalityCalc` | `training_epoch_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2920` | `SelectionInequalityCalc` | `resnet50_time_per_image` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload size | `0.01 * (second / count)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2921` | `SelectionInequalityCalc` | `resnet18_time_per_image` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload size | `0.002 * (second / count)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:2922` | `SelectionInequalityCalc` | `trap_selection_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `50 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3133` | `RandomAccessPenalty` | `random_read_q` | `Scenarios.*` | unit-bearing scenario input | `4 * KB` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3233` | `DataEchoingRoi` | `pipeline_throughput` | `Scenarios.*` | unit-bearing scenario input | `300 * (count / second)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3234` | `DataEchoingRoi` | `gpu_throughput` | `Hardware.*` | hardware-related quantitative input | `800 * (count / second)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3235` | `DataEchoingRoi` | `echo_epoch_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `90` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3356` | `CostBreakdown` | `raw_data_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50000 * USD` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3358` | `CostBreakdown` | `cost_per_label` | `Infrastructure.Pricing.Labeling or Scenarios.*` | human-labeling price input | `0.05 * (USD / count)` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3359` | `CostBreakdown` | `storage_cost` | `Infrastructure.Pricing.* or Scenarios.*` | scenario/profile input | `200 * USD` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3360` | `CostBreakdown` | `train_compute_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `25000 * USD` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3361` | `CostBreakdown` | `storage_size` | `Systems.Storage or Datasets.*` | storage/data fact | `150 * GB` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3362` | `CostBreakdown` | `storage_duration` | `Scenarios.* or Ops.*` | storage observation or utilization scenario | `12 * ureg.month` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3363` | `CostBreakdown` | `train_epoch_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3364` | `CostBreakdown` | `train_gpu_count` | `Systems.Clusters or Systems.Nodes` | fleet/topology fact | `8` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3365` | `CostBreakdown` | `train_duration` | `Scenarios.* or Ops.*` | scenario/workload policy | `24 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3467` | `BreakevenCalc` | `cost_inference` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3545` | `DeduplicationAmortization` | `cost_build` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50000` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3546` | `DeduplicationAmortization` | `cost_compute_once` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5000` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3701` | `DistributedOverheadCalc` | `t_embed` | `Scenarios.* or Ops.*` | scenario/workload policy | `20 * minute` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3702` | `DistributedOverheadCalc` | `t_dedup` | `Scenarios.* or Ops.*` | scenario/workload policy | `15 * minute` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3703` | `DistributedOverheadCalc` | `t_score` | `Scenarios.* or Ops.*` | scenario/workload policy | `30 * minute` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:3704` | `DistributedOverheadCalc` | `t_select` | `Scenarios.* or Ops.*` | scenario/workload policy | `2 * minute` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4401` | `ChinchillaDiagnostic` | `tokens_per_param_opt` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `20` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4404` | `ChinchillaDiagnostic` | `llama2_70b_training_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0 * c.TRILLION * c.count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4560` | `FpFallacyCalc` | `curated_sample_count` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4561` | `FpFallacyCalc` | `raw_sample_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_000_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4562` | `FpFallacyCalc` | `curated_accuracy_pct` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `92.0` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4563` | `FpFallacyCalc` | `raw_accuracy_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `88.0` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4568` | `FpFallacyCalc` | `training_run_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100_000_000 * USD` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4570` | `FpFallacyCalc` | `cifar10_coreset_pct_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `50` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4571` | `FpFallacyCalc` | `cifar10_acc_retained_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `98` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4572` | `FpFallacyCalc` | `imagenet_acc_retained_pct` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `95` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4654` | `FpPitfallCalc` | `selection_time_bad` | `Scenarios.* or Ops.*` | scenario/workload policy | `10 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4655` | `FpPitfallCalc` | `training_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `2 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4656` | `FpPitfallCalc` | `full_training_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `8 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4657` | `FpPitfallCalc` | `selection_time_good` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.5 * hour` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4658` | `FpPitfallCalc` | `total_sample_count` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_000_000 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4661` | `FpPitfallCalc` | `min_sample_threshold` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `150 * count` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4662` | `FpPitfallCalc` | `al_latency` | `Scenarios.* or Ops.*` | scenario/workload policy | `14 * day` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4663` | `FpPitfallCalc` | `model_drift_epoch_count` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4664` | `FpPitfallCalc` | `batch_size_small_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/data_selection/data_selection.qmd:4665` | `FpPitfallCalc` | `batch_size_large_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |

## vol1/frameworks

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:1888` | `CompilationBenchmark` | `throughput` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `{ "ResNet-50": {"eager": 1450, "compile": 2150, "tensorrt": 3800, "compile_min_s": 15, "compile_max_s": 30}, "BERT-Ba...` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:2607` | `ResNetMemory` | `training_min` | `Scenarios.*` | unit-bearing scenario input | `10 * GB` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:2608` | `ResNetMemory` | `training_max` | `Scenarios.*` | unit-bearing scenario input | `15 * GB` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3179` | `AdminTax` | `params_count` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1 * BILLION * param` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3180` | `AdminTax` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3181` | `AdminTax` | `layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3225` | `ResNetAdminTaxRatio` | `training_min` | `Scenarios.*` | unit-bearing scenario input | `10 * GB` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3226` | `ResNetAdminTaxRatio` | `training_max` | `Scenarios.*` | unit-bearing scenario input | `15 * GB` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3465` | `DeviceBandwidthHierarchy` | `tensor_4mb` | `Scenarios.*` | unit-bearing scenario input | `4 * MB` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:3701` | `DataloaderThroughput` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `64` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:4858` | `TrainingStepDims` | `image_side` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `28` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:4974` | `TrainingStepCalc` | `_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:4976` | `TrainingStepCalc` | `_hidden` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `256` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:5066` | `MnistTrainingStepCalc` | `_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:5068` | `MnistTrainingStepCalc` | `_hidden` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `256` |
| `book/quarto/contents/vol1/frameworks/frameworks.qmd:5193` | `Model7BMemory` | `a100_mem_display` | `Hardware.*` | hardware-related quantitative input | `80 * GB` |

## vol1/hw_acceleration

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:259` | `AmdahlH100` | `p_resnet` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.95` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:260` | `AmdahlH100` | `p_gpt2` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.80` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:2059` | `AcceleratorEfficiencyAnchor` | `energy_dividend` | `Infrastructure.*` | infrastructure input | `200` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:2226` | `TilingPrinciple` | `layer_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:2519` | `AcceleratorEconomics` | `price_h100` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `25000` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:2520` | `AcceleratorEconomics` | `price_tpu` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `8000` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:2521` | `AcceleratorEconomics` | `price_gaudi` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `12000` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3015` | `TensorLifecycleCalc` | `kws_samples_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `16_000` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3842` | `TransformerLayerCalc` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `768` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3843` | `TransformerLayerCalc` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3844` | `TransformerLayerCalc` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3845` | `TransformerLayerCalc` | `num_heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:3946` | `Conv2dAnalysisCalc` | `conv_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:4043` | `DenseLayerAnalysisCalc` | `dense_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:4140` | `LayernormAnalysisCalc` | `ln_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:4141` | `LayernormAnalysisCalc` | `ln_seq` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:4142` | `LayernormAnalysisCalc` | `ln_hidden` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `768` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:5800` | `FpMemoryEnergyCalc` | `layernorm_ai` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.5` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:5801` | `FpMemoryEnergyCalc` | `peak_flops` | `Hardware.* or Scenarios.*` | scenario/profile input | `300 * (TFLOP / second)` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:5802` | `FpMemoryEnergyCalc` | `peak_bw` | `Scenarios.*` | unit-bearing scenario input | `2 * (TB / second)` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:5853` | `FpMultigpuScalingCalc` | `gradient_size` | `Hardware.*` | hardware-related quantitative input | `1.0 * GB` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:5854` | `FpMultigpuScalingCalc` | `step_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `50 * millisecond` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6030` | `FeasibilityAssessment` | `gpu_memory` | `Hardware.* or Scenarios.*` | scenario/profile input | `16 * GB` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6034` | `FeasibilityAssessment` | `mem_bw` | `Scenarios.*` | unit-bearing scenario input | `1 * (TB / second)` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6038` | `FeasibilityAssessment` | `latency_target` | `Scenarios.*` | unit-bearing scenario input | `50 * millisecond` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6130` | `CarbonRoiCalc` | `cpu_power` | `Hardware.*` | hardware-related quantitative input | `100 * watt` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6131` | `CarbonRoiCalc` | `cpu_flops` | `Hardware.* or Scenarios.*` | scenario/profile input | `1 * (TFLOP / second)` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6133` | `CarbonRoiCalc` | `npu_power` | `Hardware.*` | hardware-related quantitative input | `5 * watt` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6134` | `CarbonRoiCalc` | `npu_flops` | `Hardware.* or Scenarios.*` | scenario/profile input | `10 * (TFLOP / second)` |
| `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:6136` | `CarbonRoiCalc` | `inferences_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `1 * BILLION` |

## vol1/introduction

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/introduction/introduction.qmd:618` | `AlexNetBreakthrough` | `alexnet_top5_error` | `Scenarios.* or Ops.*` | scenario/workload policy | `15.3` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:619` | `AlexNetBreakthrough` | `second_place_error` | `Scenarios.* or Ops.*` | scenario/workload policy | `26.2` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:1098` | `GPT3Scale` | `avg_token_bytes` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.4` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:1214` | `AlexNetEvolutionTable` | `alexnet_top5_error` | `Scenarios.* or Ops.*` | scenario/workload policy | `15.3` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:1277` | `AlphaGoZeroCompute` | `days` | `Scenarios.* or Ops.*` | scenario/workload policy | `3` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:1279` | `AlphaGoZeroCompute` | `hours_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:2101` | `TrainingScaleTransition` | `gpt4_class_training_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0e25` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:2262` | `EfficiencyGainsMetrics` | `efficiency_window_years` | `Scenarios.* or Ops.*` | scenario/workload policy | `2019 - 2012` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:2463` | `TrainingComputeGrowthFigure` | `gpt4_class_training_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0e25` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:2642` | `EfficiencyParadoxBridge` | `gpt4_class_training_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0e25` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:3391` | `DriftFallacy` | `drift_points_per_month` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.8` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:3392` | `DriftFallacy` | `months` | `Scenarios.*` | unit-bearing scenario input | `6` |
| `book/quarto/contents/vol1/introduction/introduction.qmd:3450` | `EfficiencyScaleSummary` | `gpt4_class_training_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0e25` |

## vol1/ml_ops

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:215` | `SkewEconomics` | `skew_error_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.01` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:216` | `SkewEconomics` | `cost_per_error` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.10` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1196` | `SilentFailureCost` | `days_manual` | `Scenarios.* or Ops.*` | scenario/workload policy | `28` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1197` | `SilentFailureCost` | `days_auto` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1384` | `RetrainingInterval` | `retrain_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1730` | `ABSampleSize` | `power` | `Infrastructure.*` | infrastructure input | `0.80` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1731` | `ABSampleSize` | `baseline_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.05` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1733` | `ABSampleSize` | `comparison_users_per_variant` | `Scenarios.* or Ops.*` | scenario/workload policy | `25_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1893` | `LatencyBudget` | `request_parse` | `Scenarios.* or Ops.*` | scenario/workload policy | `5` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:1897` | `LatencyBudget` | `model_speedup` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2100` | `KVCacheFootprint` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `8_192` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2162` | `CostPerInference` | `inferences_per_hour` | `Scenarios.* or Ops.*` | scenario/workload policy | `50_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2163` | `CostPerInference` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2220` | `DriftDetectionDelay` | `qps_high` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2222` | `DriftDetectionDelay` | `samples_needed` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2438` | `ObservabilitySampling` | `telemetry_per_request` | `Scenarios.* or Ops.*` | scenario/workload policy | `1 * KB` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2439` | `ObservabilitySampling` | `high_frequency_seconds` | `Scenarios.*` | unit-bearing scenario input | `1 * second` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2440` | `ObservabilitySampling` | `low_frequency_seconds` | `Scenarios.*` | unit-bearing scenario input | `60 * second` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2441` | `ObservabilitySampling` | `success_sample_fraction` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `0.01` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2442` | `ObservabilitySampling` | `error_sample_fraction` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1.0` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2727` | `MonitoringBudget` | `samples_per_min` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `4` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2737` | `MonitoringBudget` | `queries_per_hr` | `Scenarios.* or Ops.*` | scenario/workload policy | `12` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2738` | `MonitoringBudget` | `work_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `8` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2739` | `MonitoringBudget` | `work_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `22` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2847` | `MonitoringROI` | `avg_incident_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2848` | `MonitoringROI` | `annual_monitoring_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:2954` | `SliceAnalysisCalc` | `traffic_pcts` | `Scenarios.* or Ops.*` | scenario/workload policy | `[45, 30, 20, 5]` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3215` | `FraudDetectionImprovement` | `current_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.92` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3216` | `FraudDetectionImprovement` | `target_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.94` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3217` | `FraudDetectionImprovement` | `infra_cost_increase` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.30` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3219` | `FraudDetectionImprovement` | `monthly_false_positive_alerts` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `250_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3494` | `SingleModelROI` | `incident_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `25_000` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3495` | `SingleModelROI` | `hours_saved_monthly` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `20` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3496` | `SingleModelROI` | `hourly_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3596` | `OuraValidationGap` | `accel_only_accuracy` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.57` |
| `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3597` | `OuraValidationGap` | `enhanced_accuracy` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.79` |

## vol1/ml_systems

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:416` | `ThrottlingScenario` | `duration_min` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:571` | `EnergyTransmission` | `data_size` | `Scenarios.*` | unit-bearing scenario input | `1.0 * MB` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1390` | `GPT3TrainingScale` | `gpt3_days` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `15` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1391` | `GPT3TrainingScale` | `gpt3_cost_m` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `4.6` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1392` | `GPT3TrainingScale` | `gpt3_v100_count` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10000` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1533` | `DistancePenalty` | `safety_budget` | `Scenarios.*` | unit-bearing scenario input | `10 * millisecond` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1639` | `CloudEdgeTCO` | `requests_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `1_000_000` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1641` | `CloudEdgeTCO` | `response_size` | `Scenarios.*` | unit-bearing scenario input | `100 * KB` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1645` | `CloudEdgeTCO` | `gpu_instances` | `Systems.*` | system-level fact | `4` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1653` | `CloudEdgeTCO` | `server_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `15000` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1654` | `CloudEdgeTCO` | `server_life_years` | `Systems.*` | system-level fact | `3` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1657` | `CloudEdgeTCO` | `cooling_overhead` | `Infrastructure.* or Scenarios.Sustainability` | infrastructure/sustainability fact | `0.30` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1816` | `VoiceAssistantWall` | `ww_cloud_cost_per_device` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.50` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1817` | `VoiceAssistantWall` | `ww_edge_power_min_mw` | `Infrastructure.*` | infrastructure input | `0.1` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1818` | `VoiceAssistantWall` | `ww_edge_power_max_mw` | `Infrastructure.*` | infrastructure input | `1` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1819` | `VoiceAssistantWall` | `ww_edge_cost_per_year` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.01` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1823` | `VoiceAssistantWall` | `vi_queries_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `20` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1824` | `VoiceAssistantWall` | `vi_gpu_ms_per_query` | `Scenarios.* or Ops.*` | scenario/workload policy | `200` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1825` | `VoiceAssistantWall` | `vi_gpus_per_datacenter` | `Systems.*` | system-level fact | `10_000` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1826` | `VoiceAssistantWall` | `vi_audio_sample_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `16_000` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1828` | `VoiceAssistantWall` | `vi_waking_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `16` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2138` | `EdgeEnergyTransmissionRecap` | `data_size` | `Scenarios.*` | unit-bearing scenario input | `1.0 * MB` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2380` | `EdgeSizingThroughput` | `headroom` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2455` | `EdgeSizingFleetTCO` | `headroom` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2456` | `EdgeSizingFleetTCO` | `derate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.5` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2457` | `EdgeSizingFleetTCO` | `years` | `Scenarios.*` | unit-bearing scenario input | `3` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2462` | `EdgeSizingFleetTCO` | `coral_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2463` | `EdgeSizingFleetTCO` | `jetson_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `600` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2464` | `EdgeSizingFleetTCO` | `nuc_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `400` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2810` | `ThermalQuantCalc` | `baseline_power` | `Infrastructure.*` | infrastructure input | `12 * watt` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2813` | `ThermalQuantCalc` | `temp_rise` | `Scenarios.*` | unit-bearing scenario input | `1 * ureg.delta_degC / second` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2814` | `ThermalQuantCalc` | `trip_time` | `Scenarios.*` | unit-bearing scenario input | `60 * second` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2883` | `MobileBatteryCapacity` | `phone_battery` | `Scenarios.*` | unit-bearing scenario input | `h_phone.battery_capacity if h_phone.battery_capacity else 15 * Wh` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2884` | `MobileBatteryCapacity` | `phone_battery_high` | `Scenarios.*` | unit-bearing scenario input | `22 * Wh` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2885` | `MobileBatteryCapacity` | `baseline_runtime` | `Scenarios.* or Ops.*` | scenario/workload policy | `24 * hour` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:2886` | `MobileBatteryCapacity` | `added_ml_power` | `Infrastructure.*` | infrastructure input | `1 * watt` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:3299` | `WildlifeMonitoring` | `raw_audio_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `4.3 * GB` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:3300` | `WildlifeMonitoring` | `summary_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `400 * KB` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4383` | `MobilePowerFallacyCalc` | `battery` | `Scenarios.*` | unit-bearing scenario input | `h_phone.battery_capacity if h_phone.battery_capacity else 15 * Wh` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4384` | `MobilePowerFallacyCalc` | `low_power` | `Infrastructure.*` | infrastructure input | `1 * watt` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4385` | `MobilePowerFallacyCalc` | `high_power` | `Infrastructure.*` | infrastructure input | `5 * watt` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4456` | `TcoPitfallCalc` | `cloud_compute` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2000 * USD / ureg.month` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4459` | `TcoPitfallCalc` | `edge_hardware` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500 * USD / ureg.month` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4460` | `TcoPitfallCalc` | `edge_network` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `3000 * USD / ureg.month` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4461` | `TcoPitfallCalc` | `edge_maintenance` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500 * USD / ureg.month` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4462` | `TcoPitfallCalc` | `edge_reliability` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2000 * USD / ureg.month` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4514` | `AmdahlCameraCalc` | `cam_isp` | `Scenarios.*` | unit-bearing scenario input | `100 * millisecond` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4515` | `AmdahlCameraCalc` | `cam_ml` | `Scenarios.*` | unit-bearing scenario input | `60 * millisecond` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4516` | `AmdahlCameraCalc` | `cam_post` | `Scenarios.*` | unit-bearing scenario input | `40 * millisecond` |
| `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:4520` | `AmdahlCameraCalc` | `general_model_speedup` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100` |

## vol1/ml_workflow

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:353` | `IterationTax` | `weeks_total` | `Scenarios.* or Ops.*` | scenario/workload policy | `26` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:354` | `IterationTax` | `hours_per_week` | `Scenarios.* or Ops.*` | scenario/workload policy | `168` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:359` | `IterationTax` | `large_cycle_time_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `168` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:364` | `IterationTax` | `small_cycle_time_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:926` | `ConstraintPropagation` | `ML_WORKFLOW_STAGE_MONITORING` | `Scenarios.* or Ops.*` | scenario/workload policy | `6` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:927` | `ConstraintPropagation` | `ML_WORKFLOW_CONSTRAINT_COST_BASE` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:933` | `ConstraintPropagation` | `weeks_per_cycle` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:936` | `ConstraintPropagation` | `stage5_latency_q` | `Scenarios.*` | unit-bearing scenario input | `100 * millisecond` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:937` | `ConstraintPropagation` | `stage5_memory_q` | `Hardware.* or Scenarios.*` | scenario/profile input | `500 * MB` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:938` | `ConstraintPropagation` | `jetson_latency_q` | `Scenarios.*` | unit-bearing scenario input | `50 * millisecond` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:939` | `ConstraintPropagation` | `jetson_model_q` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `200 * MB` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:940` | `ConstraintPropagation` | `resnet_variant_q` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `200 * MB` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1105` | `BandwidthCompute` | `patients_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `150` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1107` | `BandwidthCompute` | `mb_per_photo_q` | `Scenarios.*` | unit-bearing scenario input | `5.0 * MB` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1108` | `BandwidthCompute` | `clinic_hours_q` | `Scenarios.* or Ops.*` | scenario/workload policy | `8.0 * hour` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1109` | `BandwidthCompute` | `uplink_q` | `Scenarios.*` | unit-bearing scenario input | `2.0 * (megabit / second)` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1110` | `BandwidthCompute` | `summary_per_patient_q` | `Scenarios.*` | unit-bearing scenario input | `10.0 * KB` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1592` | `HyperparameterGrid` | `values_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1905` | `DeploymentEconomics` | `patients_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `50` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1906` | `DeploymentEconomics` | `billable_images_per_patient` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1910` | `DeploymentEconomics` | `cloud_inf_cost_q` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.01 * USD` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1911` | `DeploymentEconomics` | `cloud_network_cost_q` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `45000 * (USD / year)` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1912` | `DeploymentEconomics` | `cloud_latency_q` | `Scenarios.*` | unit-bearing scenario input | `200 * millisecond` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1916` | `DeploymentEconomics` | `edge_unit_cost_q` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500 * USD` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1917` | `DeploymentEconomics` | `edge_maint_cost_q` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `25000 * (USD / year)` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1918` | `DeploymentEconomics` | `edge_inf_cost_q` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.001 * USD` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:1919` | `DeploymentEconomics` | `edge_latency_q` | `Scenarios.*` | unit-bearing scenario input | `50 * millisecond` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:2061` | `MonitoringThresholds` | `latency_p95_target_q` | `Scenarios.*` | unit-bearing scenario input | `50 * millisecond` |
| `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:2211` | `FallaciesConstraintPropagation` | `ML_WORKFLOW_CONSTRAINT_COST_BASE` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2` |

## vol1/model_compression

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:396` | `QuantizationSpeedup` | `kv_cache_gb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.0` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:561` | `ModelDeviceComparison` | `resnet_mem` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100 * MB` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:564` | `ModelDeviceComparison` | `dscnn_mem` | `Scenarios.*` | unit-bearing scenario input | `500 * KB` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:2368` | `LowRankFactorization` | `bytes_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:3392` | `QuantizationSavings` | `gpu_mem_fp16_gb` | `Systems.*` | system-level fact | `24` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:3393` | `QuantizationSavings` | `gpu_mem_int4_gb` | `Systems.*` | system-level fact | `8` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:7295` | `AmdahlCompression` | `model_fraction` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.20` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:7376` | `ResNet50Int8Metrics` | `calibration_error_fp32` | `Scenarios.* or Ops.*` | scenario/workload policy | `2.1` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:7377` | `ResNet50Int8Metrics` | `calibration_error_int8` | `Scenarios.* or Ops.*` | scenario/workload policy | `3.4` |
| `book/quarto/contents/vol1/model_compression/model_compression.qmd:8150` | `FallaciesAnalysis` | `overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `15` |

## vol1/model_serving

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:90` | `BlackFridayCalc` | `bf_latency_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:91` | `BlackFridayCalc` | `bf_qps_normal_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:92` | `BlackFridayCalc` | `bf_qps_spike_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `10000` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:94` | `BlackFridayCalc` | `bf_collapse_latency_s_value` | `Scenarios.*` | unit-bearing scenario input | `10 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:760` | `StaticBatchCalc` | `inference_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:761` | `StaticBatchCalc` | `dynamic_latency_budget_ms_value` | `Scenarios.*` | unit-bearing scenario input | `100 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:802` | `CostLatencyCalc` | `latency_a_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:803` | `CostLatencyCalc` | `throughput_a_rps_value` | `Scenarios.*` | unit-bearing scenario input | `200 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:804` | `CostLatencyCalc` | `latency_b_ms_value` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:805` | `CostLatencyCalc` | `throughput_b_rps_value` | `Scenarios.*` | unit-bearing scenario input | `800 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:947` | `ResNetServingSpectrum` | `cloud_vram` | `Hardware.* or Scenarios.*` | scenario/profile input | `2 * GB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:953` | `ResNetServingSpectrum` | `mobile_energy_npu` | `Hardware.*` | hardware-related quantitative input | `0.8 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:954` | `ResNetServingSpectrum` | `mobile_energy_cpu` | `Hardware.*` | hardware-related quantitative input | `4.2 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:958` | `ResNetServingSpectrum` | `tiny_energy` | `Infrastructure.*` | infrastructure input | `12.0 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1487` | `TailLatencyRatioCalc` | `mean_latency_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1488` | `TailLatencyRatioCalc` | `p99_latency_ms_value` | `Scenarios.*` | unit-bearing scenario input | `2000 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1569` | `LatencyTableCalc` | `l_jpeg_value` | `Scenarios.*` | unit-bearing scenario input | `3.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1570` | `LatencyTableCalc` | `l_resize_value` | `Scenarios.*` | unit-bearing scenario input | `1.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1571` | `LatencyTableCalc` | `l_norm_value` | `Scenarios.*` | unit-bearing scenario input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1572` | `LatencyTableCalc` | `l_transfer_value` | `Scenarios.*` | unit-bearing scenario input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1573` | `LatencyTableCalc` | `l_inf_value` | `Scenarios.*` | unit-bearing scenario input | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1574` | `LatencyTableCalc` | `l_post_value` | `Scenarios.*` | unit-bearing scenario input | `0.1 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1625` | `LatencyBudgetCalc` | `jpeg_decode_ms_value` | `Scenarios.*` | unit-bearing scenario input | `3.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1626` | `LatencyBudgetCalc` | `resize_ms_value` | `Scenarios.*` | unit-bearing scenario input | `1.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1627` | `LatencyBudgetCalc` | `normalize_ms_value` | `Scenarios.*` | unit-bearing scenario input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1628` | `LatencyBudgetCalc` | `cpu_gpu_ms_value` | `Hardware.*` | hardware-related quantitative input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1629` | `LatencyBudgetCalc` | `resnet_inference_ms_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1630` | `LatencyBudgetCalc` | `postprocess_ms_value` | `Scenarios.*` | unit-bearing scenario input | `0.1 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1631` | `LatencyBudgetCalc` | `tensorrt_inference_ms_value` | `Scenarios.*` | unit-bearing scenario input | `2.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1696` | `DlrmLatencyCalc` | `dlrm_input_ms_value` | `Hardware.*` | hardware-related quantitative input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1697` | `DlrmLatencyCalc` | `dlrm_embed_ms_value` | `Scenarios.*` | unit-bearing scenario input | `6.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1698` | `DlrmLatencyCalc` | `dlrm_mlp_ms_value` | `Scenarios.*` | unit-bearing scenario input | `1.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1699` | `DlrmLatencyCalc` | `dlrm_post_ms_value` | `Scenarios.*` | unit-bearing scenario input | `1.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1758` | `AmdahlServingCalc` | `preprocess_ms_value` | `Scenarios.*` | unit-bearing scenario input | `(3.0 + 1.0 + 0.5) * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1759` | `AmdahlServingCalc` | `cpu_gpu_ms_value` | `Hardware.*` | hardware-related quantitative input | `0.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1760` | `AmdahlServingCalc` | `resnet_inference_ms_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1761` | `AmdahlServingCalc` | `postprocess_ms_value` | `Scenarios.*` | unit-bearing scenario input | `0.1 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1887` | `ResolutionScalingCalc` | `r2_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload size | `2 * IMAGE_DIM_RESNET` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1923` | `ResolutionBottleneckCalc` | `act_224` | `Scenarios.*` | unit-bearing scenario input | `12.5 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1924` | `ResolutionBottleneckCalc` | `act_384` | `Scenarios.*` | unit-bearing scenario input | `36.8 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1925` | `ResolutionBottleneckCalc` | `act_512` | `Scenarios.*` | unit-bearing scenario input | `65.5 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1926` | `ResolutionBottleneckCalc` | `act_640` | `Scenarios.*` | unit-bearing scenario input | `102.4 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1928` | `ResolutionBottleneckCalc` | `ai_224_value` | `Hardware.*` | hardware-related quantitative input | `85 * (flop / byte)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1929` | `ResolutionBottleneckCalc` | `ai_384_value` | `Hardware.*` | hardware-related quantitative input | `49 * (flop / byte)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1930` | `ResolutionBottleneckCalc` | `ai_512_value` | `Hardware.*` | hardware-related quantitative input | `28 * (flop / byte)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1931` | `ResolutionBottleneckCalc` | `ai_640_value` | `Hardware.*` | hardware-related quantitative input | `18 * (flop / byte)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1933` | `ResolutionBottleneckCalc` | `ridge_point_value` | `Hardware.*` | hardware-related quantitative input | `16 * (flop / byte)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:1986` | `AdaptiveResolutionCalc` | `adaptive_accuracy_retention_pct_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `99.2` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2198` | `CapacityPlanning` | `lambda_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000.0` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2253` | `BatchingTax` | `lambda_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `500.0` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2376` | `CapacityPlanningCalc` | `cp_peak_qps_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `5000 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2377` | `CapacityPlanningCalc` | `cp_service_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2378` | `CapacityPlanningCalc` | `cp_p99_target_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2379` | `CapacityPlanningCalc` | `cp_v100_throughput_value` | `Hardware.*` | hardware-related quantitative input | `1143 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2380` | `CapacityPlanningCalc` | `cp_headroom_factor_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.3` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2384` | `CapacityPlanningCalc` | `mm1_rho_example_util_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `0.7` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2597` | `ColdStartCalc` | `cs_ssd_value` | `Systems.Storage or Scenarios.*` | storage-related scenario input | `0.5 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2598` | `ColdStartCalc` | `cs_s3_value` | `Systems.Storage or Datasets.*` | storage/data fact | `4.0 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2599` | `ColdStartCalc` | `cs_cuda_value` | `Scenarios.*` | unit-bearing scenario input | `0.4 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2600` | `ColdStartCalc` | `cs_compile_value` | `Scenarios.*` | unit-bearing scenario input | `30.0 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2601` | `ColdStartCalc` | `cs_warmup_value` | `Scenarios.*` | unit-bearing scenario input | `0.2 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2602` | `ColdStartCalc` | `cs_runtime_overhead_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.4 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2706` | `ModelSwapCalc` | `model_size` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10 * GB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2778` | `BatchThroughputCalc` | `batch1_throughput_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2779` | `BatchThroughputCalc` | `batch32_throughput_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `1280 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2780` | `BatchThroughputCalc` | `batch32_inference_ms_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `25.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2781` | `BatchThroughputCalc` | `batch_window_ms_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `10.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2782` | `BatchThroughputCalc` | `batch1_inference_total_ms_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2956` | `BatchingBudgetCalc` | `batch_window_ms_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `20 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2957` | `BatchingBudgetCalc` | `slo_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:2958` | `BatchingBudgetCalc` | `inference_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3031` | `BatchingAnalysisCalc` | `T_window_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `10.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3032` | `BatchingAnalysisCalc` | `fixed_overhead_ms_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3033` | `BatchingAnalysisCalc` | `per_image_ms_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `0.6 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3034` | `BatchingAnalysisCalc` | `batch_sizes_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `[1, 4, 8, 16, 32]` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3035` | `BatchingAnalysisCalc` | `il_overhead_ms_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3036` | `BatchingAnalysisCalc` | `il_compute_b1_ms_value` | `Scenarios.*` | unit-bearing scenario input | `0.6 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3037` | `BatchingAnalysisCalc` | `il_compute_b32_ms_value` | `Scenarios.*` | unit-bearing scenario input | `19.2 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3038` | `BatchingAnalysisCalc` | `il_threshold_pct_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `10` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3150` | `BatchingOptimization` | `s1_window` | `Scenarios.* or Ops.*` | scenario/workload policy | `5.0` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3151` | `BatchingOptimization` | `s1_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3155` | `BatchingOptimization` | `s2_window` | `Scenarios.* or Ops.*` | scenario/workload policy | `25.0` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3156` | `BatchingOptimization` | `s2_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `48` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3242` | `SloViolationCalc` | `qps_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `500 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3243` | `SloViolationCalc` | `T_slo_value` | `Scenarios.*` | unit-bearing scenario input | `10.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3244` | `SloViolationCalc` | `p99_batch_count_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `11` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3382` | `PracticalConfigCalc` | `pc_slo_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3383` | `PracticalConfigCalc` | `pc_qps_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `500 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3385` | `PracticalConfigCalc` | `pc_mem_limit_batch_count_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3386` | `PracticalConfigCalc` | `pc_config_window_ms_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `12 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3387` | `PracticalConfigCalc` | `pc_config_batch_count_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3388` | `PracticalConfigCalc` | `pc_base_service_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5.0 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3389` | `PracticalConfigCalc` | `pc_per_image_ms_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `0.6 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3547` | `TrafficAdaptiveBatchingCalc` | `arrival_rates` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `(100, 500, 1_000, 5_000)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3681` | `MobileServingCalc` | `m_cam_ms` | `Scenarios.*` | unit-bearing scenario input | `8 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3682` | `MobileServingCalc` | `m_jpeg_ms` | `Scenarios.*` | unit-bearing scenario input | `15 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3683` | `MobileServingCalc` | `m_resize_ms` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3684` | `MobileServingCalc` | `m_npu_ms` | `Hardware.*` | hardware-related quantitative input | `12 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3685` | `MobileServingCalc` | `m_ui_ms` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3686` | `MobileServingCalc` | `peak_memory` | `Hardware.* or Scenarios.*` | scenario/profile input | `150 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3687` | `MobileServingCalc` | `m_cam_mj` | `Scenarios.*` | unit-bearing scenario input | `0.08 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3688` | `MobileServingCalc` | `m_jpeg_mj` | `Scenarios.*` | unit-bearing scenario input | `1.5 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3689` | `MobileServingCalc` | `m_resize_mj` | `Scenarios.*` | unit-bearing scenario input | `0.4 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3690` | `MobileServingCalc` | `m_npu_mj` | `Hardware.*` | hardware-related quantitative input | `0.8 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3691` | `MobileServingCalc` | `m_ui_mj` | `Scenarios.*` | unit-bearing scenario input | `0.2 * mJ` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3927` | `CarbonCostCalc` | `cc_concurrent_requests_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `114` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3928` | `CarbonCostCalc` | `cc_tokens_per_sec_req_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `8 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3929` | `CarbonCostCalc` | `cc_host_overhead` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `300 * watt` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3930` | `CarbonCostCalc` | `cc_response_tokens_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `500` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:3932` | `CarbonCostCalc` | `cc_idle_power` | `Infrastructure.*` | infrastructure input | `300 * watt` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4087` | `RuntimeComparisonCalc` | `rt_pytorch_ms_value` | `Scenarios.*` | unit-bearing scenario input | `8.5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4088` | `RuntimeComparisonCalc` | `rt_torchscript_ms_value` | `Scenarios.*` | unit-bearing scenario input | `6.2 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4089` | `RuntimeComparisonCalc` | `rt_onnx_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5.1 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4090` | `RuntimeComparisonCalc` | `rt_trt_fp32_ms_value` | `Hardware.*` | hardware-related quantitative input | `2.8 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4091` | `RuntimeComparisonCalc` | `rt_trt_fp16_ms_value` | `Hardware.*` | hardware-related quantitative input | `1.4 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4092` | `RuntimeComparisonCalc` | `rt_trt_int8_ms_value` | `Hardware.*` | hardware-related quantitative input | `0.9 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4191` | `PrecisionTradeoffCalc` | `pt_fp32_ms_value` | `Hardware.*` | hardware-related quantitative input | `2.8 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4192` | `PrecisionTradeoffCalc` | `pt_fp32_mem` | `Hardware.*` | hardware-related quantitative input | `98 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4194` | `PrecisionTradeoffCalc` | `pt_fp16_ms_value` | `Hardware.*` | hardware-related quantitative input | `1.4 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4195` | `PrecisionTradeoffCalc` | `pt_fp16_mem` | `Hardware.*` | hardware-related quantitative input | `49 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4198` | `PrecisionTradeoffCalc` | `pt_int8_ms_value` | `Hardware.*` | hardware-related quantitative input | `0.9 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4199` | `PrecisionTradeoffCalc` | `pt_int8_mem` | `Hardware.*` | hardware-related quantitative input | `25 * MB` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4427` | `CostAnalysisCalc` | `ca_cpu_cost_value` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.17 * (USD / hour)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4428` | `CostAnalysisCalc` | `ca_cpu_throughput_value` | `Hardware.*` | hardware-related quantitative input | `50 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4429` | `CostAnalysisCalc` | `ca_t4_cost_value` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.53 * (USD / hour)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4430` | `CostAnalysisCalc` | `ca_t4_throughput_value` | `Hardware.*` | hardware-related quantitative input | `400 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4431` | `CostAnalysisCalc` | `ca_v100_cost_value` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `3.06 * (USD / hour)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4432` | `CostAnalysisCalc` | `ca_v100_throughput_value` | `Hardware.*` | hardware-related quantitative input | `1200 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4509` | `CapacityPlanningRecap` | `cp_peak_qps_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `5000 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4510` | `CapacityPlanningRecap` | `cp_service_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4511` | `CapacityPlanningRecap` | `cp_p99_target_ms_value` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4512` | `CapacityPlanningRecap` | `cp_v100_throughput_value` | `Hardware.*` | hardware-related quantitative input | `1143 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4513` | `CapacityPlanningRecap` | `cp_headroom_factor_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.3` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4653` | `LlmCaseStudyHwSpecs` | `prompt_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1000` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4654` | `LlmCaseStudyHwSpecs` | `decode_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `256` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4782` | `LlmServingCalc` | `decode_tokens_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `256` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4783` | `LlmServingCalc` | `prompt_tokens_value` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1000` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4920` | `FallacyLatencyCalc` | `fl_service_slow_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4921` | `FallacyLatencyCalc` | `fl_service_fast_ms_value` | `Scenarios.*` | unit-bearing scenario input | `2 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:4982` | `FallacyUtilizationCalc` | `fu_service_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5077` | `TailLatencyCalc` | `tl_service_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5123` | `FallacyBatchingCalc` | `fb_batch_small_count_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `16` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5124` | `FallacyBatchingCalc` | `fb_batch_large_count_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5125` | `FallacyBatchingCalc` | `fb_throughput_small_value` | `Scenarios.*` | unit-bearing scenario input | `1143 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5126` | `FallacyBatchingCalc` | `fb_throughput_large_value` | `Scenarios.*` | unit-bearing scenario input | `1280 * (count / second)` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5127` | `FallacyBatchingCalc` | `fb_inf_small_ms_value` | `Scenarios.*` | unit-bearing scenario input | `14 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5128` | `FallacyBatchingCalc` | `fb_inf_large_ms_value` | `Scenarios.*` | unit-bearing scenario input | `25 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5173` | `FallacyCalibrationCalc` | `fc_imagenet_acc_pct_value` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `76.1` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5213` | `FallacyColdstartCalc` | `cs_compile_time_s_value` | `Scenarios.*` | unit-bearing scenario input | `30 * second` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5214` | `FallacyColdstartCalc` | `cs_steady_latency_ms_value` | `Scenarios.*` | unit-bearing scenario input | `5 * ms` |
| `book/quarto/contents/vol1/model_serving/model_serving.qmd:5215` | `FallacyColdstartCalc` | `cs_cold_latency_ms_value` | `Scenarios.*` | unit-bearing scenario input | `500 * ms` |

## vol1/nn_architectures

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:687` | `MLPDefinitionCosts` | `layer_width` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:1113` | `MLPLayerStats` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2310` | `RNNCompute` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2433` | `AttentionDefinitionCosts` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2691` | `QKVProjectionCosts` | `tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `6` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2692` | `QKVProjectionCosts` | `d_model` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `768` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2693` | `QKVProjectionCosts` | `qkv_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2304` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2891` | `AttentionComputeCosts` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2892` | `AttentionComputeCosts` | `head_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `64` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2994` | `AttentionMemory` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100_000` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2996` | `AttentionMemory` | `num_layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `32` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2997` | `AttentionMemory` | `num_heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3088` | `TransformerComplexityAnchor` | `base_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3089` | `TransformerComplexityAnchor` | `doubled_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3090` | `TransformerComplexityAnchor` | `mid_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3091` | `TransformerComplexityAnchor` | `long_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3468` | `KVCacheSizing` | `layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `32` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3469` | `KVCacheSizing` | `kv_tensors` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3470` | `KVCacheSizing` | `heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `32` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3471` | `KVCacheSizing` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3472` | `KVCacheSizing` | `head_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:3689` | `DLRMEmbedding` | `num_users` | `Scenarios.* or Ops.*` | scenario/workload policy | `1_000_000_000` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:4049` | `ResNetSkipOverhead` | `memory_overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `20` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:4050` | `ResNetSkipOverhead` | `epoch_cost_pct` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `10` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5006` | `ThroughputCeilingCalc` | `midrange_gpu_throughput` | `Hardware.* or Scenarios.*` | scenario/profile input | `10 * (TFLOP / second)` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5080` | `WildlifeModelSizing` | `mnv1_params_m` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4.2` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5082` | `WildlifeModelSizing` | `mnv2_params_m` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.2` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5085` | `WildlifeModelSizing` | `inference_power_mw` | `Infrastructure.*` | infrastructure input | `200` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5086` | `WildlifeModelSizing` | `inferences_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5095` | `WildlifeModelSizing` | `accuracy_target_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `90` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5097` | `WildlifeModelSizing` | `model_budget_mb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5319` | `TransformerComplexityPitfall` | `base_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5320` | `TransformerComplexityPitfall` | `mid_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5430` | `KVCacheServingPitfall` | `layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `32` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5431` | `KVCacheServingPitfall` | `kv_tensors` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5432` | `KVCacheServingPitfall` | `heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `32` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5433` | `KVCacheServingPitfall` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:5434` | `KVCacheServingPitfall` | `head_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |

## vol1/nn_computation

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:877` | `ParadigmInfrastructureRecap` | `mnist_dims` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `[784, 128, 64, 10]` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1539` | `` | `start_of_year` | `Scenarios.*` | unit-bearing scenario input | `datetime(date.year, 1, 1)` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1540` | `` | `days_in_year` | `Scenarios.* or Ops.*` | scenario/workload policy | `366 if date.is_leap_year else 365` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1670` | `` | `model_row` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `model_row.sort_values('Compute', ascending=False).iloc[0]` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1672` | `` | `offset` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `label_offsets.get(model_name, (15, 15))` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1798` | `TrainingEnergyScale` | `lenet_train_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `3` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1799` | `TrainingEnergyScale` | `lenet_workstation_power` | `Infrastructure.*` | infrastructure input | `0.75 * kW` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:1802` | `TrainingEnergyScale` | `household_energy_per_year` | `Infrastructure.*` | infrastructure input | `10.5 * MWh` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:2239` | `MnistArchitectureConstants` | `mnist_l1_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:2240` | `MnistArchitectureConstants` | `mnist_l2_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `128` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:2241` | `MnistArchitectureConstants` | `mnist_l3_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `64` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:2242` | `MnistArchitectureConstants` | `mnist_l4_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:2917` | `MnistArchCheckpointRecap` | `mnist_dims` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `[784, 128, 64, 10]` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3030` | `MnistScaleComparison` | `mnist_input_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3031` | `MnistScaleComparison` | `mnist_output_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3032` | `MnistScaleComparison` | `mnist_large_l1` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1000` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3033` | `MnistScaleComparison` | `mnist_large_l2` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1000` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3034` | `MnistScaleComparison` | `mnist_small_l1` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3035` | `MnistScaleComparison` | `mnist_small_l2` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3288` | `MnistModelSizeLocal` | `mnist_l1_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3289` | `MnistModelSizeLocal` | `mnist_l2_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `128` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3290` | `MnistModelSizeLocal` | `mnist_l3_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `64` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3291` | `MnistModelSizeLocal` | `mnist_l4_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3368` | `MnistTrainingMemoryCalc` | `bytes_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3483` | `MnistArchitectureRecap` | `mnist_l1_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3484` | `MnistArchitectureRecap` | `mnist_l2_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `128` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3485` | `MnistArchitectureRecap` | `mnist_l3_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `64` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3486` | `MnistArchitectureRecap` | `mnist_l4_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `10` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3538` | `MemoryExplosionCalc` | `mnist_l1_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3539` | `MemoryExplosionCalc` | `mnist_l2_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `128` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3540` | `MemoryExplosionCalc` | `mnist_l3_dim` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `64` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3622` | `MentalMathCalc` | `mm_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100 * Mparam` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3624` | `MentalMathCalc` | `mm_overhead` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3625` | `MentalMathCalc` | `mm_gpu` | `Hardware.*` | hardware-related quantitative input | `16 * GB` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3701` | `MnistTrainingIntroRecap` | `total_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `109_386` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3875` | `MnistForwardIntroLocal` | `image_height` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `28` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3876` | `MnistForwardIntroLocal` | `image_width` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `28` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:3877` | `MnistForwardIntroLocal` | `first_hidden_units` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4025` | `MnistFlopsCalc` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4425` | `BackpropMemory` | `layers_dims` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `[784, 512, 256, 10]` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4426` | `BackpropMemory` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4427` | `BackpropMemory` | `bytes_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4496` | `AdamMemoryOverhead` | `bytes_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:4622` | `MnistEpochLocal` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5034` | `MnistInferenceResourceLocal` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5755` | `FallacyQuantExamples` | `model_latency` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `20 * ms` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5756` | `FallacyQuantExamples` | `preprocessing_latency` | `Scenarios.*` | unit-bearing scenario input | `25 * ms` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5757` | `FallacyQuantExamples` | `postprocessing_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5758` | `FallacyQuantExamples` | `latency_budget` | `Scenarios.*` | unit-bearing scenario input | `50 * ms` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5761` | `FallacyQuantExamples` | `mnist_dims` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `[784, 128, 64, 10]` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5762` | `FallacyQuantExamples` | `mnist_batch_size` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `32` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5892` | `MnistWeightsCalc` | `mnist_pixels` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `784` |
| `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5934` | `SummaryParadigmCostRecap` | `mnist_dims` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `[784, 128, 64, 10]` |

## vol1/responsible_engr

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:982` | `FairnessPrice` | `hire_value` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100_000 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:983` | `FairnessPrice` | `bad_hire_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50_000 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:986` | `FairnessPrice` | `positive_base_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.50` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1443` | `GDPRReviewLoad` | `review_rate_pct` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.1` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1685` | `InferenceCostCalc` | `data_prep_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `100 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1686` | `InferenceCostCalc` | `hyperparam_hours` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `500 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1687` | `InferenceCostCalc` | `train_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1690` | `InferenceCostCalc` | `users_daily_value` | `Scenarios.* or Ops.*` | scenario/workload policy | `10_000_000` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1691` | `InferenceCostCalc` | `recs_per_user_count` | `Scenarios.* or Ops.*` | scenario/workload policy | `20` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1692` | `InferenceCostCalc` | `inference_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1876` | `TCOCalc` | `t_data_prep_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `100 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1877` | `TCOCalc` | `t_hparam_exps` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1878` | `TCOCalc` | `t_hparam_cost_exp` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `40.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1879` | `TCOCalc` | `t_final_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1883` | `TCOCalc` | `i_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1884` | `TCOCalc` | `o_monitor_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50000.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1885` | `TCOCalc` | `o_oncall_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100000.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1886` | `TCOCalc` | `o_incident_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `20000.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2145` | `CarbonScaleCalc` | `car_annual_emissions` | `Infrastructure.*` | infrastructure input | `4.6 * metric_ton` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2829` | `ResponsibleTcoRecap` | `t_data_prep_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `100 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2830` | `ResponsibleTcoRecap` | `t_hparam_exps` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2831` | `ResponsibleTcoRecap` | `t_hparam_cost_exp` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `40.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2832` | `ResponsibleTcoRecap` | `t_final_time` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * hour` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2836` | `ResponsibleTcoRecap` | `i_latency` | `Scenarios.*` | unit-bearing scenario input | `10 * ms` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2837` | `ResponsibleTcoRecap` | `o_monitor_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50000.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2838` | `ResponsibleTcoRecap` | `o_oncall_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100000.0 * USD` |
| `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:2839` | `ResponsibleTcoRecap` | `o_incident_yr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `20000.0 * USD` |

## vol1/training

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol1/training/training.qmd:76` | `TrainingScenarios` | `gpt2_cost_2019` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50_000` |
| `book/quarto/contents/vol1/training/training.qmd:77` | `TrainingScenarios` | `gpt4_cost_est` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100 * MILLION` |
| `book/quarto/contents/vol1/training/training.qmd:78` | `TrainingScenarios` | `gpt2_fwd_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `3e9` |
| `book/quarto/contents/vol1/training/training.qmd:79` | `TrainingScenarios` | `gpt2_total_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1e19` |
| `book/quarto/contents/vol1/training/training.qmd:82` | `TrainingScenarios` | `ckpt_layers` | `Systems.Storage or Datasets.*` | storage/data fact | `48` |
| `book/quarto/contents/vol1/training/training.qmd:299` | `GPT2LighthouseSpecs` | `gpt2_total_flops` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1e19` |
| `book/quarto/contents/vol1/training/training.qmd:450` | `GPT2Compute` | `heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `25` |
| `book/quarto/contents/vol1/training/training.qmd:454` | `GPT2Compute` | `batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/training/training.qmd:455` | `GPT2Compute` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:456` | `GPT2Compute` | `steps` | `Scenarios.* or Ops.*` | scenario/workload policy | `50_000` |
| `book/quarto/contents/vol1/training/training.qmd:920` | `ResNetBatchMemory` | `act_mem_b32` | `Scenarios.*` | unit-bearing scenario input | `8 * GB` |
| `book/quarto/contents/vol1/training/training.qmd:921` | `ResNetBatchMemory` | `grad_mem_b32` | `Scenarios.*` | unit-bearing scenario input | `4 * GB` |
| `book/quarto/contents/vol1/training/training.qmd:1015` | `AdamMemory` | `params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `100 * Mparam` |
| `book/quarto/contents/vol1/training/training.qmd:1389` | `GPT2ActivationMemory` | `heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `25` |
| `book/quarto/contents/vol1/training/training.qmd:1393` | `GPT2ActivationMemory` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `8` |
| `book/quarto/contents/vol1/training/training.qmd:1394` | `GPT2ActivationMemory` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:1398` | `GPT2ActivationMemory` | `ffn_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `hidden_dim * 4` |
| `book/quarto/contents/vol1/training/training.qmd:1736` | `AttentionIntensity` | `heads_small` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12` |
| `book/quarto/contents/vol1/training/training.qmd:2481` | `GPT2DataPipeline` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/training/training.qmd:2482` | `GPT2DataPipeline` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:2483` | `GPT2DataPipeline` | `token_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500_000` |
| `book/quarto/contents/vol1/training/training.qmd:2486` | `GPT2DataPipeline` | `bpe_vocab` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50_257` |
| `book/quarto/contents/vol1/training/training.qmd:2488` | `GPT2DataPipeline` | `gpu_forward_ms` | `Scenarios.* or Ops.*` | scenario/workload policy | `80` |
| `book/quarto/contents/vol1/training/training.qmd:2490` | `GPT2DataPipeline` | `throughput_samples` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `380` |
| `book/quarto/contents/vol1/training/training.qmd:2778` | `TrainingDimensions` | `layer_dim_h` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `512` |
| `book/quarto/contents/vol1/training/training.qmd:2779` | `TrainingDimensions` | `layer_dim_w` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:2780` | `TrainingDimensions` | `layer_batch` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `64` |
| `book/quarto/contents/vol1/training/training.qmd:2783` | `TrainingDimensions` | `conv_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `64` |
| `book/quarto/contents/vol1/training/training.qmd:2797` | `TrainingDimensions` | `wave_batch_32` | `Scenarios.* or Ops.*` | scenario/workload policy | `32` |
| `book/quarto/contents/vol1/training/training.qmd:2798` | `TrainingDimensions` | `wave_batch_33` | `Scenarios.* or Ops.*` | scenario/workload policy | `33` |
| `book/quarto/contents/vol1/training/training.qmd:2799` | `TrainingDimensions` | `wave_batch_64` | `Scenarios.* or Ops.*` | scenario/workload policy | `64` |
| `book/quarto/contents/vol1/training/training.qmd:2800` | `TrainingDimensions` | `wave_batch_65` | `Scenarios.* or Ops.*` | scenario/workload policy | `65` |
| `book/quarto/contents/vol1/training/training.qmd:2949` | `VRAMRequirements` | `gpu_capacity` | `Hardware.*` | hardware-related quantitative input | `24 * GB` |
| `book/quarto/contents/vol1/training/training.qmd:2955` | `VRAMRequirements` | `batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol1/training/training.qmd:2956` | `VRAMRequirements` | `seq` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol1/training/training.qmd:3595` | `UtilizationGap` | `gpu_advertised_flops` | `Hardware.* or Hardware.Tech.*` | hardware specification | `300 * (TFLOP / second)` |
| `book/quarto/contents/vol1/training/training.qmd:3596` | `UtilizationGap` | `gpu_real_flops_min` | `Hardware.* or Hardware.Tech.*` | hardware specification | `90 * (TFLOP / second)` |
| `book/quarto/contents/vol1/training/training.qmd:3597` | `UtilizationGap` | `gpu_real_flops_max` | `Hardware.* or Hardware.Tech.*` | hardware specification | `150 * (TFLOP / second)` |
| `book/quarto/contents/vol1/training/training.qmd:3609` | `UtilizationGap` | `gpt2_attn_time_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50` |
| `book/quarto/contents/vol1/training/training.qmd:3610` | `UtilizationGap` | `gpt2_data_time_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `25` |
| `book/quarto/contents/vol1/training/training.qmd:3611` | `UtilizationGap` | `gpt2_compute_time_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `25` |
| `book/quarto/contents/vol1/training/training.qmd:3615` | `UtilizationGap` | `seq_pipeline_time` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `105` |
| `book/quarto/contents/vol1/training/training.qmd:3617` | `UtilizationGap` | `pipeline_speedup_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `(1 - opt_pipeline_time/seq_pipeline_time) * 100` |
| `book/quarto/contents/vol1/training/training.qmd:4036` | `PreprocessingScenarios` | `buffer_batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `256` |
| `book/quarto/contents/vol1/training/training.qmd:4037` | `PreprocessingScenarios` | `buffer_image_res` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:4741` | `AttentionMemoryCalc` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol1/training/training.qmd:5455` | `GPT2WalkthroughCalc` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol1/training/training.qmd:5456` | `GPT2WalkthroughCalc` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol1/training/training.qmd:5467` | `GPT2WalkthroughCalc` | `checkpoint_factor` | `Systems.Storage or Datasets.*` | storage/data fact | `4` |
| `book/quarto/contents/vol1/training/training.qmd:5468` | `GPT2WalkthroughCalc` | `recompute_overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `33` |
| `book/quarto/contents/vol1/training/training.qmd:6727` | `TrainingCarbonFootprint` | `cf_sustained_flops` | `Hardware.* or Hardware.Tech.*` | hardware specification | `150 * (TFLOP / second)` |
| `book/quarto/contents/vol1/training/training.qmd:6730` | `TrainingCarbonFootprint` | `cf_cpu_tdp_per_host` | `Hardware.* or Scenarios.*` | scenario/profile input | `200 * watt` |
| `book/quarto/contents/vol1/training/training.qmd:6880` | `FallaciesPitfallsSetup` | `fp_model_20b_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `20 * Bparam` |
| `book/quarto/contents/vol1/training/training.qmd:6882` | `FallaciesPitfallsSetup` | `fp_model_20b_fp16` | `Hardware.*` | hardware-related quantitative input | `20 * 2 * GB` |
| `book/quarto/contents/vol1/training/training.qmd:6883` | `FallaciesPitfallsSetup` | `fp_model_20b_optim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `20 * 4 * 2 * GB` |
| `book/quarto/contents/vol1/training/training.qmd:6885` | `FallaciesPitfallsSetup` | `fp_data_threshold_m` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol1/training/training.qmd:6890` | `FallaciesPitfallsSetup` | `fp_gpu_count` | `Systems.Clusters or Systems.Nodes` | fleet/topology fact | `8` |
| `book/quarto/contents/vol1/training/training.qmd:6891` | `FallaciesPitfallsSetup` | `fp_sync_overhead_min` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30` |
| `book/quarto/contents/vol1/training/training.qmd:6892` | `FallaciesPitfallsSetup` | `fp_sync_overhead_max` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50` |
| `book/quarto/contents/vol1/training/training.qmd:6895` | `FallaciesPitfallsSetup` | `fp_single_gpu_hours` | `Systems.*` | system-level fact | `24` |
| `book/quarto/contents/vol1/training/training.qmd:6896` | `FallaciesPitfallsSetup` | `fp_cluster_hours` | `Systems.*` | system-level fact | `36` |
| `book/quarto/contents/vol1/training/training.qmd:6899` | `FallaciesPitfallsSetup` | `fp_batch_small` | `Scenarios.* or Ops.*` | scenario/workload policy | `512` |
| `book/quarto/contents/vol1/training/training.qmd:6901` | `FallaciesPitfallsSetup` | `fp_batch_large` | `Scenarios.* or Ops.*` | scenario/workload policy | `4096` |
| `book/quarto/contents/vol1/training/training.qmd:6904` | `FallaciesPitfallsSetup` | `fp_failure_days_min` | `Systems.*` | system-level fact | `3` |
| `book/quarto/contents/vol1/training/training.qmd:6905` | `FallaciesPitfallsSetup` | `fp_failure_days_max` | `Systems.*` | system-level fact | `5` |
| `book/quarto/contents/vol1/training/training.qmd:6910` | `FallaciesPitfallsSetup` | `fp_training_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `48` |
| `book/quarto/contents/vol1/training/training.qmd:6911` | `FallaciesPitfallsSetup` | `fp_divergence_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `10000` |
| `book/quarto/contents/vol1/training/training.qmd:6914` | `FallaciesPitfallsSetup` | `fp_util_batch_256` | `Scenarios.* or Ops.*` | scenario/workload policy | `90` |
| `book/quarto/contents/vol1/training/training.qmd:6915` | `FallaciesPitfallsSetup` | `fp_util_batch_16_min` | `Scenarios.* or Ops.*` | scenario/workload policy | `60` |
| `book/quarto/contents/vol1/training/training.qmd:6916` | `FallaciesPitfallsSetup` | `fp_util_batch_16_max` | `Scenarios.* or Ops.*` | scenario/workload policy | `70` |
| `book/quarto/contents/vol1/training/training.qmd:6917` | `FallaciesPitfallsSetup` | `fp_effective_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `512` |
| `book/quarto/contents/vol1/training/training.qmd:6918` | `FallaciesPitfallsSetup` | `fp_physical_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `64` |

## vol2/backmatter/appendix_assumptions.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd:79` | `FleetQuickCalc` | `hours_per_month` | `Scenarios.* or Ops.*` | scenario/workload policy | `720 * hour` |

## vol2/backmatter/appendix_dam.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:44` | `DAMTaxonomy` | `ex1_gpu_util_pct` | `Systems.*` | system-level fact | `25` |
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:45` | `DAMTaxonomy` | `ex1_disk_sat_pct` | `Scenarios.* or Ops.*` | storage observation or utilization scenario | `100` |
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:49` | `DAMTaxonomy` | `ex2_latency` | `Scenarios.*` | unit-bearing scenario input | `0.050 * second` |
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:59` | `DAMTaxonomy` | `ex4_gpu_old_n` | `Systems.*` | system-level fact | `4` |
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:60` | `DAMTaxonomy` | `ex4_gpu_new_n` | `Systems.*` | system-level fact | `8` |
| `book/quarto/contents/vol2/backmatter/appendix_dam.qmd:61` | `DAMTaxonomy` | `ex4_cost_k` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `200` |

## vol2/backmatter/appendix_inference.qmd

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/backmatter/appendix_inference.qmd:273` | `LittlesLawVerify` | `arrival_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `100 / second` |
| `book/quarto/contents/vol2/backmatter/appendix_inference.qmd:274` | `LittlesLawVerify` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `16` |

## vol2/collective_communication

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:553` | `LlamaBudgetCalc` | `gradient_bf16` | `Hardware.*` | hardware-related quantitative input | `param_count * (2 * byte / param)` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:557` | `LlamaBudgetCalc` | `ib_alpha` | `Scenarios.*` | unit-bearing scenario input | `3 * microsecond` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:558` | `LlamaBudgetCalc` | `compute_per_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * ms` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1367` | `HierarchicalAllreduceCalc` | `gradient` | `Hardware.*` | hardware-related quantitative input | `1 * GB` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1370` | `HierarchicalAllreduceCalc` | `ib_alpha` | `Scenarios.*` | unit-bearing scenario input | `3 * microsecond` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1722` | `CompressionPayback` | `t_overhead_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1943` | `OverlapBudgetCalc` | `backward_per_layer` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `15 * ms` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1944` | `OverlapBudgetCalc` | `gradient_per_layer` | `Hardware.*` | hardware-related quantitative input | `880 * MB` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1945` | `OverlapBudgetCalc` | `bucket_size` | `Systems.Storage or Scenarios.*` | storage-related scenario input | `100 * MB` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1946` | `OverlapBudgetCalc` | `allreduce_per_bucket` | `Systems.Storage or Scenarios.*` | storage-related scenario input | `3 * ms` |
| `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:1948` | `OverlapBudgetCalc` | `params_b_overlap` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `7` |

## vol2/compute_infrastructure

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:1226` | `RooflineLandscapeMath` | `training_batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `2048` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:1442` | `RooflineInferenceMath` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `2048` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2284` | `BandwidthStaircase` | `transfer` | `Scenarios.*` | unit-bearing scenario input | `10 * GB` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2361` | `InfraFrontierNodeTpRecap` | `microbatch` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2362` | `InfraFrontierNodeTpRecap` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2595` | `NodeReliabilityScenario` | `run_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `14` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2771` | `NodeCapacityTrainMemRecap` | `host_dram` | `Hardware.* or Scenarios.*` | scenario/profile input | `2 * TB` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3118` | `H100TdpRackRecap` | `comm_phase_power` | `Infrastructure.*` | infrastructure input | `400 * watt` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3119` | `H100TdpRackRecap` | `transition` | `Scenarios.*` | unit-bearing scenario input | `100 * microsecond` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3603` | `CheckpointOverheadBudget` | `cadence` | `Scenarios.* or Ops.*` | scenario/workload policy | `10 * minute` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3604` | `CheckpointOverheadBudget` | `target_overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3605` | `CheckpointOverheadBudget` | `reference_write` | `Scenarios.*` | unit-bearing scenario input | `30 * second` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3606` | `CheckpointOverheadBudget` | `run_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `14` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3607` | `CheckpointOverheadBudget` | `checkpoints_retained` | `Systems.Storage or Datasets.*` | storage/data fact | `10` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3762` | `CheckpointComputeTradeoff` | `write_bw` | `Scenarios.*` | unit-bearing scenario input | `100 * (GB / second)` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3918` | `FabricSizingScenario` | `endpoint_links_per_gpu` | `Scenarios.* or Ops.*` | scenario/workload policy | `2` |
| `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:4176` | `A100FallaciesRecap` | `option_a_hbm` | `Hardware.* or Scenarios.*` | scenario/profile input | `192 * GB` |

## vol2/conclusion

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/conclusion/conclusion.qmd:132` | `ConclusionScaleFacts` | `llama_failures` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `419` |
| `book/quarto/contents/vol2/conclusion/conclusion.qmd:133` | `ConclusionScaleFacts` | `llama_days` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `54` |
| `book/quarto/contents/vol2/conclusion/conclusion.qmd:134` | `ConclusionScaleFacts` | `hours_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol2/conclusion/conclusion.qmd:412` | `FermiEstimate` | `brain_firing_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `1.0 / second` |
| `book/quarto/contents/vol2/conclusion/conclusion.qmd:413` | `FermiEstimate` | `brain_power` | `Infrastructure.*` | infrastructure input | `20 * watt` |

## vol2/data_storage

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:590` | `TextImageBandwidth` | `text_tokens_per_gpu` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:592` | `TextImageBandwidth` | `image_batch_per_gpu` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `256` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:593` | `TextImageBandwidth` | `image_bytes` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `150_000 * byte` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:594` | `TextImageBandwidth` | `step` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.2 * second` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:755` | `HbmMemoryBudget` | `activation_min` | `Scenarios.*` | unit-bearing scenario input | `10 * GB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:756` | `HbmMemoryBudget` | `activation_max` | `Scenarios.*` | unit-bearing scenario input | `20 * GB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:757` | `HbmMemoryBudget` | `comm_min` | `Scenarios.*` | unit-bearing scenario input | `2 * GB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:758` | `HbmMemoryBudget` | `comm_max` | `Scenarios.*` | unit-bearing scenario input | `4 * GB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:894` | `NVMeTierCalcs` | `pfs_node_bw` | `Systems.Storage or Scenarios.*` | scenario/profile input | `4.0 * (GB / second)` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:895` | `NVMeTierCalcs` | `images_per_sec` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1000` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:896` | `NVMeTierCalcs` | `img_size` | `Scenarios.*` | unit-bearing scenario input | `150 * KB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:897` | `NVMeTierCalcs` | `hdd_iops` | `Systems.Storage` | storage subsystem fact | `100` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:898` | `NVMeTierCalcs` | `imagenet_images` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1_280_000` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1204` | `ObjectStorageCost` | `dataset_size_tb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1283` | `GlacierCost` | `dataset_size_tb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1284` | `GlacierCost` | `cost_glacier_gb_mo` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.004` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1383` | `PipelineCalcs` | `img_size` | `Scenarios.*` | unit-bearing scenario input | `150 * KB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1384` | `PipelineCalcs` | `batch_img_gpu` | `Scenarios.* or Ops.*` | scenario/workload policy | `256` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1386` | `PipelineCalcs` | `t_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `200 * ms` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1387` | `PipelineCalcs` | `t_comp` | `Scenarios.*` | unit-bearing scenario input | `200 * ms` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1388` | `PipelineCalcs` | `t_io` | `Scenarios.*` | unit-bearing scenario input | `250 * ms` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1766` | `HashValidationCost` | `dataset` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100 * TB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1767` | `HashValidationCost` | `throughput_per_core` | `Scenarios.*` | unit-bearing scenario input | `18.5 * (MB / second)` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1818` | `GDSLatency` | `images_per_s_per_gpu` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `8_000` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:1964` | `EconRatios` | `dataset_size` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100 * TB` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2110` | `BuildVsBuyStorageEconomics` | `capex_low` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `3_000_000` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2111` | `BuildVsBuyStorageEconomics` | `capex_high` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5_000_000` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2116` | `BuildVsBuyStorageEconomics` | `dataset_pb` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `1` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2236` | `CheckpointModelIntro` | `ckpt_total` | `Systems.Storage or Datasets.*` | storage/data fact | `gpt3_params * (10 * (byte / param))` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2339` | `CheckpointStormCalc` | `checkpoint_interval` | `Scenarios.TrainingRuns or Systems.Reliability` | checkpoint policy/workload cadence | `600 * second` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2417` | `CheckpointFleetRetention` | `checkpoint_count_val` | `Systems.Storage or Datasets.*` | storage/data fact | `30 * 24 * 60 // 10` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2419` | `CheckpointFleetRetention` | `retention_recent` | `Systems.Storage or Datasets.*` | storage/data fact | `3 * ckpt_total` |
| `book/quarto/contents/vol2/data_storage/data_storage.qmd:2420` | `CheckpointFleetRetention` | `retained_every_100_count_val` | `Scenarios.TrainingRuns or Systems.Reliability` | checkpoint policy/workload cadence | `checkpoint_count_val // 100` |

## vol2/distributed_training

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:102` | `DistTrainReliabilityFacts` | `reliability_node_count` | `Systems.Clusters or Systems.Nodes` | fleet/topology fact | `100` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:103` | `DistTrainReliabilityFacts` | `node_hourly_survival` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.999` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:424` | `HierarchicalAllReduceDebug` | `tensor_bytes` | `Scenarios.*` | unit-bearing scenario input | `3.0 * GB` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:598` | `Scaling8GPU` | `single_gpu_step_s` | `Scenarios.* or Ops.*` | scenario/workload policy | `1.8` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:599` | `Scaling8GPU` | `batch_per_gpu_val` | `Scenarios.* or Ops.*` | scenario/workload policy | `16` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:600` | `Scaling8GPU` | `fits_mem` | `Scenarios.*` | unit-bearing scenario input | `32 * GB` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:1501` | `ScalingWorkers` | `critical_batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `4_000` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:1684` | `ModelParallelMemoryFacts` | `bytes_adam_state` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12 * (byte / param)` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:1772` | `A100CapacityContext` | `bytes_adam_state` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12 * (byte / param)` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:1774` | `A100CapacityContext` | `activation_approx` | `Scenarios.*` | unit-bearing scenario input | `50 * GB` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2312` | `TensorParallel3DTraffic` | `microbatch` | `Scenarios.* or Ops.*` | scenario/workload policy | `4` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2313` | `TensorParallel3DTraffic` | `sequence_length` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2314` | `TensorParallel3DTraffic` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `12288` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2316` | `TensorParallel3DTraffic` | `layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `96` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2317` | `TensorParallel3DTraffic` | `allreduces_per_layer` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2399` | `HybridMemoryAccounting` | `simplified_optimizer_bytes_per_param` | `Hardware.*` | hardware-related quantitative input | `2 * (BYTES_FP32 / param)` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2400` | `HybridMemoryAccounting` | `full_adam_bytes_per_param` | `Hardware.*` | hardware-related quantitative input | `3 * (BYTES_FP32 / param)` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2772` | `RLHFKVCache` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2773` | `RLHFKVCache` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `256` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2850` | `RLHFBudget` | `bytes_train` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `16 * (byte / param)` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2859` | `RLHFBudget` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1024` |
| `book/quarto/contents/vol2/distributed_training/distributed_training.qmd:2860` | `RLHFBudget` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `256` |

## vol2/edge_intelligence

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:1162` | `StorageWall` | `model_params_m` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:1165` | `StorageWall` | `adapter_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50000` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:2014` | `FederatedSavings` | `model_params_m` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `5` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:2016` | `FederatedSavings` | `bits_per_update_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:2523` | `FederatedConvergence` | `local_epochs` | `Scenarios.* or Ops.*` | scenario/workload policy | `5` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:2524` | `FederatedConvergence` | `reduced_local_epochs` | `Scenarios.* or Ops.*` | scenario/workload policy | `2` |
| `book/quarto/contents/vol2/edge_intelligence/edge_intelligence.qmd:2525` | `FederatedConvergence` | `epsilon` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.01` |

## vol2/fault_tolerance

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:1057` | `MemBandwidthProtection` | `ecc_overhead` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.125` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2289` | `YoungDalyOptimal` | `mtbf` | `Scenarios.* or Ops.*` | scenario/workload policy | `3.69 * hour` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2400` | `CheckpointOverheadCalc` | `gpt35_size_gb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `240` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2401` | `CheckpointOverheadCalc` | `bert_size_gb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1.3` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2403` | `CheckpointOverheadCalc` | `resnet_size_gb` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.1` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2407` | `CheckpointOverheadCalc` | `mtbf_hours` | `Systems.*` | system-level fact | `5` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2408` | `CheckpointOverheadCalc` | `mtbf_s` | `Systems.*` | system-level fact | `mtbf_hours * 3600` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2543` | `CheckpointDebug` | `pcie_bw` | `Hardware.* or Hardware.Tech.*` | hardware specification | `32 * (GB / second)` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2549` | `CheckpointDebug` | `overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2550` | `CheckpointDebug` | `residual_overhead_pct` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2551` | `CheckpointDebug` | `base_weeks` | `Scenarios.* or Ops.*` | scenario/workload policy | `2` |
| `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:2552` | `CheckpointDebug` | `extra_cost_k` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `500` |

## vol2/fleet_orchestration

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:944` | `TopologyPlacement` | `lat_rack_ms` | `Scenarios.* or Ops.*` | scenario/workload policy | `85` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1131` | `ElasticScaling` | `window_h` | `Scenarios.* or Ops.*` | scenario/workload policy | `24` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1435` | `SpotTrainingEconomics` | `run_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `14 * HOURS_PER_DAY` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1438` | `SpotTrainingEconomics` | `checkpoint_overhead_fraction` | `Systems.Storage or Datasets.*` | storage/data fact | `0.05` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1440` | `SpotTrainingEconomics` | `restart_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.5` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1704` | `AutoscalingLag` | `qps_per_replica` | `Scenarios.* or Ops.*` | scenario/workload policy | `5` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1705` | `AutoscalingLag` | `traffic_start` | `Scenarios.* or Ops.*` | scenario/workload policy | `40` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1706` | `AutoscalingLag` | `traffic_end` | `Scenarios.* or Ops.*` | scenario/workload policy | `80` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1814` | `GpuSharingRoi` | `small_footprint` | `Scenarios.*` | unit-bearing scenario input | `26 * GiB` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1818` | `GpuSharingRoi` | `gpus_per_large` | `Systems.*` | system-level fact | `8` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:1819` | `GpuSharingRoi` | `models_per_gpu_shared` | `Systems.*` | system-level fact | `2` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2032` | `ChargebackExample` | `days` | `Scenarios.* or Ops.*` | scenario/workload policy | `34` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2103` | `PreemptionTax` | `lost_duration` | `Scenarios.* or Ops.*` | scenario/workload policy | `15 * minute` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2104` | `PreemptionTax` | `reload_duration` | `Scenarios.* or Ops.*` | scenario/workload policy | `20 * minute` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2105` | `PreemptionTax` | `warmup_duration` | `Scenarios.* or Ops.*` | scenario/workload policy | `10 * minute` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2107` | `PreemptionTax` | `preemptions_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `12` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2160` | `QuotaHoardingCost` | `weeks` | `Scenarios.* or Ops.*` | scenario/workload policy | `3` |
| `book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd:2161` | `QuotaHoardingCost` | `hours_per_week` | `Scenarios.* or Ops.*` | scenario/workload policy | `7 * HOURS_PER_DAY` |

## vol2/inference

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/inference/inference.qmd:315` | `InferenceEconomics` | `qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `10000` |
| `book/quarto/contents/vol2/inference/inference.qmd:316` | `InferenceEconomics` | `serve_duration_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `730` |
| `book/quarto/contents/vol2/inference/inference.qmd:317` | `InferenceEconomics` | `cost_per_query` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `1e-5` |
| `book/quarto/contents/vol2/inference/inference.qmd:1526` | `A100HardwareScenario` | `activation_transfer` | `Scenarios.*` | unit-bearing scenario input | `100 * MB` |
| `book/quarto/contents/vol2/inference/inference.qmd:1639` | `RecsysLatencyBreakdown` | `routing` | `Scenarios.*` | unit-bearing scenario input | `0.2 * ms` |
| `book/quarto/contents/vol2/inference/inference.qmd:1640` | `RecsysLatencyBreakdown` | `accumulation` | `Scenarios.*` | unit-bearing scenario input | `0.5 * ms` |
| `book/quarto/contents/vol2/inference/inference.qmd:1641` | `RecsysLatencyBreakdown` | `embedding` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2.0 * ms` |
| `book/quarto/contents/vol2/inference/inference.qmd:1642` | `RecsysLatencyBreakdown` | `feature` | `Scenarios.*` | unit-bearing scenario input | `1.0 * ms` |
| `book/quarto/contents/vol2/inference/inference.qmd:1643` | `RecsysLatencyBreakdown` | `ranking` | `Scenarios.*` | unit-bearing scenario input | `1.5 * ms` |
| `book/quarto/contents/vol2/inference/inference.qmd:2010` | `KVCacheCapacityEstimator` | `d_head` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |
| `book/quarto/contents/vol2/inference/inference.qmd:2012` | `KVCacheCapacityEstimator` | `context_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `131_072` |
| `book/quarto/contents/vol2/inference/inference.qmd:2013` | `KVCacheCapacityEstimator` | `total_hbm` | `Hardware.* or Scenarios.*` | scenario/profile input | `640 * GB` |
| `book/quarto/contents/vol2/inference/inference.qmd:2015` | `KVCacheCapacityEstimator` | `system_reserved` | `Scenarios.*` | unit-bearing scenario input | `20 * GB` |
| `book/quarto/contents/vol2/inference/inference.qmd:2330` | `PrefixCachingAnalysis` | `system_prompt_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2000` |
| `book/quarto/contents/vol2/inference/inference.qmd:2331` | `PrefixCachingAnalysis` | `concurrent_users` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol2/inference/inference.qmd:2332` | `PrefixCachingAnalysis` | `avg_response_tokens` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `500` |
| `book/quarto/contents/vol2/inference/inference.qmd:2335` | `PrefixCachingAnalysis` | `head_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `8192` |
| `book/quarto/contents/vol2/inference/inference.qmd:2431` | `KVCacheCalc` | `kv_layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `80` |
| `book/quarto/contents/vol2/inference/inference.qmd:2432` | `KVCacheCalc` | `kv_heads` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `64` |
| `book/quarto/contents/vol2/inference/inference.qmd:2433` | `KVCacheCalc` | `kv_head_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `128` |
| `book/quarto/contents/vol2/inference/inference.qmd:2434` | `KVCacheCalc` | `kv_seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol2/inference/inference.qmd:3114` | `TpSpeedupCalc` | `seq_per_layer_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30.0` |
| `book/quarto/contents/vol2/inference/inference.qmd:3115` | `TpSpeedupCalc` | `tp8_per_layer_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4.35` |
| `book/quarto/contents/vol2/inference/inference.qmd:3118` | `TpSpeedupCalc` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `8192` |
| `book/quarto/contents/vol2/inference/inference.qmd:3120` | `TpSpeedupCalc` | `activation` | `Scenarios.*` | unit-bearing scenario input | `8 * MB` |
| `book/quarto/contents/vol2/inference/inference.qmd:3269` | `MoEEconomics` | `dense_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `400 * Bparam` |
| `book/quarto/contents/vol2/inference/inference.qmd:3613` | `MetaEmbeddingScaleCheck` | `storage` | `Systems.Storage or Datasets.*` | storage/data fact | `100 * TB` |
| `book/quarto/contents/vol2/inference/inference.qmd:3758` | `AllReduceInterconnect` | `message` | `Scenarios.*` | unit-bearing scenario input | `8 * MB` |
| `book/quarto/contents/vol2/inference/inference.qmd:3759` | `AllReduceInterconnect` | `layer_budget` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30 * millisecond` |
| `book/quarto/contents/vol2/inference/inference.qmd:3981` | `HeterogeneousGpuCluster` | `h100_cap_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol2/inference/inference.qmd:3982` | `HeterogeneousGpuCluster` | `a100_cap_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `600` |
| `book/quarto/contents/vol2/inference/inference.qmd:3983` | `HeterogeneousGpuCluster` | `target_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `15000` |
| `book/quarto/contents/vol2/inference/inference.qmd:4473` | `NoisyNeighborQuotaScenario` | `tenant_base_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol2/inference/inference.qmd:4474` | `NoisyNeighborQuotaScenario` | `tenant_burst_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `500` |
| `book/quarto/contents/vol2/inference/inference.qmd:4475` | `NoisyNeighborQuotaScenario` | `tenant_throttled_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `120` |
| `book/quarto/contents/vol2/inference/inference.qmd:5514` | `QuantizedServingCapacity` | `a100_kv_budget` | `Hardware.*` | hardware-related quantitative input | `80 * GB` |

## vol2/introduction

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/introduction/introduction.qmd:1657` | `GPT3SyncTax` | `t_compute_iter` | `Scenarios.*` | unit-bearing scenario input | `1.2 * second` |
| `book/quarto/contents/vol2/introduction/introduction.qmd:1658` | `GPT3SyncTax` | `million_steps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1_000_000` |
| `book/quarto/contents/vol2/introduction/introduction.qmd:1662` | `GPT3SyncTax` | `energy_per_bit` | `Infrastructure.*` | infrastructure input | `15.0 * pJ / bit` |
| `book/quarto/contents/vol2/introduction/introduction.qmd:2055` | `EdgeLatencyDistance` | `seconds_per_hour` | `Scenarios.* or Ops.*` | scenario/workload policy | `3600` |
| `book/quarto/contents/vol2/introduction/introduction.qmd:2096` | `LinearScalingFallacy` | `coord_overhead_fraction` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.40` |

## vol2/network_fabrics

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:553` | `NetworkAlphaBeta` | `n_small` | `Scenarios.*` | unit-bearing scenario input | `4000 * byte` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:554` | `NetworkAlphaBeta` | `n_large` | `Scenarios.*` | unit-bearing scenario input | `350 * MB` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:555` | `NetworkAlphaBeta` | `n_demo_large` | `Scenarios.*` | unit-bearing scenario input | `100 * MB` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:771` | `AllReduceBottleneck` | `gpu_util` | `Systems.*` | system-level fact | `0.50` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:772` | `AllReduceBottleneck` | `flops_per_sample_base` | `Models.* or Scenarios.TrainingRuns` | workload compute requirement | `5e13 * flop` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:773` | `AllReduceBottleneck` | `grad_bytes_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4 * byte / param` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:1055` | `BisectionScaling` | `gradient` | `Hardware.*` | hardware-related quantitative input | `100 * GB` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:1330` | `PfcStormRisk` | `links_per_gpu` | `Scenarios.* or Ops.*` | scenario/workload policy | `3` |
| `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:1838` | `BandwidthFallacy` | `msg` | `Scenarios.*` | unit-bearing scenario input | `10 * KB` |

## vol2/ops_scale

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:324` | `PlatformRoi` | `engineer_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:325` | `PlatformRoi` | `platform_cost_per_month` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `120000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:388` | `PlatformEconomics` | `hours_per_model_month` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `40` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:390` | `PlatformEconomics` | `engineer_cost_hr` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:391` | `PlatformEconomics` | `platform_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2 * MILLION` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:393` | `PlatformEconomics` | `hours_saved_per_model` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:597` | `TrainingCapacityCost` | `training_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `30` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:650` | `MaintenanceDividend` | `years` | `Scenarios.*` | unit-bearing scenario input | `3` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:651` | `MaintenanceDividend` | `engineer_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:767` | `DebtPriority` | `config_resolution_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:769` | `DebtPriority` | `monitoring_impact` | `Scenarios.* or Ops.*` | scenario/workload policy | `3` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:770` | `DebtPriority` | `monitoring_frequency` | `Scenarios.* or Ops.*` | scenario/workload policy | `2` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:771` | `DebtPriority` | `monitoring_resolution_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `2` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:775` | `DebtPriority` | `pipeline_resolution_cost` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `3` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:777` | `DebtPriority` | `deployment_delay_weeks` | `Scenarios.* or Ops.*` | scenario/workload policy | `6` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:779` | `DebtPriority` | `models_per_year` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `40` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:780` | `DebtPriority` | `hours_per_week` | `Scenarios.* or Ops.*` | scenario/workload policy | `40` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:781` | `DebtPriority` | `engineer_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:846` | `DeploymentRoi` | `engineer_rate` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `150` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:847` | `DeploymentRoi` | `manual_hours_per_deploy` | `Scenarios.* or Ops.*` | scenario/workload policy | `10` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:848` | `DeploymentRoi` | `auto_hours_per_deploy` | `Scenarios.* or Ops.*` | scenario/workload policy | `0.5` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:849` | `DeploymentRoi` | `automation_setup_hours` | `Scenarios.* or Ops.*` | scenario/workload policy | `120` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1110` | `BuildVsBuyFigureScenario` | `network_capex_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5_000_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1111` | `BuildVsBuyFigureScenario` | `facility_capex_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `10_000_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1112` | `BuildVsBuyFigureScenario` | `annual_opex_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `1_500_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1297` | `TenKGpuClusterTco` | `network_capex_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `25_000_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1298` | `TenKGpuClusterTco` | `facility_capex_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `75_000_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1301` | `TenKGpuClusterTco` | `annual_staffing_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5_000_000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:2396` | `SilentFailure` | `qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `5000` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:3135` | `CheckpointComputeTradeoff` | `write_bw` | `Scenarios.*` | unit-bearing scenario input | `100 * (GB / second)` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:3429` | `DriftLatency` | `power` | `Infrastructure.*` | infrastructure input | `0.80` |
| `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:3430` | `DriftLatency` | `dl_qps` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |

## vol2/performance_engineering

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:578` | `WorkloadIntensityCalc` | `decode_batch` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:579` | `WorkloadIntensityCalc` | `decode_hidden` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:583` | `WorkloadIntensityCalc` | `gemm_flops` | `Models.* or Scenarios.TrainingRuns` | workload compute requirement | `2 * M_dim * C_dim * K_dim * flop` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:593` | `WorkloadIntensityCalc` | `decode_flops_val` | `Models.* or Scenarios.TrainingRuns` | workload compute requirement | `2 * decode_batch * decode_hidden * decode_hidden * flop` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:787` | `FusionTrafficCalc` | `hidden_dim` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:788` | `FusionTrafficCalc` | `batch_size` | `Scenarios.* or Ops.*` | scenario/workload policy | `2048` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:791` | `FusionTrafficCalc` | `num_layers` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `80` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:888` | `FlashAttentionSavings` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `8192` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1188` | `KVCacheAnalysis` | `seq_len` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4096` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1192` | `KVCacheAnalysis` | `gpu_total` | `Hardware.*` | hardware-related quantitative input | `80 * GB` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1611` | `OverlapCalc` | `tokens_per_gpu` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2048` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1612` | `OverlapCalc` | `forward_flops` | `Models.* or Scenarios.TrainingRuns` | workload compute requirement | `2 * m.parameters.to(param).magnitude * tokens_per_gpu * flop` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1874` | `LayerNormRoofline` | `achieved_flops` | `Models.* or Scenarios.TrainingRuns` | workload compute requirement | `15 * (TFLOPs / second)` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:1875` | `LayerNormRoofline` | `achieved_bw` | `Scenarios.*` | unit-bearing scenario input | `2.8 * (TB / second)` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:2117` | `FleetEfficiencyCalc` | `local_tokens_per_step` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2204` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:2118` | `FleetEfficiencyCalc` | `fleet_tokens_per_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `35446` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:2121` | `FleetEfficiencyCalc` | `t_local_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `180.0 * ms` |
| `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:2122` | `FleetEfficiencyCalc` | `t_fleet_step` | `Scenarios.* or Ops.*` | scenario/workload policy | `245.0 * ms` |

## vol2/responsible_ai

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:248` | `FairnessTaxAnalysis` | `default_prob_threshold` | `Scenarios.* or Ops.*` | scenario/workload policy | `30` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:251` | `FairnessTaxAnalysis` | `default_prob_threshold_parity` | `Scenarios.* or Ops.*` | scenario/workload policy | `50` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1733` | `PrivacyPriceAnalysis` | `base_cost_m` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `4.6` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1738` | `PrivacyPriceAnalysis` | `cost_mult_strong` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `3.0` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1739` | `PrivacyPriceAnalysis` | `overhead_mod` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `30.0` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1836` | `UnlearningCostAnalysis` | `full_cost_m` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `4.6` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1838` | `UnlearningCostAnalysis` | `full_time_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `34` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:1842` | `UnlearningCostAnalysis` | `deletion_requests_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `1000` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2379` | `RepresentationTax` | `images_per_subgroup` | `Datasets.* or Scenarios.DataWorkloads` | dataset/workload specification | `100_000` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2380` | `RepresentationTax` | `low_cost_per_image` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2381` | `RepresentationTax` | `high_cost_per_image` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `200` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2382` | `RepresentationTax` | `overhead_low` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.30` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2383` | `RepresentationTax` | `overhead_high` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.50` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2661` | `ExplainabilityRetrofitCost` | `overhead_low_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `50` |
| `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd:2662` | `ExplainabilityRetrofitCost` | `overhead_high_ms` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `200` |

## vol2/robust_ai

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/robust_ai/robust_ai.qmd:2539` | `HuberOutlierScaling` | `normal_error_multiple` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |

## vol2/security_privacy

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:164` | `DPCostAnalysis` | `epsilon` | `Scenarios.* or Ops.*` | scenario/workload policy | `1.0` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1402` | `DefenseOverheadAnalysis` | `daily_queries` | `Scenarios.* or Ops.*` | scenario/workload policy | `1 * MILLION` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1403` | `DefenseOverheadAnalysis` | `daily_users` | `Scenarios.* or Ops.*` | scenario/workload policy | `10 * THOUSAND` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1404` | `DefenseOverheadAnalysis` | `training_cost_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `5 * THOUSAND` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1405` | `DefenseOverheadAnalysis` | `extraction_queries` | `Scenarios.* or Ops.*` | scenario/workload policy | `5 * MILLION` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1407` | `DefenseOverheadAnalysis` | `attacker_daily_queries` | `Scenarios.* or Ops.*` | scenario/workload policy | `25 * THOUSAND` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1408` | `DefenseOverheadAnalysis` | `extraction_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `200` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1409` | `DefenseOverheadAnalysis` | `free_tier_queries` | `Scenarios.* or Ops.*` | scenario/workload policy | `100` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1410` | `DefenseOverheadAnalysis` | `paid_tier_price_usd_month` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `50` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1411` | `DefenseOverheadAnalysis` | `paid_tier_queries` | `Scenarios.* or Ops.*` | scenario/workload policy | `10_000` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1412` | `DefenseOverheadAnalysis` | `alert_queries_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `5_000` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1414` | `DefenseOverheadAnalysis` | `price_per_query_usd` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.001` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1415` | `DefenseOverheadAnalysis` | `legit_queries_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `300` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1417` | `DefenseOverheadAnalysis` | `output_accuracy_no_defense_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `90` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1418` | `DefenseOverheadAnalysis` | `output_accuracy_with_defense_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `72` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1419` | `DefenseOverheadAnalysis` | `top5_accuracy_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `99.2` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1420` | `DefenseOverheadAnalysis` | `baseline_top5_accuracy_pct` | `Scenarios.* or Ops.*` | scenario/workload policy | `99.3` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1422` | `DefenseOverheadAnalysis` | `monitoring_ms` | `Scenarios.* or Ops.*` | scenario/workload policy | `2.0` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:2560` | `TEEMemoryFootprint` | `small_model_limit_q` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `10 * MB` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:2616` | `EncryptionOverhead` | `t_inference_q` | `Scenarios.*` | unit-bearing scenario input | `20 * ms` |
| `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:2617` | `EncryptionOverhead` | `overhead_aes_q` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.5 * ms` |

## vol2/sustainable_ai

| File:Line | Cell | Symbol | Target | Reason | RHS |
|---|---|---|---|---|---|
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:259` | `CarbonFrontier` | `energy` | `Infrastructure.*` | infrastructure input | `10_000 * MWh` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:312` | `AutoPlacement` | `energy` | `Infrastructure.*` | infrastructure input | `10_000 * MWh` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:313` | `AutoPlacement` | `carbon_tax` | `Infrastructure.* or Scenarios.Sustainability` | infrastructure/sustainability fact | `100.0` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:314` | `AutoPlacement` | `duration_days` | `Infrastructure.*` | infrastructure input | `(energy.to(MWh).magnitude / 143.0) / 24.0` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:698` | `EnergyWallScenario` | `grid_annual_growth` | `Infrastructure.* or Scenarios.Sustainability` | infrastructure/sustainability fact | `0.02` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:871` | `LifecycleCarbonEstimate` | `training_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `30` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:879` | `LifecycleCarbonEstimate` | `amortization_window_months` | `Scenarios.* or Ops.*` | scenario/workload policy | `1` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1081` | `PueEfficiency` | `p_it` | `Scenarios.*` | unit-bearing scenario input | `2.0 * MW` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1084` | `PueEfficiency` | `elec_price` | `Infrastructure.Pricing.* or Scenarios.*` | economic input or scenario price | `0.07` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1613` | `TrainingEmissions` | `training_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `14` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1743` | `EmbodiedCarbonAmort` | `t_job` | `Scenarios.* or Ops.*` | scenario/workload policy | `10 * hour` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1816` | `TrainingEmbodiedRecap` | `training_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `14` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1908` | `InferenceLifecycleExample` | `queries_per_day` | `Scenarios.* or Ops.*` | scenario/workload policy | `10_000_000` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1909` | `InferenceLifecycleExample` | `kwh_per_query` | `Scenarios.*` | unit-bearing scenario input | `0.001 * kWh` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:1914` | `InferenceLifecycleExample` | `training_days` | `Scenarios.* or Ops.*` | scenario/workload policy | `14` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2379` | `GridQueue` | `cluster_power` | `Systems.*` | system-level fact | `7 * MW` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2508` | `RackPowerBudget` | `nodes_per_rack` | `Systems.Clusters or Systems.Nodes` | fleet/topology fact | `4` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2510` | `RackPowerBudget` | `host_kw` | `Scenarios.*` | unit-bearing scenario input | `3.2 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2511` | `RackPowerBudget` | `nvswitch_kw` | `Scenarios.*` | unit-bearing scenario input | `1.6 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2512` | `RackPowerBudget` | `ib_kw` | `Scenarios.*` | unit-bearing scenario input | `0.8 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2513` | `RackPowerBudget` | `conversion_kw` | `Scenarios.*` | unit-bearing scenario input | `2.8 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2514` | `RackPowerBudget` | `cooling_kw` | `Infrastructure.* or Scenarios.*` | scenario/profile input | `2.7 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2641` | `SustainableCoolingRackPowerRecap` | `nodes_per_rack` | `Systems.Clusters or Systems.Nodes` | fleet/topology fact | `4` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2642` | `SustainableCoolingRackPowerRecap` | `non_gpu_power` | `Hardware.*` | hardware-related quantitative input | `10.3 * kilowatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2700` | `PueSavings` | `it_power` | `Infrastructure.*` | infrastructure input | `7.0 * MW` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:2706` | `PueSavings` | `gpt3_households` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `120.0` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3191` | `OnDeviceLearningEnergy` | `model_params` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `1_000_000_000` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3192` | `OnDeviceLearningEnergy` | `forward_nj_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `2` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3193` | `OnDeviceLearningEnergy` | `backward_nj_per_param` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `4` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3325` | `WakeWordPower` | `vad_power` | `Infrastructure.*` | infrastructure input | `0.01 * milliwatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3327` | `WakeWordPower` | `wake_detector_power` | `Infrastructure.*` | infrastructure input | `0.1 * milliwatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3329` | `WakeWordPower` | `full_model_power` | `Infrastructure.*` | infrastructure input | `10 * milliwatt` |
| `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3330` | `WakeWordPower` | `full_model_duty_s` | `Models.* or Scenarios.TrainingRuns` | model/workload specification | `0.05` |
