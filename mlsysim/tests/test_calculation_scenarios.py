from mlsysim.core.series_loader import load_series
'Calculation scenarios: named inputs plus the solver that owns the arithmetic.\n\nChapters and lecture slides quote these values; pinning them here means a\nregistry or solver change that moves one shows up as a failing test instead\nof a silent change in print. Values were checked against the hand-derived\nchapter results when the scenarios were added (2026-10-20).\n'
import pytest
from mlsysim import Models, Scenarios

def test_decode_bound_llama70b_h100():
    r = Scenarios.Llama3_70B_DecodeH100.result
    assert r.bottleneck == 'Memory'

def test_tensor_parallel_step():
    s = Scenarios.Gpt3TensorParallelStepDgxH100
    r = s.result
    assert r.collectives_per_step == 192
    pcie = s.variant(link_bandwidth=Scenarios.BufferTransfer10GB.links['PCIe']).result

def test_memory_interface_widths():
    from mlsysim import Hardware
    m = Hardware.Tech.Memory
    assert m.HBM3.interface_width.to('bit').magnitude == 1024
    assert m.DDR5.interface_width.to('bit').magnitude == 64

def test_series_lookups():
    from mlsysim import Literature
    s = load_series('vol2/02_compute_infrastructure/data/ai_supercomputers.csv')
    last = s.largest('power_mw', 'first_operational_date', '2025-12-31')
    assert last['first_operational_date'] <= '2025-12-31'
    assert last['power_mw'] == max((r['power_mw'] for r in s.entries if r['power_mw'] and r['first_operational_date'] <= '2025-12-31'))

def test_communication_intensity():
    r = Scenarios.Gpt3DataParallelIntensity.result
    assert r.local_tokens == 8192

def test_growth_and_runs():
    g = Scenarios.NotableRunGrowth2012to2024.result
    assert g.run_count == 146

def test_spine_optics_buildout():
    s = Scenarios.SpineOpticsTraining10K
    assert s.n_links == 20000

def test_nccl_allreduce_fit():
    r = Scenarios.NcclAllReduceH100NodeFit.result
    assert r.row_count == 32

def test_ring_allreduce_across_oversubscribed_spine():
    r = Scenarios.RingAllReduceAcross4to1Spine.result
    assert r.devices_per_side == 512

def test_torus_against_fat_tree():
    s = Scenarios.TorusVsFatTree4K
    assert (s.n_endpoints, s.side, s.torus_cut_links, s.fat_tree_cut_links) == (4096, 16, 512, 2048)
    assert (s.torus_diameter, s.fat_tree_path_switches) == (24, 5)

def test_component_exposure():
    t = Scenarios.TransceiverDegradationRoce4K
    assert t.n_components == 24576
    links = Scenarios.DegradedLinksReferenceFabric

def test_storage_footprint_of_the_175b_run():
    r = Scenarios.Gpt3StorageFootprint.result
    assert r.saves == 4320
    assert r.shard_count == 2000
    assert r.archived_checkpoints == 44

def test_input_demand_stall_and_prefetch():
    st = Scenarios.OverlappedInputStall.result
    p = Scenarios.PrefetchDepthByTier.result
    assert p.depths == {'nvme': 1, 'pfs': 1, 'object': 3}

def test_measured_storage_demand():
    r = Scenarios.MLPerfStorageV1Demand.result
    assert r.result_count == 124

def test_warehouse_elementary_stop():
    r = Scenarios.WarehouseAisleStopAtDriveLimit.result
    assert r.fits
    from mlsysim.core.units import Q_

def test_floor_traction_loss():
    r = Scenarios.HydraulicFluidSlickTraction.result
    assert not r.premise_holds

def test_robot_policy_compute_gap():
    v = Scenarios.OpenVlaComputeGap.result
    assert v.reference_model == 'Llama 3.1-405B'
    g = Scenarios.Gr00tN1ComputeGap.result
    assert g.reference_model == 'Grok 4'
    assert g.domain_models_with_compute == 25

def test_append_only_context():
    r = Scenarios.AppendOnlyRepairContext.result
    assert r.last_call_tokens.to('token').magnitude == 51500
    assert r.cumulative_input_tokens.to('token').magnitude == 892500

def test_task_horizon_trend_reproduces_source():
    s = Scenarios.FrontierTaskHorizonTrend
    r = s.result
    assert r.doubling_recent.to('day').magnitude == pytest.approx(s.published_recent_doubling.to('day').magnitude, abs=1)
    assert r.first_date == '2019-02-14'

def test_trajectory_resend():
    r = Scenarios.CodingAgentResend30Turns.result
    assert r.turns == 30
    assert r.resent_tokens == 892500
    assert r.first_prompt_term == 240000 and r.appended_term == 652500
    assert r.reused_prefix_tokens == 51500
    assert r.output_tokens == 6000
    r60 = Scenarios.CodingAgentResend60Turns.result
    assert r60.resent_tokens == 3135000 and r60.reused_prefix_tokens == 96500
    assert Scenarios.CodingAgentResend90Turns.result.resent_tokens == 6727500

def test_model_calls():
    c = Scenarios.CodingAgentFirstTurnCall.result
    w = Scenarios.CodingAgentWholeFileEdit.result
    e = Scenarios.CodingAgentLocalizedEdit.result
    s = Scenarios.SupportAgentTurnCall.result
    k = Scenarios.CanceledStreamDecodeSavings
    assert k.skipped_steps == 4000

def test_admission_sampling_checker():
    a = Scenarios.MultimodalContextAdmission.result
    assert (a.input_ceiling, a.headroom, a.total_demand, a.excess) == (7168, 1168, 9072, 880)
    assert not a.admitted
    t = Scenarios.TemperatureNucleusExample.result
    p = Scenarios.PatchCheckerPosterior.result

def test_allreduce_schedules():
    b = Scenarios.DdpBucketRingNdr.result
    assert b.ring_rounds[1024] == 2046
    g = Scenarios.GradientBufferAllReduceNdr.result
    assert g.star_ring_limit[64] == 32
    assert g.tree_rounds == {8: 6, 64: 12, 1024: 20}
    r = Scenarios.RingVsTreeSmallBuffer.result
    assert r.ring_startup_time[64].to('microsecond').magnitude == 1260
    assert r.tree_startup_time[64].to('microsecond').magnitude == 120
    s = Scenarios.Llama2_70BGradientRing64Ndr

def test_hierarchical_allreduce():
    h = Scenarios.HierarchicalAllReduce8x8DgxH100.result
    a = Scenarios.HierarchicalAllReduceReferenceCluster.result
    assert 4 < a.speedup_vs_flat_ring < 5

def test_moe_dispatch():
    d = Scenarios.MoeDispatch8Gpu64Experts.result
    assert d.tokens_per_pair == 64
    assert d.hot_remote_tokens == 672

def test_ring_trace_and_error_feedback():
    t = Scenarios.RingTraceFourGpu.result
    assert t.totals == [10, 26, 18, 34]
    assert t.scatter_reduce[0] == [(3, 17), (0, 3), (1, 13), (2, 11)]
    assert t.held == [{0, 1, 2, 3}] * 4
    e = Scenarios.ErrorFeedbackFiveSteps.result
    assert e.sent == [0, 1, 0, 0, 1]
    assert e.naive_total_sent == 0

def test_compression_and_overlap():
    p = Scenarios.CompressionPayback8x
    assert p.pays(p.communication_time) and (not p.pays(p.fast_communication_time))
    c = Scenarios.Int8CompressionRun.result
    o = Scenarios.OverlapBudget7B64Gpu.result
    assert o.buckets_per_layer == 9

def test_parallel_layouts_175b():
    a = Scenarios.Gpt3DataParallel1K.result
    assert a.n_devices == 1024
    assert a.min_devices_for_state == 35
    b = Scenarios.Gpt3TensorDataParallel1K.result
    c = Scenarios.Gpt3ThreeDParallel1K.result
    d = Scenarios.Gpt3ThreeDParallel16K.result
    assert d.n_devices == 16384

def test_gpt2_data_parallel_scale_out():
    one = Scenarios.Gpt2DataParallelOneNode.result
    four = Scenarios.Gpt2DataParallelFourNodes10GbE.result
    acc = Scenarios.Gpt2GradientAccumulationOneNode.result
    assert acc.global_batch == 512

def test_pipeline_tensor_parallel_and_ring_attention():
    g = Scenarios.GPipeFourStageWalkthrough
    assert (g.slots, g.idle_slots) == (14, 6)
    t = Scenarios.Gpt3TensorParallelLayerTp8.result
    r = Scenarios.H100RingAttentionChunk.min_chunk_tokens

def test_alignment_memory_and_fleet_facts():
    a = Scenarios.Llama2_70B_PpoAlignment.result
    assert a.min_devices_for_ppo == 19
    m = Scenarios.MLPerfLlmScalingSeries.result

def test_memory_wall_residual_share_and_recompute():
    w = Scenarios.NotableModelMemoryWall.result
    assert (w.crossover_name, w.crossover_year) == ('GPT-3', 2020)
    r = Scenarios.ResidualOnePercent512
    a = Scenarios.Gpt3ActivationsMicrobatch4

def test_input_space_and_gpt3_cluster_time():
    i = Scenarios.ImageNetInputSpace
    assert i.elements == 150528
    assert i.digits == 362508
    assert i.test_examples == 50000

def test_measured_silent_degradation():
    r = Scenarios.ImageNetV2Shift.result
    assert r.model_count == 67
    assert r.baseline_count == 3

def test_battery_heat_and_energy():
    b = Scenarios.AlwaysOnObjectDetectorPhone
    t = Scenarios.UnoptimizedVisionModelPhone
    assert t.headroom.to('W').magnitude == pytest.approx(0.0, abs=1e-09)
    e = Scenarios.EnergyPerInferenceTiers
    o = Scenarios.RadioUploadVsMobileNetV2

def test_doorbell_braking_voice_and_pipeline():
    f = Scenarios.SmartDoorbellFrame.result
    assert Scenarios.SmartDoorbell.evaluate().performance.status == 'PASS'
    assert Scenarios.SmartDoorbellResponsiveChime.evaluate().performance.status == 'FAIL'
    v = Scenarios.VoiceAssistantBillionDevices
    p = Scenarios.PhoneCameraPipeline

def test_braking_ceilings_and_hub_drive():
    r = Scenarios.WarehouseBaseBrakingCeilings.result
    assert r.a_tip is None and r.passes is False
    e = Scenarios.WarehouseBaseHubDriveElectrical.result

def test_full_system_failure_cascade():
    r = Scenarios.FullSystemFailures10K.result
    assert r.component_rates['power'] == 0

def test_straggler_replicas_probe_sdc():
    s = Scenarios.HalfSpeedStraggler1K
    early, late = Scenarios.InferenceReadinessProbe.detection_window
    assert (early.to('s').magnitude, late.to('s').magnitude) == (12, 17)
    a = Scenarios.SdcExposure10K
    b = Scenarios.SdcExposure100K

def test_kalos_job_size_mix():
    r = Scenarios.KalosJobSizeMix.result
    assert r.job_count == 19907

def test_drf_and_packing():
    d = Scenarios.DrfTwoTenantExample.result
    assert d['bundle_counts'] == {'A': 4, 'B': 2}

def test_idle_capacity_and_policy_gain():
    assert Scenarios.PartialGangHold.missing_gpus == 124
    u = Scenarios.MixedFleetPolicyFix1K

def test_elastic_sharing_and_lending():
    e = Scenarios.ElasticResizeDecision
    c = Scenarios.ServingConsolidationA100
    assert (c.exclusive_gpus, c.shared_gpus) == (180, 130)
    assert c.min_devices_large == 5
    d = Scenarios.DiurnalServingLending1K
    assert (d.peak_demand, d.lent, d.static_active, d.dynamic_active, d.headroom) == (283, 133, 768, 901, 6)
    t = Scenarios.TensorParallelExchange1GB.result.times

def test_quota_replica_and_saturated_queue():
    q = Scenarios.TeamQuotaLending500
    assert (q.used, q.lendable) == (300, 200)
    r = Scenarios.Gpt3SelfContainedReplicaDgxH100
    assert (r.tensor_parallel, r.pipeline_depth, r.replica_gpus) == (8, 8, 64)

def test_screener_requirements_and_late_memory_limit():
    r = Scenarios.RetinopathyScreenerRequirements
    assert r.passes('all clinic photos')
    assert r.failing_slices == ['cataract patients']
    late = Scenarios.RetinopathyScreenerLateMemoryLimit

def test_clinic_upload_and_cloud_versus_edge():
    u = Scenarios.RuralClinicRetinalUpload
    c = Scenarios.RetinopathyScreeningCloudVersusEdge
    assert c.annual_images == 9125000

def test_probe_and_allocation_axes():
    h = Scenarios.ProbeVsGeneratedExplanation.result
    a = Scenarios.OutputCeilingThreeAllocations.result
    assert (a.depth_input, a.breadth_input, a.feedback_input) == (4096, 16384, 16380)
    assert (a.depth_output, a.breadth_output, a.feedback_output) == (4096, 4096, 3072)
    assert a.breadth_premium == 12288 and a.grown_premium == 98304

def test_pools_estimator_and_coverage():
    p = Scenarios.IndependentCandidatePools.result
    c = Scenarios.RepeatedSamplingCoverageVsVote.result
    assert c.vote_gain_after_knee < 0.01 < 0.05 < c.coverage_gain_small

def test_verified_pool_and_argmax_noise():
    v = Scenarios.SparseValidPoolFirstPass.result
    s = Scenarios.SparseValidPoolFirstPass
    post = Scenarios.PatchCheckerPosterior.variant(defect_rate=1 - s.valid_rate, false_accept_rate=s.false_accept_rate, valid_accept_rate=1 - s.false_reject_rate).result.posterior
    g = Scenarios.ArgmaxScoreNoise.result
    assert abs(g.expected_max[1]) < 1e-06

def test_revision_tree_beam_policies_and_run_cost():
    r = Scenarios.ScopedPlanRevision.result
    assert (r.full_output, r.scoped_output, r.avoided_share) == (1800, 450, 0.75)
    t = Scenarios.StepCheckedSearchTree.result
    b = Scenarios.FlatVsBeamSearch.result
    assert b.frontier_counts == [3, 9, 12, 12] and (b.flat_waves, b.beam_waves) == (2, 7)
    assert (b.beam_output_tokens, b.beam_prompt_tokens, b.scorer_tokens) == (4608, 156288, 13440)
    d = Scenarios.DeadlineBoundRepairPolicies.result
    assert d.rounds == 2

def test_repair_context_budget():
    r = Scenarios.RepairContextBudget.result
    assert r.input_limit == 120000 and r.fixed_tokens == 39000
    assert r.observation_budget == 81000
    assert r.turns_fit == 20 and r.overflow_turn == 21 and (r.overflow_observations == 84000)
    assert r.digest_turns == 40 and r.max_observation_per_turn == 2700
    assert r.selected_request == 47000
    assert r.append_bill == 3030000 and r.select_bill == 1410000
    assert r.append_cumulative[-1] == r.append_bill and r.select_cumulative[-1] == r.select_bill
    assert r.final_append_request == 159000 and r.final_append_with_reserve == 167000

def test_toy_head_attention_dilution():
    r = Scenarios.ToyHeadAttentionDilution.result
    assert r.weight_with_lookalikes is None
    x = Scenarios.ToyHeadLookalikeExcerpt.result

def test_fiction_livebench_through_60k():
    r = Scenarios.FictionLiveBenchThrough60K.result
    assert r.scored_models == 53 and r.held_count == 3

def test_repair_prefix_layout_bill():
    r = Scenarios.RepairPrefixLayoutBill.result
    assert r.request_tokens == 47000 and r.total_input == 1410000 and (r.warm_calls == 27)
    assert r.uncached_tokens == 327000 and r.cached_tokens == 1083000

def test_repair_compaction_watermarks():
    r = Scenarios.RepairCompactionWatermarks.result
    assert r.cycle_turns == 12 and r.trigger_length == 95000 and (r.overshoot == 3000)
    assert r.length_with_reserve == 103000 and r.fits_window
    assert r.passes == 2 == sum((1 for x in r.lengths if x >= Scenarios.RepairCompactionWatermarks.trigger))

def test_superseded_excerpt_invalidation():
    r = Scenarios.SupersededExcerptInvalidation.result
    assert r.removed_total == 46150 and r.removed_suffix == 31150
    assert r.replaced_total == 46174 and r.replaced_suffix == 31174
    assert r.appended_total == 47128 and r.appended_suffix == 128

def test_compaction_constraint_retention():
    r = Scenarios.CompactionConstraintRetention.result
    assert r.first_horizon_below_target == 14
    assert Scenarios.UncheckedRepairHorizon.result.first_horizon_below_target == 11

def test_protected_field_check_payback():
    r = Scenarios.ProtectedFieldCheckPayback.result
    assert r.exposures == 36
    assert r.checked_rounds_below_floor == 20

def test_intent_model_evaluations():
    from mlsysim import Embodied
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    f = Scenarios.IntentModelEvaluationFp16.result
    assert f.prefill_bound == 'compute'
    i = Scenarios.IntentModelEvaluationInt4.result
    assert i.t_total == rm.intent_inference_latency
    assert i.t_prefill == f.t_prefill
    five = Scenarios.IntentModelFiveHertz
    assert not five.fits

def test_three_view_tokens_and_kv_history():
    from mlsysim import Embodied
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    t = Scenarios.ThreeViewVlaTokens
    assert t.patches_per_view == 256 and t.visual_tokens == 768 and (t.total_tokens == 816)
    assert t.single_view_tokens == rm.intent_input_tokens
    k = Scenarios.ThreeViewKvHistory
    assert k.frames == 50

def test_decoder_families_and_proposer_traffic():
    d = Scenarios.IntentBackboneDecoderFamilies.result
    assert d.binned_tokens == 112
    assert d.t_expert_step_bandwidth > d.t_expert_step_compute
    t = Scenarios.WarehouseProposerTraffic.result

def test_conveyor_evidence_horizon():
    e = Scenarios.ConveyorMugEvidenceHorizon.result
    assert e.min_supply_setpoints == 4
    half = Scenarios.ConveyorMugEvidenceHorizonHalfRenewal.result
    assert half.min_supply_setpoints == 6

def test_labeling_cost_granularity_and_effort():
    r = Scenarios.ImageNetLabelingVersusResNetRun
    g = Scenarios.FullHdMaskVersusBoxes
    assert g.mask_entries == 2073600
    h = Scenarios.MswcByHand

def test_resnet50_a100_input_feed():
    from mlsysim.core.units import Q_
    f = Scenarios.ResNet50A100InputFeed
    assert f.workers_per_accelerator == 9
    e = Scenarios.ResNet50EightA100InputFeed
    assert e.workers_pooled == 69
    r = Scenarios.RecommendationFeatureLookups

def test_loudness_drift_statistics():
    k = Scenarios.LoudnessDriftKs1000
    assert k.flags

def test_llama2_70b_runtime_goodput_128():
    sc = Scenarios.Llama2_70BRuntimeGoodput128
    r = sc.result
    assert sc.crossover(100) == 12928

def test_shared_pool_goodput_and_llama3():
    g = Scenarios.Llama2_70BSharedPoolGoodput128.result
    assert g.point_value['P'] > g.point_value['S'] > g.point_value['R']
    l3 = Scenarios.Llama3_405BGoodput16K.result

def test_decode_floor_layernorm_and_tp_exchanges():
    d = Scenarios.Llama2_70BDecodeFloorTp8.result
    ln = Scenarios.LayerNormMeasuredH100
    t = Scenarios.Llama2_70BNodeTensorParallelStep.result
    assert t.collectives_per_step == 320
    m = Scenarios.Llama2_70BDecodeAllReduceTp8

def test_compilation_payback_trace_volume_and_software_gains():
    c = Scenarios.Decode13BCompilationPayback
    sg = Scenarios.MlperfInferenceSoftwareGainsDgxH100
    assert sg.months() == 19

def test_serving_cost_goodput_tax_and_knees():
    c = Scenarios.LlmServingFirstYear
    assert round(c.crossover_weeks) == 6
    t = Scenarios.InteractiveRequestServingTax

def test_littles_law_vision_and_chat():
    v = Scenarios.VisionServingLittlesLaw
    assert v.throughput_floor == 10
    assert v.replicas_needed == 13

def test_kv_admission_in_flight_and_handoff():
    long = Scenarios.Llama3_70BLongContextAdmissionDgxH100.result
    assert long.kv_per_token.to('byte').magnitude == 327680
    assert long.head_dim == 128
    assert long.max_requests == 12

def test_prefix_sharing_and_precision_dividend():
    p = Scenarios.ChatbotSharedSystemPrompt
    g = Scenarios.Llama2_70BPrecisionDividendGqaTp8.result
    assert (g.max_batch, g.max_batch_low) == (372, 901)
    m = Scenarios.Llama2_70BPrecisionDividendMhaTp8.result
    assert (m.max_batch, m.max_batch_low) == (46, 112)

def test_chunk_tp_decode_budget_and_speculation():
    assert Scenarios.LongPromptChunkedPrefill.iterations == 20
    t = Scenarios.Llama2_70BPrefillTensorParallel8
    b = Scenarios.Llama3_70B_ReplicaDecodeBudget
    assert b.allreduces_per_step == 160
    s = Scenarios.Llama3_70B_ReplicaSpeculation

def test_fleet_routing_tail_cold_start_and_tenants():
    h = Scenarios.HeterogeneousH100A100Routing
    f = Scenarios.ServingFanout100Leaves
    c = Scenarios.Llama70BReplicaColdStart
    n = Scenarios.SharedPoolNoisyNeighbor
    assert n.other_sequences == 75
    b = Scenarios.FourRequestDecodeBatch
    assert (b.slot_iterations, b.useful_iterations, b.requests_advanced) == (800, 500, 7)
    assert b.total_tokens == [150, 280, 220, 240]

def test_inferencex_frontier_and_token_cost():
    x = Scenarios.InferenceXServingFrontier
    assert x.config_count == 325
    b = x.solve('b200', 100)
    assert x.max_concurrency_at('b200', 50, framework='sglang') == 64

def test_retrieval_records_in_allowance():
    r = Scenarios.RetrievalRecordsInAllowance.result
    assert r.allowance == 6144 and r.record == 576 and (r.records == 10)
    assert r.used == 5888 and r.remaining == 256
    assert r.body_only == 12 and r.body_only_used == 7040 and (r.overshoot == 896) and (r.overcount == 2)
    assert r.scaled_metadata == 128 and r.scaled_metadata_records == 9

def test_typed_traversal_depth_three():
    r = Scenarios.TypedTraversalDepthThree.result
    assert r.nodes_by_depth == [1, 4, 13, 40] and r.tokens_by_depth == [280, 820, 2440, 7300]
    assert r.allowance == 6144 and r.fit == 33 and (r.packed == 6040)
    assert r.shallow_nodes == 13 and r.shallow_required == 2440

def test_repair_edit_index_maintenance():
    r = Scenarios.RepairEditIndexMaintenance.result
    assert r.chunks == 16 and r.batches == 1

def test_lexical_graph_against_dense_store():
    r = Scenarios.LexicalGraphAgainstDenseStore.result
    assert r.chunks == 100000 and r.edited_chunks == 30

def test_correlated_log_and_imitation_bounds():
    e = Scenarios.PermissionTickTelemetryHour
    b = Scenarios.ImitationBoundsFiveSecondEpisode
    assert b.horizon_steps == 500

def test_teleop_ingestion():
    i = Scenarios.WarehouseTeleopIngestion
    assert i.record_bytes == 224

def test_demonstration_campaign():
    c = Scenarios.LatchDemonstrationCampaign
    assert c.attempts_per_session == 52
    assert c.campaign_attempts == 7353
    assert c.gear_cycles == 102942
    assert c.cable_cycles == 58824

def test_hub_counts_and_pooled_corpus():
    h = Scenarios.LeRobotHubDatasetCounts
    assert h.count_at('2024-12') == 1085
    assert h.count_at('2025-12') == 26548
    assert h.latest_count == 80132
    assert h.counted == 59031
    assert h.median_bin_low == 8
    assert h.median_bin_high == 16
    o = Scenarios.OpenXEmbodimentPoolComposition
    assert o.datasets == 72
    assert o.total == 2419193
    assert o.top_datasets == 2

def test_coverage_grids_and_gripper_transfer():
    g = Scenarios.WarehouseManipulatorStateGrid
    assert g.dims == 28
    k = Scenarios.ContactAngleCoverage
    assert k.cells == 600
    assert k.occupied('cautious') == 8
    assert k.occupied('exploratory') == 510
    assert k.bin_span('cautious') == (13, 4, 4, 2)
    t = Scenarios.LatchScenarioTable
    assert (t.cells, t.supported, t.absences) == (6, 1, 5)
    r = Scenarios.LatchScenarioTableAfterRecovery
    assert (r.cells, r.supported, r.absences) == (6, 2, 4)
    gr = Scenarios.GripperStrokeTransfer
    assert gr.donor_clears and (not gr.target_clears)

def test_repair_request_zones():
    z = Scenarios.RepairRequestZones
    assert z.total_tokens == 32768 and z.shared_tokens == 14336

def test_kv_bytes_per_token_by_attention_design():
    r = Scenarios.KvBytesPerTokenByAttentionDesign.result
    assert r.design == {'GPT-3 (175B)': 'MHA', 'Llama-3.1-405B': 'GQA', 'Llama-3.1-70B': 'GQA', 'Llama-2-70B': 'GQA', 'Mixtral-8x7B': 'GQA', 'DeepSeek-V3': 'MLA'}

def test_invariant_prefix_rebuild_dgx_h100():
    r = Scenarios.InvariantPrefixRebuildDgxH100.result
    assert r.suffix_ops_time < r.weight_read_time

def test_repair_tool_wait_paused_state():
    r = Scenarios.RepairToolWaitPausedState.result
    assert r.blocks == 2048 and r.state.to('byte').magnitude == 10737418240

def test_repair_block_sharing_residency():
    r = Scenarios.RepairBlockSharingResidency.result
    assert r.usable_blocks == 90292 and r.unshared_blocks == 2176
    assert r.common_blocks == 1536 and r.private_blocks == 640
    assert r.fit_unshared == 41 and r.fit_shared == 138

def test_repair_and_support_residency():
    r = Scenarios.RepairAgentResidency.result
    assert (r.fit_unshared, r.fit_shared) == (48, 60)
    s = Scenarios.SupportAgentResidency.result
    assert (s.fit_unshared, s.fit_shared) == (195, 780)

def test_permission_path_phase_and_response_time():
    p = Scenarios.WarehouseJointLoopThroughTick.result
    assert not p.fits
    r = Scenarios.WarehousePermissionTickResponse.result
    assert r.meets_deadline
    b = Scenarios.WarehousePermissionTickLongBlocking.result
    assert not b.meets_deadline

def test_servo_fieldbus_occupancy():
    one = Scenarios.WarehouseServoCanFdOneBus.result
    assert (one.bits_nominal, one.bits_data, one.frames) == (34, 113, 18)
    assert not one.fits
    assert one.axes_capacity == 8
    arm = Scenarios.WarehouseServoCanFdArmBus.result
    e = Scenarios.WarehouseServoEtherCat.result

def test_chunk_lease_budget_and_rule():
    r = Scenarios.WarehouseChunkLease.result
    assert r.fits
    long = Scenarios.WarehouseChunkLeaseLengthened.result
    assert long.fits
    stop = Scenarios.WarehouseBaseStopFullDelay.result

def test_twenty_four_megabyte_log_observation():
    r = Scenarios.TwentyFourMegabyteLogObservation.result
    assert r.carried_tokens.to('token').magnitude == 120000

def test_model_polling_intervals():
    r = Scenarios.ModelPollingEveryFiveSeconds.result
    assert r.polls == 59 and r.input_tokens == 1933312 and (r.output_tokens == 2360)

def test_fifty_tool_catalog_six_selected():
    r = Scenarios.FiftyToolCatalogSixSelected.result
    assert (r.full_tokens, r.selected_tokens, r.released_tokens) == (15000, 2200, 12800)

def test_syscall_tax_per_million_calls():
    r = Scenarios.SyscallTaxPerMillionCalls.result
    assert r.microvm_call < r.container_call < r.userspace_call

def test_parameter_efficient_updates():
    lr = Scenarios.TransformerLayerRank4Update
    assert (lr.full_params, lr.update_params) == (589824, 6144)
    ad = Scenarios.CompactEncoderResidualAdapters
    assert (ad.params_per_adapter, ad.params_total) == (25360, 304320)

def test_federated_round_mechanics():
    s = Scenarios.PhoneSocAiScoreSpread
    assert s.lowest('fleet')['processor'] == 'Snapdragon 460'

def test_on_device_learning_literature():
    from mlsysim import Literature
    L = Literature.OnDeviceLearning
    rows = load_series('slides/vol2/11_edge_intelligence/images/data/fedavg_mnist_cnn_rounds.csv').entries
    assert (rows[0]['iid_rounds'], rows[0]['non_iid_rounds'], rows[-1]['iid_rounds']) == (626, 483, 18)

def test_latch_opening_discount():
    d = Scenarios.LatchOpeningDiscount
    assert d.decisions == 320

def test_latch_decoder_budget():
    b = Scenarios.LatchDecoderBudget
    assert (b.evaluations_within_step, b.evaluations_within_renewal) == (6, 14)

def test_latch_trial_campaign_and_parallel_sim():
    r = Scenarios.LeggedGymQuadrupedRun
    assert (r.batch, r.samples) == (98304, 147456000)
    c = Scenarios.LatchTrialCampaign
    assert c.reversals == 4000000
    assert (c.seeded_trials, c.seeded_reversals) == (750, 3000)
    assert c.simulated_steps == 320000000
    assert c.simulation_speedup > 1000

def test_linear_drive_gap_fine_tune_and_dactyl():
    g = Scenarios.LinearDriveRealityGap
    f = Scenarios.LatchResidualFineTune
    assert f.exceed_samples == 300

def test_mnist_dense_layers():
    mlp = Scenarios.MnistMlpLayerStack
    assert mlp.widths == [784, 128, 64, 10]
    assert mlp.weights == 109184 and mlp.macs_per_example_dense == 109184
    l1 = Scenarios.MnistMlpFirstLayer
    assert l1.weights == 100352 and l1.parameters == 100480
    wide = Scenarios.Dense2048Layer

def test_mnist_conv_against_dense():
    first = Scenarios.MnistFirstConvLayer
    assert first.dense_weights == 25088 and first.weights == 288
    cnn = Scenarios.MnistSmallCnn

def test_imagenet_conv_layer():
    rgb = Scenarios.ImageNetRgbFilterVersusDense
    c = Scenarios.ImageNet64ChannelConv
    assert c.weights == 36864 and c.dense_weights == 205520896
    assert c.positions == 50176 and c.macs == 1849688064 and (c.outputs == 3211264)
    assert c.variant(in_channels=1).weights == 576
    assert (c.depthwise_weights, c.pointwise_weights, c.separable_weights) == (576, 4096, 4672)
    assert c.separable_limit == 9 and c.max_copies == 9

def test_mnist_dense_network_counts():
    r = Scenarios.MnistDenseNetwork.result
    assert r.layer_widths == [784, 128, 64, 10] and r.layers == 3
    assert r.weights_per_layer == [100352, 8192, 640] and r.biases == 202
    assert r.macs_per_example == 109184 and r.flops_per_example == 218368
    assert r.parameters == 109386
    assert r.parameter_bytes.to('byte').magnitude == 437544
    assert r.matmul_flops_per_layer == [6422528, 524288, 40960]
    assert r.elementwise_flops == 12928 and r.forward_flops == 7000704
    assert r.traffic_bytes.to('byte').magnitude == 588328
    assert r.input_gradient_flops == 565248 and r.training_step_flops == 14540800
    assert r.steps_per_epoch == 1875

def test_mnist_dense_network_variants():
    b1 = Scenarios.MnistDenseNetwork.variant(batch_size=1).result
    assert b1.traffic_bytes.to('byte').magnitude == 442256
    wide = Scenarios.MnistDenseNetworkWiderHidden.result
    assert wide.layer_widths == [784, 256, 128, 10] and wide.forward_flops == 15049344
    assert Scenarios.MnistDenseNetworkDoubledInputSide.result.weights_per_layer[0] == 401408

def test_mnist_epoch_cpu_versus_h100():
    r = Scenarios.MnistEpochCpuVersusH100.result
    assert r.steps == 1875 and r.ops_per_step == 12

def test_digit_paradigms_and_activation_cost():
    p = Scenarios.DigitRecognitionParadigms
    assert p.feature_count == 441 and p.rule_memory.to('byte').magnitude == 784

def test_mnist_large_batch_saved_activations():
    big = Scenarios.MnistDenseNetworkLargeBatch.result
    small = Scenarios.MnistDenseNetwork.result
    assert big.batch_size == 1024

def test_operational_load_sketch_crosses_capacity_near_fifty_models():
    o = Scenarios.OperationalLoadSketch
    assert o.crossing_models == 51

def test_feature_history_and_lookups():
    h = Scenarios.RecommendationFeatureHistory
    assert h.values == 876000000000000

def test_lifecycle_tco_payback_and_platform_benefit():
    s, p = (Scenarios.StartupOneModelTco, Scenarios.FiftyModelCompanyTco)
    assert round(100 * s.shares['iteration']) == 84 and round(100 * p.shares['inference']) == 41

def test_llama2_70b_reference_replica_decode():
    d = Scenarios.Llama2_70B_DecodeH100
    assert d.workload.name == 'Llama-2-70B'

def test_resnet50_macs_from_architecture():
    from mlsysim.physics.layers import calc_resnet_bottleneck_macs
    v1 = calc_resnet_bottleneck_macs()
    assert v1['3x3'] == 1849688064
    assert v1['total'] == 3857973248

def test_packed_qkv_projection():
    p = Scenarios.Gpt2SmallSixTokenQkvProjection
    assert (p.width, p.packed_width, p.workload.head_width) == (768, 2304, 64)
    assert p.macs == 10616832

def test_gpt2_xl_block_counts():
    b = Scenarios.Gpt2XlBlockFullWindow
    assert (b.weights_per_block, b.block_coefficient) == (30720000, 12)
    assert b.block_weights == 1474560000 and b.token_table == 80411200
    assert b.attention_weights == 491520000 and b.mlp_weights == 983040000
    assert b.exact_parameters == 1557611200
    assert (b.dense_macs, b.attention_macs) == (31457280000, 3355443200)

def test_materialized_scores_and_growth():
    a = Scenarios.TwelveHeadScoresAt100kTokens
    assert a.scores_per_layer == 120000000000
    assert Scenarios.BertWindowToFourK.score_ratio == 64 and Scenarios.BertWindowToTwoK.score_ratio == 16

def test_kv_footprints():
    assert Scenarios.Gpt2XlKvFullWindow.per_token.to('byte').magnitude == 307200
    assert Models.Language.PaLM_540B.get_kv_cache_size(1, 1).to('byte').magnitude == 120832
    assert Models.Language.Llama2_70B.queries_per_kv_head == 8

def test_recommender_gather_and_tables():
    g = Scenarios.CriteoShapedRequestOnH100
    assert g.request_bytes.to('byte').magnitude == 13312 and (g.pairs, g.interaction_macs) == (351, 44928)
    e = Scenarios.HundredMillionItemTableOnA100

def test_framework_dispatch_and_fusion():
    g = Scenarios.Gpt2XlEagerDecodeDispatch
    assert g.operations == 576 and g.launches == 576
    small = Scenarios.SmallMlpDispatch

def test_framework_mlp_training_step_and_paths():
    m = Scenarios.MnistTwoLayerTrainingStepOnH100
    assert m.parameters == 101770
    assert m.host.operations == 5 and m.host.launches == 9

def test_framework_batch512_activation_scaling():
    b = Scenarios.ResNet50Fp32AdamBatch512
    hi = Scenarios.ResNet50Fp32AdamBatch64HighActivations.activations_at(b.batch_size)
    assert b.training_total < b.hardware.memory.capacity < hi

def test_differential_privacy_scenarios():
    s = Scenarios.SalaryMeanLaplaceRelease
    m = Scenarios.MembershipAuditAtEpsilonOne
    assert all((m.audit(i / 200) <= m.ceiling(i / 200) + 1e-12 for i in range(201)))
    c = Scenarios.Cifar10DpSgdRun

def test_robust_ai_drift_and_silent_errors():
    m = Scenarios.FeatureMeanShiftTest
    assert m.p_value < m.significance_bound
    p = Scenarios.FraudCountryPsi
    assert p.shares('Other') == (0.1, 0.18)
    d = Scenarios.FraudAmountDrift
    k = Scenarios.FraudAmountKs

def test_robust_ai_answering_attacks_and_faults():
    assert Scenarios.PgdSevenTraining.passes == 8 and Scenarios.PgdSevenTraining.slowdown == 8
    r = Scenarios.SmoothingRadiusCifar10
    f = Scenarios.Fp32WeightExponentFlip
    assert f.flipped_bit == 30 and f.word_bits == 32 and (f.scale_log2 == 128)
    t = Scenarios.ImageNetTestbedV2Shift.result
    assert t.model_count == 212 and t.baseline_count == 0
    b = Scenarios.RobustBenchCifar10LinfProgress

def test_fixture_release_policy_comparison():
    r = Scenarios.FixtureReleasePolicyComparison.result
    assert r.lost_benign == r.extra_false_blocks == 10

def test_clean_attack_suites():
    r50 = Scenarios.CleanAttackSuite50.result
    assert r50.trials_for_target == 299
    r100 = Scenarios.CleanAttackSuite100.result

def test_bounded_liability_admission():
    r = Scenarios.RepairNextCallAdmission.result
    assert r.spend_check_admits and (not r.admitted)

def test_trajectory_budget_meter():
    r = Scenarios.RepairTrajectoryBudgetMeter.result
    assert r.input_tokens == 892500 and r.cached_tokens == 841000 and (r.total_tokens == 898500)
    assert r.leading_with_reuse == 'turns' and r.leading_without_reuse == 'dollars'

def test_call_and_step_ceilings_from_public_runs():
    r = Scenarios.SweBenchCallCeilingP90.result
    assert r.runs == 2100 and len(r.configurations) == 5 and r.longer_when_failed
    s = Scenarios.OSWorldStepCeilingReturn.result
    assert s.steps == [15, 50, 100] and s.flat_after_middle == 3

def test_latch_evaluation_campaign():
    c = Scenarios.LatchEvaluationCampaign
    assert [c.runs_for(t) for t in c.nine_targets] == [29, 299, 2995]
    assert c.target_runs == 2995 and c.joint_runs == 3688
    assert (c.table.cells, c.table.supported, c.table.absences) == (6, 1, 5)

def test_latch_sequential_test_and_comparison():
    s = Scenarios.LatchSequentialTest
    a = Scenarios.LatchCheckpointComparison
    assert a.runs_per_arm == 195 and a.runs_per_arm_for(0.2) == 393

def test_road_exposure_and_crash_rate_denominator():
    r = Scenarios.RoadMilesToDemonstrate
    assert sorted(r.cleared) == [1.6, 3.9, 51.0]
    assert r.requirements('crashes') == (1.6, 51.0, 65.0)
    d = Scenarios.AutosteerCrashRateDenominator

def test_closed_loop_evaluation_statistics():
    from mlsysim.physics import calc_clopper_pearson_lower_bound, calc_clopper_pearson_zero_failure_bound, calc_poisson_exact_interval, calc_fisher_exact_two_sided
    assert calc_clopper_pearson_lower_bound(0, 20) == 0.0
    lo, hi = calc_poisson_exact_interval(0)

def test_sensor_perception_capture_and_age():
    a = Scenarios.WarehouseNavCameraAge.result
    assert [round(st[2].to('ms').magnitude, 1) for st in a.stages] == [8.0, 24.6, 25.4, 26.6, 29.1, 51.1, 51.6]
    assert round(100 * a.capture_share) == 48 and round(100 * a.backbone_share) == 43
    r = Scenarios.WarehouseNavCameraRollingShutter.result
    e = Scenarios.WarehouseRackEndTimingAttitude.result
    m = Scenarios.WarehouseLongAisleMountError.result
    l = Scenarios.WarehouseRackEndLidarAge.result
    c = Scenarios.WarehouseLidarClockConversion.result

def test_sensor_perception_error_scales():
    s = Scenarios.WarehouseStereoRigDepthError.result
    p = Scenarios.VisualServoDelayPhaseMargin.result
    k = Scenarios.ClearanceInsetConformalRank.result
    assert (k.min_count, k.rank, k.rank_from_top) == (99, 991, 10)

def test_gpt2_xl_training_memory():
    a = Scenarios.Gpt2XlBatch32Activations
    m = Scenarios.Gpt2MixedPrecisionAdamOnH100
    assert m.bytes_per_trained_parameter.to('byte').magnitude == 16
    f = Scenarios.Gpt2Fp32AdamBatch32OnH100
    u = Scenarios.Gpt2Fp32AdamOnH100
    assert u.update_bytes_per_parameter.to('byte').magnitude == 28

def test_responsible_ai_gap_evidence():
    t = Scenarios.LoanApprovalGapTest
    assert t.z == pytest.approx(4.264, abs=0.001) and t.z_at(10) == pytest.approx(1.3484, abs=0.001)
    lo, hi = t.interval
    assert (round(lo, 2), round(hi, 2)) == (0.17, 0.43)
    slo, shi = t.interval_at(10)
    assert (round(slo, 2), round(shi, 2)) == (-0.12, 0.72)
    assert t.n_at_critical == pytest.approx(21.128, abs=0.001)
    tax = Scenarios.CreditParityAccuracyCost
    m = Scenarios.FairnessMonitor10K

def test_responsible_ai_safeguard_costs():
    r = Scenarios.InlineShapRetrofit
    x = Scenarios.ImageExplanationLog
    u = Scenarios.Gpt3ShardedUnlearning
    assert u.full_retrain_time.to('day').magnitude == pytest.approx(35.548, abs=0.001) and u.planning_days == 36
    p = Scenarios.Gpt3PrivacyPlan
    b = Scenarios.RewardModelLabelBudget
    a = Scenarios.RadiologyAutomationBias

def test_responsible_ai_literature_series():
    from mlsysim import Literature
    dp = load_series('vol2/13_security_privacy/data/abadi_dpsgd_accuracy.csv').entries
    cifar8 = next((r for r in dp if r['dataset'] == 'CIFAR-10' and r['epsilon'] == 8))
    assert cifar8['accuracy_pct'] == Literature.DifferentialPrivacy.Cifar10AbadiAccuracyAtEps8Pct
    assert len(load_series('slides/vol2/16_responsible_ai/images/data/ofqual_2020_grade_matrix.csv').entries) == 7

def test_durable_execution_checkpoint_cadence():
    t = Scenarios.TwoCrashCheckpointTrace.result
    assert t.lost_turns == {10: (7, 3), 25: (12, 8), 50: (37, 33)} and t.best_stride == 25
    over = Scenarios.TwoCrashCheckpointTrace.stride_overhead(10, Scenarios.TwoCrashCheckpointTrace.rebuild_per_turn)
    y = Scenarios.ReexecutedWorkCheckpointInterval.result

def test_durable_execution_flat_optimum():
    y = Scenarios.ReexecutedWorkCheckpointInterval
    assert y.miss_factors == [2, 10]

def test_failure_recovery_repair_and_pivot():
    r = Scenarios.TestConfigRepairVsRebuild.result
    assert r.one_in == 16
    p = Scenarios.RemediationCanaryPivotLoss.result

def test_failure_recovery_loop_and_containment():
    from mlsysim.physics import calc_cleanup_critical_path, calc_layered_retry_attempts
    lp = Scenarios.RepairLoopToTurnCeiling.result
    assert (lp.ceiling_input, lp.stop_input, lp.peak_request) == (1397500, 10600, 55200)
    assert lp.input_by_turn[50] == lp.ceiling_input and lp.input_by_turn[4] == lp.stop_input
    t = Scenarios.ThreeLayerRetryAmplification.result
    assert t.by_layer == [1, 3, 9, 27] and t.coordinated == 3 and (calc_layered_retry_attempts(2, 3) == 8)
    b = Scenarios.BlockingCallBulkhead.result
    assert (b.uncapped_free, b.queued, b.rejected, b.protected_workers) == (44, 32, 160, 236)
    j = Scenarios.RetryCohortJitter.result
    assert calc_cleanup_critical_path({'a': 1, 'b': 2, 'c': 3}, {'c': ['a', 'b']}) == 5

def test_spatial_memory_estimation():
    f = Scenarios.WarehouseRackEndRangeFusion.result
    c = Scenarios.RackEndMisdatedUpdateConsistency.result
    assert Scenarios.WarehouseObservationReplayBuffer.result.slots == 52
    i = Scenarios.WarehouseImuOneSecondCoast.result
    o = Scenarios.WarehouseUmbmarkOdometryDrift.result
    k = Scenarios.WarehouseKittiOdometryDrift.result
    assert (k.count, k.best_pct, k.laser_median_pct, k.count_below_threshold) == (127, 0.53, 0.68, 0)

def test_spatial_memory_maps():
    q = Scenarios.WarehousePermissionClearanceQuery.result
    assert [round(q.rows[p].occupancy_reads) for p in (2, 5, 10)] == [22449, 1437, 180]
    assert q.rows[5].octree_depth == 8 and round(q.rows[5].octree_reads) == 11494
    h = Scenarios.WorkcellHostMapSizing.result
    r = Scenarios.WarehouseDepthRaymarchTraffic.result

def test_conveyor_mug_intent_cadence():
    c = Scenarios.ConveyorMugIntentCadence.result
    assert c.renewal_age == c.intent_period + Scenarios.ConveyorMugIntentCadence.machine.intent_inference_latency

def test_conveyor_mug_reach_occupancy_and_lease_expiry():
    r = Scenarios.ConveyorMugDynamicReach.result
    assert not r.fits and r.fresh_captures == 2
    o = Scenarios.AisleCoworkerForwardOccupancy.result
    e = Scenarios.ConveyorMugIntentLeaseExpiry.result
    assert (e.binding_untracked, e.binding_tracked) == ('evidence', 'administrative')

def test_referent_sets_confidence_region_and_latch_stall():
    s = Scenarios.MugJarReferentSet.result
    assert s.set_size == 2 and s.refuse
    low = Scenarios.MugJarReferentSetLowJar.result
    g = Scenarios.DistantHandleConfidenceRegion.result
    assert g.axes_within == 2
    x = Scenarios.LatchUnbrakedStall
    gs = Scenarios.ConveyorMugGoalSetOffsets

def test_carbon_priced_region_placement():
    r = Scenarios.CarbonPricedRegionPlacement.result
    assert r.best_region == 'Quebec' and r.best_region_untaxed == 'Quebec'

def test_cluster_expansion_doubling():
    e = Scenarios.Gpt3ClusterDoubling1K
    assert e.expanded_devices == 2048

def test_paired_comparison_and_published_newcombe_examples():
    from mlsysim.physics import calc_newcombe_paired_difference, calc_mcnemar_exact_two_sided
    r = Scenarios.RepairCandidatePairedComparison.result
    assert (r.pairs_if_gain_holds, r.pairs_if_no_gain) == (217, 1948)
    z = 1.959963984540054
    _, lo, hi = calc_newcombe_paired_difference(20, 12, 2, 16, z)
    assert lo == pytest.approx(0.0562, abs=5e-05) and hi == pytest.approx(0.3292, abs=5e-05)
    _, lo, hi = calc_newcombe_paired_difference(27, 0, 0, 27, z)
    assert lo == pytest.approx(-0.0351, abs=5e-05) and hi == pytest.approx(0.0351, abs=5e-05)
    assert calc_mcnemar_exact_two_sided(0, 0) == 1.0

def test_leaderboards_and_selection_cost():
    s = Scenarios.TerminalBenchScaffoldSpread.result
    assert s.models == 16 and s.widest == 'claude-opus-4-6' and (s.widest_count == 11)
    t = Scenarios.SweBenchVerifiedRankTies.result
    m = Scenarios.TwoTaskTypeConsistency.result
    c = Scenarios.CappedCandidateSelection.result

def test_repair_trial_span_timeline():
    r = Scenarios.RepairTrialSpanTimeline.result
    assert r.changed_critical == ['retrieve', 'propose', 'lint', 'synthesize']
    lint = Scenarios.RepairTrialSpanTimeline.variant(intervention={'lint': Scenarios.RepairTrialSpanTimeline.intervention['tests']})
    assert lint.result.saving.to('s').magnitude == pytest.approx(0.0, abs=1e-12)

def test_verdict_aware_trace_retention():
    r = Scenarios.VerdictAwareTraceRetention.result
    assert (r.head_count, r.head_failed, r.head_unresolved, r.tail_control, r.tail_count) == (500, 20, 5, 190, 690)

def test_interface_repair_branches_wilson():
    b = Scenarios.InterfaceRepairBranches
    assert b.clean_failure_upper(72) >= 0.05 > b.clean_failure_upper(73)
    lo, hi = b.acceptance_interval('original')
    lo, hi = b.acceptance_interval('warning')

def test_repair_candidate_paired_comparison_as_read_by_operations():
    """The Agent Evaluation comparison that the release decision record cites (shared with Agent Operations)."""
    r = Scenarios.RepairCandidatePairedComparison.result
    assert r.pairs_if_gain_holds == 217

def test_openai_sycophancy_timeline_and_dapper():
    from mlsysim import Literature
    inc = Literature.IncidentReports
    rows = {r['sampling_frequency']: r for r in load_series('slides/vol3/14_agent_operations/images/data/dapper_sampling_overhead.csv').entries}
    assert rows['1/1']['latency_change_pct'] == 16.3 and rows['1/1024']['latency_change_pct'] == -0.2
    assert Literature.DistributedTracing.DapperFirstProductionSamplingDenominator == 1024

def test_gpt2_xl_step_fits_one_h100():
    flash = Scenarios.Gpt2MixedPrecisionFlashAttentionOnH100
    assert flash.max_resident_batch == 23
    ckpt = Scenarios.Gpt2XlBatch32SegmentCheckpointed
    assert ckpt.segment_layers == 6
    act = Scenarios.Gpt2XlBatch32Activations

def test_online_softmax_and_attention_tiles():
    a = Scenarios.AttentionScores4096TokensFp32
    assert a.tile_count == 1024

def test_model_compression_scenarios():
    from mlsysim import Literature
    from mlsysim.physics import calc_csr_bytes, calc_pareto_frontier, calc_sparse_breakeven_zero_share, calc_sparse_bytes_per_nonzero
    from mlsysim.core.units import BYTES_FP32, BYTES_INT32
    t = Scenarios.Llama2_7BTokenOnFlagshipPhone
    b = Scenarios.MobileNetV2BlurPipelinePhone
    m = Scenarios.SmallestModelAtSixtyPercentMmlu
    assert m.first['model'] == 'Gopher 280B' and m.last['model'] == 'Qwen2.5-Coder-3B'
    r = Scenarios.Rank128Factorization4096
    f = Scenarios.ResNet50Conv256FusionH100
    w = Scenarios.WearableMobileNetV2SparseFlash
    csr = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32)
    coo = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32, indices=2)
    assert calc_csr_bytes(6, 4, BYTES_FP32, BYTES_INT32).to('byte').magnitude == 68
    t3 = load_series('slides/vol1/10_model_compression/images/data/mobilenetv3_table3_pixel1.csv').entries
    front = [t3[i]['model'] for i in calc_pareto_frontier([(r['pixel1_ms'], r['top1']) for r in t3])]
    assert 'V2 1.0' not in front and 'V3-Large 0.75' in front

def test_accelerator_trends_and_latency_in_cycles():
    w = Scenarios.DenseParametersVsBandwidthSince2014
    e = Scenarios.AcceleratorEnvelopeSinceTpuV2
    assert e.best_bandwidth_since == '2025-02-15'
    g = Scenarios.NvidiaGenerationsV100ToB200

def test_trajectory_planning_joint_rates_handoffs_and_replacement():
    good = Scenarios.WarehouseArmRightAngleElbow.result
    assert good.exceeds_joint_limit
    bad = Scenarios.WarehouseArmNearStraightElbow.result
    m = Scenarios.GenericReplannerMissionExposure.result
    r = Scenarios.WarehouseArmChunkReplacement.result
    s = Scenarios.WarehouseArmStaleAnchor.result
    w = Scenarios.WarehouseBaseDynamicWindow.result

def test_warehouse_jerk_limited_stop_matches_finished_budget():
    """The trajectory chapter's S-curve stop and the enforcement chapter's finished budget share one jerk bound and one stop."""
    base = Scenarios.WarehouseBaseJerkLimitedStop.result
    budget = Scenarios.WarehouseFinishedStopAtDriveLimit.result
    assert base.jerk.m_as('m/s**3') == pytest.approx(budget.jerk.m_as('m/s**3'))
    assert base.stop_distance.m_as('mm') == pytest.approx(budget.stop_distance.m_as('mm'))
    assert base.jerk_ramp_travel.m_as('mm') == pytest.approx(budget.jerk_ramp_travel.m_as('mm'))

def test_repair_agent_failure_triage():
    r = Scenarios.RepairAgentFailureTriage.result
    assert r.failures == 3000 and r.removed == {'interface': 2100, 'runtime': 600}
    assert r.training_tokens.to('token').magnitude == 36864000

def test_trajectory_minhash_screen():
    m = Scenarios.TrajectoryMinHashScreen
    assert m.signature_bytes.to('byte').magnitude == 512

def test_recovery_pool_denominators():
    r = Scenarios.RecoveryPoolDenominators.result
    assert r.stored_tokens.to('token').magnitude == 514250000
    assert r.admitted_stored_tokens.to('token').magnitude == 465250000
    assert r.target_tokens.to('token').magnitude == 64000000

def test_fine_tuning_serialization_and_target_mask():
    """Per-call examples, the recovery record's mask, and recorded role shares (added 2026-10-20)."""
    s = Scenarios.RepairTrajectoryPerCallExamples.result
    m = Scenarios.RecoveryRecordTargetMask.result
    assert m.stored == 16384 and m.predicted == 16383
    assert m.targets_without_rationale == 800 and m.targets_with_rationale == 2200
    r = Scenarios.SweBenchVerifiedRoleShares.result
    assert r.runs == 2100 and len(r.configurations) == 5

def test_fine_tuning_weighting_packing_and_bounds():
    w = Scenarios.RepairTraceWithLookupWeighting.result
    p = Scenarios.EightExamplePackingBatch.result
    assert p.stored == 50176 and p.padded_positions == 262144 and (p.buffers == 2)
    assert p.buffer_loads == [32768, 17408]
    i = Scenarios.RepairPolicyImitationBounds

def test_placement_shared_dram_read():
    r = Scenarios.WarehouseSharedDieLoadedRead.result
    assert r.queued_transactions == 88 and r.tile_transactions == 64 and (r.backlog_bytes == 360448)
    assert r.backlog_wait.to('us').magnitude == pytest.approx(1.8794, abs=0.0005)
    assert r.snapshot_read.to('us').magnitude == pytest.approx(0.0201, abs=0.0005)
    assert r.bypass_read.to('us').magnitude == pytest.approx(0.0402, abs=0.0005)
    assert (r.backlogs_sense, r.backlogs_wcet, r.backlogs_deadline) == (21, 82, 114)
    assert r.end_deadline.to('us').magnitude == pytest.approx(309.27, abs=0.01)
    one = Scenarios.WarehouseSharedDieLoadedReadOneChannel
    o = one.result
    assert o.backlog_wait.to('us').magnitude == pytest.approx(28.38, abs=0.01)
    assert (o.backlogs_sense, o.backlogs_wcet) == (1, 5)

def test_placement_stall_shares_and_droop():
    assert Scenarios.WarehouseTickMshrStallShare.breakeven_share == pytest.approx(0.042593, abs=1e-06)
    assert Scenarios.WarehouseTickMshrStallShareAtDeadline.breakeven_share == pytest.approx(0.064815, abs=1e-06)
    assert Scenarios.WarehouseTickWritebackStallShare.breakeven_share == pytest.approx(0.002849, abs=1e-06)
    ten = Scenarios.WarehouseTickTenfoldStallShare
    assert ten.breakeven_share == pytest.approx(0.09465, abs=1e-06)
    assert ten.breakeven_stall.to('us').magnitude == pytest.approx(12.778, abs=0.001)
    d = Scenarios.PackageRailDroop
    assert round(d.delay_expansion_percent) == 39

def test_placement_thermal_throttle_and_trace():
    r = Scenarios.WarehouseSharedDieThermalThrottle.result
    assert r.time_to_trip.to('s').magnitude == pytest.approx(24.728, abs=0.001)
    assert r.recovery.to('s').magnitude == pytest.approx(2.1755, abs=0.0001)
    assert r.recovery_from_loaded.to('s').magnitude == pytest.approx(2.5476, abs=0.0001)
    assert (r.ticks_exposed, r.ticks_exposed_from_loaded) == (2175, 2547)
    assert r.theta_never_trips.to('K/W').magnitude == pytest.approx(1.9231, abs=0.0001)
    t = Scenarios.WarehouseRejectedPlacementTrace
    assert t.hours_minutes_seconds == (2, 46, 40)

def test_matmul_tiles_intensity_and_capacity():
    t = Scenarios.Matmul4096Fp16Tile128
    assert (t.output_tiles, t.tiles_k, t.tile_products) == (1024, 32, 32768)

def test_quantization_grids_and_integer_layer():
    """Affine and symmetric grids, the INT8 dot product, and FP8 E4M3's value grid (added 2026-10-20)."""
    from mlsysim.core.units import FP8_FORMAT_BITS
    from mlsysim.physics import calc_float_values, calc_block_bits_per_weight
    g = Scenarios.Uint8RangeMinusOneToThree
    assert g.scale == pytest.approx(4 / 255) and g.zero_point_exact == pytest.approx(63.75)
    assert g.zero_point == 64 and g.codes == [0, 128, 64, 255]
    assert g.readback[0] == pytest.approx(-1.003922, rel=1e-06)
    e4m3 = calc_float_values(*FP8_FORMAT_BITS['e4m3'], ieee=False)
    assert len(e4m3) == 127 and max(e4m3) == 448 and (e4m3[1] == 2 ** (-9))
    assert max(calc_float_values(*FP8_FORMAT_BITS['e5m2'])) == 57344
    assert calc_block_bits_per_weight(32, 4, 16) == 4.5

def test_quantized_decode_energy_and_deployment_profile():
    """GPT-2 XL decode traffic, INT8 operation energy, and the ResNet-50 INT8 profile (added 2026-10-20)."""
    f = Scenarios.Gpt2XlFp16DecodeFullContextPhone
    q = Scenarios.Gpt2XlInt4DecodeFullContextPhone
    assert f.kv_cache_bytes.m_as('GB') == pytest.approx(0.3145728)
    assert f.token_bytes.m_as('GB') == pytest.approx(3.3145728)
    assert f.token_time.m_as('ms') == pytest.approx(64.73775)
    assert q.token_bytes.m_as('GB') == pytest.approx(1.0645728)
    assert q.separate_unpack_bytes.m_as('GB') == pytest.approx(7.0645728)
    assert (f.token_time / q.token_time).m_as('') == pytest.approx(3.113523, rel=1e-06)
    e = Scenarios.Int8VsFp32OperationEnergy45nm
    assert e.add_ratio == pytest.approx(30) and e.mul_ratio == pytest.approx(18.5)
    assert e.mac_ratio == pytest.approx(46 / 3) and e.read_over_narrow_mac == pytest.approx(6400 / 3)
    r = Scenarios.ResNet50Int8ProfileH100
    assert r.top1_drop == pytest.approx(0.3) and r.passes_top1 and (not r.passes_calibration)
    assert r.size_reduction == pytest.approx(4) and r.energy_improvement == pytest.approx(r.throughput_gain)

def test_supervisory_intervention_takeover_budgets():
    """Takeover deadlines, ceilings, the session waterfall, blends, and authentication on the warehouse base (added 2026-10-20)."""
    from mlsysim import Literature
    from mlsysim.physics.robotics import calc_scurve_max_permitted_velocity
    s = Scenarios
    aisle = s.WarehouseInLoopTakeoverAtAisleSpeed.result
    assert aisle.machine.spare.m_as('mm') == pytest.approx(131.0075)
    assert aisle.deadline.m_as('ms') == pytest.approx(100.775)
    assert aisle.deadline_after_admission.m_as('ms') == pytest.approx(37.46428)
    assert aisle.machine.required_distance.m_as('mm') == pytest.approx(s.WarehouseFinishedStopAtAisleSpeed.result.required_distance.m_as('mm'))
    assert aisle.with_takeover.required_distance.m_as('mm') == pytest.approx(1755.4925)
    assert aisle.with_takeover_and_dispatch.required_distance.m_as('mm') == pytest.approx(1837.796436)
    assert aisle.with_takeover.speed_ceiling.m_as('m/s') == pytest.approx(0.836188, abs=1e-06)
    ceil_da = aisle.with_takeover_and_dispatch.speed_ceiling
    assert ceil_da.m_as('m/s') == pytest.approx(s.WarehouseInLoopTakeoverAtAisleSpeed.site.v_takeover_in_loop_ceiling.m_as('m/s'))
    w = aisle.with_takeover
    a = w.deceleration
    root = (a * (w.ceiling_root - w.ceiling_lead_time)).m_as('m/s')
    assert root == pytest.approx(w.speed_ceiling.m_as('m/s'))
    assert w.ceiling_lead_time.m_as('s') == pytest.approx(0.879225)
    assert w.rigid_speed_ceiling.m_as('m/s') == pytest.approx(1.2320607, rel=1e-06)
    assert aisle.timeout_ticks == 100 and aisle.timeout_ticks_after_admission == 37
    assert aisle.takeover_over_spare == pytest.approx(6.0034731) and aisle.takeover_over_spare_after_admission == pytest.approx(16.148716, rel=1e-06)
    ool = s.WarehouseOutOfLoopTakeoverAtAisleSpeed.result
    assert ool.with_takeover_and_dispatch.required_distance.m_as('mm') == pytest.approx(6855.796436)
    assert ool.with_takeover.speed_ceiling.m_as('m/s') == pytest.approx(0.19107, abs=1e-05)
    slow = s.WarehouseInLoopTakeoverAtRequestSpeed.result
    assert slow.deadline.m_as('ms') == pytest.approx(751.608333, rel=1e-08)
    assert slow.spare_after_takeover.m_as('mm') == pytest.approx(109.95625)
    assert slow.spare_after_admission.m_as('mm') == pytest.approx(62.47321)
    assert slow.lease_extension_travel.m_as('mm') == pytest.approx(30.0)
    assert slow.spare_after_lease.m_as('mm') == pytest.approx(32.47321)
    assert s.WarehouseInLoopTakeoverAtRetiredSlowdown.result.time_left_after_admission.m_as('ms') == pytest.approx(-5.036, abs=0.005)
    crawl = s.WarehouseOutOfLoopTakeoverAtCrawl.result
    assert crawl.time_left.m_as('s') == pytest.approx(1.32283, abs=1e-05)
    assert crawl.spare_after_takeover.m_as('mm') == pytest.approx(198.4245, abs=0.0001)
    assert s.WarehouseOutOfLoopTakeoverAtCrawlP99.result.time_left.m_as('s') < 0
    assert s.WarehouseFixedWindowAtAisleSpeed.result.with_takeover.required_distance.m_as('mm') == pytest.approx(1618.9925)
    assert round(s.WarehouseFixedWindowAtCrawl.result.with_takeover.required_distance.m_as('mm')) == 307
    ladder = s.WarehouseTakeoverCeilingLadder.result.rows
    ceilings = [r.with_takeover.speed_ceiling.m_as('m/s') for r in ladder.values()]
    assert all((hi > lo for hi, lo in zip(ceilings, ceilings[1:])))
    assert ladder['mhp_slow'].takeover == s.WarehouseInLoopTakeoverAtAisleSpeed.takeover
    assert ladder['p90'].takeover.m_as('ms') == pytest.approx(s.WarehouseOutOfLoopTakeoverAtAisleSpeed.takeover.m_as('ms'))
    assert ladder['mhp'].with_takeover.speed_ceiling.m_as('m/s') == pytest.approx(1.0094, abs=5e-05)
    site = s.WarehouseInLoopTakeoverAtAisleSpeed.site
    direct = calc_scurve_max_permitted_velocity(site.d_clear, site.tau_delay + ladder['surprise'].takeover, site.a_brake, site.j_brake, site.delta_loc + site.eps_track, site.delta_margin)
    assert direct.m_as('m/s') == pytest.approx(ladder['surprise'].with_takeover.speed_ceiling.m_as('m/s'))
    b = s.WarehouseTakeoverBrakingBlend.result
    assert b.window.m_as('ms') == pytest.approx(527.34375) and b.window_ticks == 528
    assert b.blend_travel.m_as('mm') == pytest.approx(316.0531, abs=0.0001)
    assert b.blended_stop.m_as('mm') == pytest.approx(328.4471, abs=0.0001)
    assert b.step_stop.m_as('mm') == pytest.approx(140.625) and b.scurve_travel.m_as('mm') == pytest.approx(246.09375)
    assert b.reserve_use.m_as('mm') == pytest.approx(156.9703, abs=0.0001)
    assert b.reserve_use > slow.spare_after_takeover
    assert b.arm_window.m_as('ms') == pytest.approx(37.5) and b.arm_window_ticks == 38
    assert b.step_jerk.m_as('m/s**3') == pytest.approx(2000.0)
    auth = s.WarehouseRemoteCommandAuthentication.result
    assert auth.verify_time.m_as('ms') == pytest.approx(1.8) and auth.mac_time.m_as('us') == pytest.approx(4.96275)
    assert auth.forged_to_miss == 55 and auth.forged_to_miss_after_admission == 20 and (auth.forged_rate == 546)
    assert auth.spare_after_freshness.m_as('mm') == pytest.approx(1.0075)
    rec = s.WaymoCaliforniaDisengagementRecord
    assert rec.rate_drop > 3 and rec.miles_growth > 3
    assert rec.latest['disengagements'] == 174
    import math
    vals = sorted((r['mean_takeover_time_s'] for r in load_series('slides/vol4/14_intervention/images/data/zhang2019_mean_takeover_times.csv').entries))
    assert len(vals) == 520 and sum(vals) / len(vals) == pytest.approx(2.72, abs=0.005)
    pct = lambda p: vals[max(math.ceil(p / 100 * len(vals)) - 1, 0)]
    assert pct(50) == pytest.approx(site.t_takeover_driving_p50.m_as('s'))
    assert pct(90) == pytest.approx(site.t_takeover_driving_p90.m_as('s'))
    assert pct(99) == pytest.approx(site.t_takeover_driving_p99.m_as('s'))

def test_rack_end_fault_injection_oracles():
    """Renewal- and flag-referenced oracles, onset trials, drive bus, and rail hold-up (added 2026-10-20)."""
    r = Scenarios.WarehouseRenewalReferencedStop.result
    assert Scenarios.WarehouseRenewalReferencedStop.delay.m_as('ms') == pytest.approx(82.0)
    assert r.required_distance.m_as('mm') == pytest.approx(751.9125)
    assert r.delay_travel.m_as('mm') == pytest.approx(106.6)
    assert r.stop_distance.m_as('mm') == pytest.approx(711.9125)
    rr = Scenarios.WarehouseRenewalReferencedStopRestricted.result
    assert rr.required_distance.m_as('mm') == pytest.approx(629.6253648, rel=1e-08)
    assert Scenarios.WarehouseFinishedStopRestricted.result.required_distance.m_as('mm') == pytest.approx(831.2253648, rel=1e-08)
    assert Scenarios.WarehouseFlagReferencedStop.result.required_distance.m_as('mm') == pytest.approx(673.9125)
    late = Scenarios.WarehouseRenewalStopLateOnsetTrial
    t = late.result
    assert t.onset_margin.m_as('ms') == pytest.approx(-30.0) and t.extra_travel.m_as('mm') == pytest.approx(39.0)
    assert t.travel.m_as('mm') == pytest.approx(750.9125) and t.distance_margin.m_as('mm') == pytest.approx(1.0)
    assert t.onset_margin_normalized == pytest.approx(-30 / 82) and (not t.passes)
    assert t.rest_time_bound.m_as('ms') == pytest.approx(1013.25)
    assert late.probe_at.m_as('ms') == pytest.approx(65.0)
    c = Scenarios.WarehouseRenewalStopNinetyMsOnsetTrial.result
    assert c.onset_margin.m_as('ms') == pytest.approx(-8.0) and c.extra_travel.m_as('mm') == pytest.approx(10.4)
    h = Scenarios.WarehousePermissionRailHoldUp.result
    assert h.flag_path.m_as('ms') == pytest.approx(22.0)
    assert h.stop_time.m_as('ms') == pytest.approx(1211.4328579, rel=1e-08)
    assert h.flag_to_set.m_as('ms') == pytest.approx(1313.4328579, rel=1e-08)
    assert h.margin.m_as('ms') == pytest.approx(686.5671421, rel=1e-08)
    b = Scenarios.WarehouseDriveBusStop.result
    assert b.crossing_resistance.m_as('mohm') == pytest.approx(128.5714286, rel=1e-08)

def test_fault_campaign_statistics():
    """Diagnostic-coverage injections and the queue-delay rare event (added 2026-10-20)."""
    import math
    c = Scenarios.WarehouseDiagnosticCoverageHundredFit.result
    assert c.raw_rate == pytest.approx(9800) and c.coverage == pytest.approx(1 - 100 / 9800)
    assert c.injections == 293 and c.injections_exact == pytest.approx(292.0813362, rel=1e-08)
    assert c.rule_of_three == pytest.approx(294)
    assert round(c.allocation['gate drivers']) == 144 and round(c.allocation['compute']) == 30
    assert Scenarios.WarehouseDiagnosticCoverageTenFit.result.injections == 2935
    q = Scenarios.WarehouseQueueDelaySpareExhaustion
    assert q.failure_threshold.m_as('ms') == pytest.approx(100.775)
    r = q.result
    assert r.failure_probability == pytest.approx(1.7652032e-09, rel=1e-06)
    assert r.n_levels == 4 and r.proposal_mean.m_as('ms') == pytest.approx(105.775)
    assert math.ceil(r.importance_runs) == 2709 and round(r.effective_sample_size) == 250
    assert r.zero_failure_runs == 1697103349 and r.monte_carlo_runs == pytest.approx(56650700000.0, rel=1e-05)
    assert 0 <= r.proposal_penalty < 0.001 and r.proposal_failure_share == pytest.approx(0.3856867, rel=1e-06)
    p = Scenarios.WarehouseQueueDelayChargedSpareExhaustion.result.failure_probability
    assert p == pytest.approx(math.exp(-37.46428 / 5), rel=1e-09)

def test_verification_literature():
    """DriveFI, patch, RoboPAIR, VNN-COMP 2025, O'Kelly, Wang, and Bookout entries (added 2026-10-20)."""
    from mlsysim import Literature
    vnn = load_series('vol4/15_verification/data/vnncomp2025_scored_benchmarks.csv')
    assert len(vnn) == 25 and max((e['params_high'] for e in vnn.entries)) == 138000000.0
    ok = load_series('slides/vol4/15_verification/images/data/okelly2018_rare_event_table1.csv')
    assert len(ok) == 12 and ok.where(method='Cross-entropy', gamma=0.14)[0]['std_error'] == pytest.approx(8.88e-06)
    w = load_series('slides/vol4/15_verification/images/data/wang2025_openvla_libero_untargeted.csv')
    assert sum((e['benign_failure_pct'] for e in w.entries)) / 4 == pytest.approx(23.5)

def test_single_machine_serving_scenarios():
    """ResNet-50 serving on one H100: cost, budget, M/M/1 plan, batching, startup, cost per image (added 2026-10-20)."""
    from mlsysim.physics import calc_exponential_quantile_factor, calc_batch_formation_wait
    from mlsysim.core.units import Q_
    svc = Scenarios.ResNet50EagerFp16BatchingH100
    assert svc.service_time(1).m_as('ms') == pytest.approx(3.528119, rel=1e-06)
    assert svc.throughput(32).m_as('1/s') == pytest.approx(7037.878, rel=1e-06)
    assert svc.fixed_share(1) == pytest.approx(0.99069, rel=0.0001)
    cost = Scenarios.ResNet50BatchOneVersusEightCostH100
    assert cost.cost_per_million(1).m_as('dollar') == pytest.approx(3.920133, rel=1e-06)
    assert cost.cost_ratio == pytest.approx(7.510333, rel=1e-06)
    assert Scenarios.PhotoLibraryPrecomputeH100.ratio == pytest.approx(315.66128, rel=1e-06)
    path = Scenarios.ResNet50RequestPathH100
    assert path.total.m_as('ms') == pytest.approx(8.628119, rel=1e-06)
    assert path.overall_speedup == pytest.approx(1.582325, rel=1e-06)
    assert path.overlapped_share('forward pass', 'JPEG decode', 'resize', 'normalize') == pytest.approx(0.784027, rel=1e-05)
    assert Scenarios.DlrmRequestPath.share('MLP forward') == pytest.approx(1.5 / 9)
    plan = Scenarios.ResNet50FleetPlanH100
    assert plan.max_utilization == pytest.approx(0.675413, rel=1e-05)
    assert (plan.accelerators, plan.accelerators_planned, plan.batched_accelerators_planned) == (27, 36, 3)
    assert plan.tail_time(0.7).m_as('ms') == pytest.approx(54.09783, rel=1e-06)
    assert plan.cost_per_million(36).m_as('dollar') == pytest.approx(8.0)
    assert calc_exponential_quantile_factor(0.99) == pytest.approx(4.60517, rel=1e-06)
    vote = Scenarios.ResNet50FasterModelAt80PctH100
    assert vote.fast_time.m_as('ms') == pytest.approx(3.007545, rel=1e-05)
    win = Scenarios.BatchWindow10ms
    assert win.empty_share(win.arrival_rates[0]) == pytest.approx(0.606531, rel=1e-05)
    assert win.seen_batch_mean(win.arrival_rates[2]) == pytest.approx(6) and win.seen_batch_tail(win.arrival_rates[2]) == 12
    assert Scenarios.BatchWindow20msIn50msBudget.budget_share == pytest.approx(0.2)
    t = Scenarios.DualTriggerBatchFourRequests
    assert t.backlog_wait.m_as('ms') == pytest.approx(6) and t.slack.m_as('ms') == pytest.approx(24.373296, rel=1e-05)
    cap = Scenarios.BatchCap16Or32At500Qps
    assert cap.formation_wait(32).m_as('ms') == pytest.approx(31) and cap.throughput_gain == pytest.approx(0.768725, rel=1e-05)
    assert calc_batch_formation_wait(16, Q_(500, '1/s')).m_as('ms') == pytest.approx(15)
    s = Scenarios.ResNet50ReplicaStartup
    assert s.warm_total.m_as('s') == pytest.approx(1.5) and s.cold_total.m_as('s') == pytest.approx(35)
    assert Scenarios.ModelSwapPcieGen4.transfer_time('PCIe Gen4 x16').m_as('ms') == pytest.approx(312.5)
    e = Scenarios.ResNet50BatchEnergyPerImage
    assert e.energy(1).m_as('mJ') == pytest.approx(39.2165, rel=1e-05) and e.movement_share(32) == pytest.approx(0.193022, rel=1e-05)
    inst = Scenarios.ResNet50CloudInstanceCost
    assert inst.cost_per_million('CPU').m_as('dollar') == pytest.approx(0.944444, rel=1e-05)
    assert inst.small_within_large_price == 7 and inst.breakeven_rate.m_as('1/s') == pytest.approx(3018.868, rel=1e-06)
    assert Scenarios.RecommenderTrafficSurge.surge_factor == pytest.approx(10)
    assert Scenarios.ImageServiceLittlesLaw.in_flight == pytest.approx(50)

def test_verifiable_reward_composition_coefficients_and_contrast():
    """RL from verifiable rewards: false accepts, baselines, shaping, forks, GRPO, contrast (added 2026-10-20)."""
    r = Scenarios.VerifierFalseAcceptsAtTenPercentPass.result
    b = Scenarios.BinaryRewardFixedBaseline.result
    f = Scenarios.DebuggerTurnForkValue.result
    every = Scenarios.ThirtyTurnForksAtEveryBoundary.result
    assert every.base_tokens == 15360 and every.branches == 232 and (every.total_tokens == 1797120)
    two = Scenarios.ThirtyTurnForksAfterTurnsTenAndTwenty.result
    g = Scenarios.RepairGroupOnePassOfThree.result
    lone = Scenarios.LonePassInGroupOfSixteen.result
    h = Scenarios.HardTaskRolloutStep.result

def test_release_scenarios_and_records():
    """Deceleration premise, dim-aisle age, argument confidence, and the release records (added 2026-10-20)."""
    from mlsysim import Embodied, Literature
    stop = Scenarios.WarehouseFinishedStopAtAisleSpeed.result
    assert stop.braking_room.m_as('mm') == pytest.approx(736.32)
    assert stop.min_deceleration.m_as('m/s**2') == pytest.approx(1.3861067, rel=1e-06)
    assert stop.deceleration_shortfall == pytest.approx(0.3069467, rel=1e-06)
    from mlsysim.physics.robotics import calc_scurve_stop_distance
    fill = calc_scurve_stop_distance(stop.speed, stop.min_deceleration, stop.jerk)
    assert fill.m_as('mm') == pytest.approx(stop.braking_room.m_as('mm'), abs=1e-09)
    assert Scenarios.WarehouseFinishedStopAtDriveLimit.result.min_deceleration is not None
    nominal = Scenarios.WarehouseNavCameraAgeAtAisleSpeed.result
    dim = Scenarios.WarehouseNavCameraAgeDimAisle.result
    assert nominal.age.m_as('ms') == pytest.approx(51.6)
    assert (dim.age - nominal.age).m_as('ms') == pytest.approx(8.0)
    assert (dim.age_travel - nominal.age_travel).m_as('mm') == pytest.approx(10.4)
    arg = Scenarios.SafetyArgumentConfidence
    assert arg.conjunction == pytest.approx(0.99 ** 10) and arg.multilegged == pytest.approx(0.99)
    scanner = Embodied.MobileManipulator.WarehouseMobileManipulator.safety_scanner
    assert scanner.response_time_standard_mode.m_as('ms') == 160.0
    rec = load_series('slides/vol4/16_release/images/data/challenger_oring_distress_record.csv')
    assert len(rec) == 23 and min((e['joint_temp_f'] for e in rec.entries)) == 53
    assert sum((1 for e in rec.entries if e['orings_distressed'] > 0)) == 7

def test_partition_screen_and_scalability_law():
    """Screening a work partition and its universal-scalability-law form (added 2026-10-20)."""
    s = Scenarios.RepairPartitionScreen
    assert s.best_workers == 6 and s.best_time == pytest.approx(0.43333333)
    assert s.elapsed[4] == pytest.approx(0.46) and s.elapsed[8] == pytest.approx(0.44)
    assert s.speedup_cap == pytest.approx(5) and s.continuous_best == pytest.approx(6.3245553)
    h = Scenarios.RepairPartitionScreenHalvedOverhead
    assert h.best_workers == 9 and s.best_speedup < h.best_speedup < s.speedup_cap
    u = Scenarios.RepairPartitionScalability
    assert u.best_workers == s.best_workers and u.peak_capacity == pytest.approx(1 / s.best_time)
    assert u.first_slower == 41 == int(u.retrograde_threshold) + 1
    r = Scenarios.ResearchFanoutScalability
    assert r.best_workers == 21 and r.peak_capacity == pytest.approx(5.46875)

def test_delegation_break_even_schedule_and_handoffs():
    """Token break-even, the dependency schedule, reference passing, and message rounds (added 2026-10-20)."""
    b = Scenarios.ResearchSystemTokenBreakEven
    assert b.token_multiple == pytest.approx(3.75) and b.max_single_acceptance == pytest.approx(0.26666667)
    assert b.cost_ratio_if_gain_is_acceptance == pytest.approx(1.9716088) and b.required_multi_acceptance == pytest.approx(2.25)
    d = Scenarios.InterfaceMigrationSchedule
    assert d.work.m_as('s') == pytest.approx(275) and d.makespan.m_as('s') == pytest.approx(174.5)
    assert d.peak_slots == 3 and d.free_float[6].m_as('s') == pytest.approx(26.5)
    assert d.finish[2] == d.finish[3] and d.speedup == pytest.approx(1.5759312)
    r = Scenarios.SixChildReferencePassing
    assert r.pasted_tokens == 288000 and r.reference_total_tokens == 23700
    assert r.reference_cost.m_as('USD') == pytest.approx(0.0711) and r.reduction == pytest.approx(12.151899)
    assert r.break_even_read_tokens == 47650 and r.full_read_tokens > r.pasted_tokens

def test_shared_writes_reviewers_and_fanout_deadline():
    """Write collisions, correlated reviewers, and a fan-out deadline (added 2026-10-20)."""
    w = Scenarios.ThousandFileSharedWrites
    assert w.collision[2] == pytest.approx(0.0960315) and w.collision[8] == pytest.approx(0.9442061)
    assert w.even_odds_fleet == 5 and abs(w.disjoint[8] - w.disjoint_approx[8]) < 0.01
    v = Scenarios.FiveCorrelatedReviewers
    assert v.independent_majority(5) == pytest.approx(0.05792) and v.correlated_majority(5) == pytest.approx(0.1667566)
    assert v.floor == pytest.approx(0.149542, rel=1e-05)
    assert v.floor < v.correlated_majority(15) < v.correlated_majority(5)
    assert v.independent_unanimous == pytest.approx(0.00032) and v.unanimous_ratio == pytest.approx(122.447619)
    f = Scenarios.EightWorkerDeadlineFanout
    assert f.all_on_time == pytest.approx(0.43046721) and f.quorum_on_time == pytest.approx(0.96190821)
    assert f.even_odds_workers == 7 and f.all_on_time_by_fleet[7] < 0.5 < f.all_on_time_by_fleet[6]

def test_credit_window_allowances_and_delegation_cohort():
    """A credit window, split allowances, and the matched delegation cohort (added 2026-10-20)."""
    c = Scenarios.DiffReviewCreditWindow
    assert c.service_rate.m_as('1/s') == pytest.approx(0.08) and c.backlog == pytest.approx(2952)
    assert c.backlog_wait.m_as('hour') == pytest.approx(10.25) and c.window_residence.m_as('s') == pytest.approx(100)
    assert c.window_for_target == 4
    m = Scenarios.MatchedDelegationCohort
    assert m.spend(m.single).m_as('USD') == pytest.approx(12) and m.spend(m.delegated).m_as('USD') == pytest.approx(18)
    assert m.discarded(m.single).m_as('USD') == pytest.approx(3) and m.discarded(m.delegated).m_as('USD') == pytest.approx(6)
    assert m.cost_ratio == pytest.approx(1.5) and m.marginal_on_time_cost.m_as('USD') == pytest.approx(0.75)
    assert not m.passes(m.single) and m.passes(m.delegated) and (m.discard_share == pytest.approx(1 / 3))

def test_responsibility_scenarios():
    """Decision chain, oversight time, the assistant's queue and labor, stage exit (added 2026-10-20)."""
    c = Scenarios.AdmittedCommandDecisionChain.completeness_by_link
    assert c[0.99] == pytest.approx(0.94148015, rel=1e-07) and c[0.9] == pytest.approx(0.531441)
    o = Scenarios.WarehouseRackEndOversightTime
    assert o.hazard_window.m_as('s') == pytest.approx(1.1 / 1.3)
    assert o.takeover_travel.m_as('m') == pytest.approx(5.8045)
    assert o.takeover_over_window == pytest.approx(5.2768182, rel=1e-07)
    q = Scenarios.WarehouseRemoteAssistantQueue
    r = q.result
    assert r.utilization == pytest.approx(0.5) and r.mean_wait.m_as('s') == pytest.approx(180.0)
    assert r.p99_wait.m_as('s') == pytest.approx(1408.328282, rel=1e-08)
    assert r.busy_per_hour.m_as('minute/hour') == pytest.approx(30.0) and q.max_machines == 20
    assert Scenarios.WarehouseRemoteAssistantQueueTenMachines.result.mean_wait.m_as('s') == pytest.approx(60.0)
    assert Scenarios.WarehouseRemoteAssistantQueueThirtyTwoMachines.result.utilization == pytest.approx(0.8)
    lab = Scenarios.WarehouseRemoteAssistantLabor
    assert lab.request_time.m_as('hour') == pytest.approx(8.0) and lab.total.m_as('hour') == pytest.approx(52 / 3)
    assert lab.shifts == pytest.approx(13 / 6)
    e = Scenarios.WarehouseSupervisedStageExit
    assert e.exposure(e.rates[0]).m_as('hour') == pytest.approx(2995.7322736, rel=1e-09)
    assert e.calendar_days(e.rates[0]).m_as('day') == pytest.approx(18.7233267, rel=1e-08)

def test_responsibility_groups_and_records():
    """Detection shortfall, group samples, exposure, retention, and the Cruise penalties (added 2026-10-20)."""
    from mlsysim import Literature
    from mlsysim.physics import calc_consecutive_detection_frames, calc_clopper_pearson_interval
    assert calc_consecutive_detection_frames(0.95, 0.001) == 3 and calc_consecutive_detection_frames(0.9, 0.001) == 3
    lo, hi = calc_clopper_pearson_interval(190, 200)
    assert lo == pytest.approx(0.9099725, rel=1e-06) and hi == pytest.approx(0.9757658, rel=1e-06)
    d = Scenarios.WarehouseGroupDetectionShortfall.result
    assert d.reference_frames == 3 and [r.frames for r in d.rows] == [4, 5, 8] and d.within_braking
    mid = d.rows[1]
    assert mid.distance_lost.m_as('mm') == pytest.approx(43.333333, rel=1e-07)
    assert mid.impact_speed.m_as('m/s') == pytest.approx(0.4163332, rel=1e-06)
    assert mid.impact_energy.m_as('J') == pytest.approx(31.893333, rel=1e-07)
    assert d.reference_stop.m_as('m') == pytest.approx(0.66118, rel=1e-06)
    g = Scenarios.PedestrianGroupEvaluationSample
    assert (g.small, g.large, g.per_group_needed) == (200, 9800, 1059)
    assert g.width_ratio == pytest.approx(7.5335335, rel=1e-06)
    x = Scenarios.WarehouseCrossingExposure
    assert x.concentration == pytest.approx(8.0) and x.rest_ratio == pytest.approx(2 / 9)
    t = Scenarios.WarehouseEventTriggeredRetention
    assert t.window.m_as('s') == pytest.approx(30.0) and t.reduction == pytest.approx(96.0)
    p = Scenarios.CruiseDisclosurePenalties
    assert p.per_day_penalty == Literature.IncidentReports.CruiseCpucPenaltyUsd == 112500
    assert p.total == 2112500
    assert len(load_series('slides/vol4/17_responsibility/images/data/li2025_pedestrian_miss_rate_by_age.csv')) == 8 and len(load_series('slides/vol4/17_responsibility/images/data/wilson2019_bdd100k_person_ap75.csv')) == 8
    li = load_series('slides/vol4/17_responsibility/images/data/li2025_pedestrian_miss_rate_by_age.csv').entries
    assert sum((e['miss_rate_child_pct'] for e in li)) / 8 == pytest.approx(43.89, abs=0.005)
    w = load_series('slides/vol4/17_responsibility/images/data/wilson2019_bdd100k_person_ap75.csv').entries

def test_ml_operations_part1_scenarios():
    """Release footprints, automation break-even, skew cost, A/B sizing, drift (added 2026-10-20)."""
    from mlsysim import Literature
    from mlsysim.core.units import Q_
    from mlsysim.physics import calc_two_proportion_detectable_lift, calc_exponential_decay_rate
    r = Scenarios.DlrmReleaseFootprintH100
    assert r.accelerators_per_copy == 2 and r.accelerators_for(2) == 4
    assert r.free_during_retrain == 2 and r.free_after(3) == 2
    assert r.retrain_accelerator_hours.m_as('hour') == pytest.approx(24)
    n = Scenarios.ResNet50Fp16ReleaseFootprintH100
    assert n.copy_bytes.m_as('MB') == pytest.approx(51.114064, rel=1e-06)
    k = Scenarios.KeywordSpotterDualBankEsp32S3
    assert k.banks_bytes.m_as('KB') == pytest.approx(400) and k.flash_share == pytest.approx(0.0953674, rel=1e-05)
    assert Scenarios.RetrainingPipelineAutomation.breakeven.m_as('week') == pytest.approx(20)
    s = Scenarios.RankingSkewOnePercent
    assert s.errors_per_day == pytest.approx(10000) and s.cost_over(Q_(365, 'day')).m_as('USD') == pytest.approx(365000)
    a = Scenarios.ConversionRelativeLiftExperiment
    assert a.per_variant == pytest.approx(745643.57, rel=1e-06)
    assert a.detectable_lift() == pytest.approx(0.00546129, rel=1e-05)
    assert calc_two_proportion_detectable_lift(0.05, a.per_variant, a.z_alpha, a.z_beta) == pytest.approx(a.lift)
    d = Scenarios.RankingServiceAccuracyDrift
    assert d.decay_rate.m_as('1/day') == pytest.approx(0.00235564, rel=1e-05)
    assert calc_exponential_decay_rate(1.0, 0.5, 1.0) == pytest.approx(0.693147, rel=1e-06)
    out = [row['outages'] for row in load_series('slides/vol1/14_ml_ops/images/data/outages_by_mlness.csv').entries]

def test_benchmark_decision_scenarios():
    """Run noise, Welch intervals, run and sample counts, paired accuracy, run order, gates (added 2026-10-20)."""
    from pytest import approx
    from mlsysim.core.units import Q_
    from mlsysim.physics import calc_t_critical, calc_spearman_rank_correlation
    assert Scenarios.AccuracyWithinOnePointSampleSize.samples_needed == 1825
    c = Scenarios.NinetyFiveAgainstNinetyFourOnSameImages
    assert c.net_gap == 10 and c.max_discordant == 26 and (c.independent_interval[0] < 0)
    b = Scenarios.WiderBeamSearchTranslation
    assert b.slowdown == approx(4.0) and b.budget_ratio == approx(2.0) and (b.metric_gain == approx(0.5))
    g = Scenarios.MobileNetV2Int8CameraTrapGates
    assert g.edge_images == 50 and g.edge_half_width == approx(12.908294, rel=1e-05)
    assert g.top1_loss == approx(0.9) and g.ece_ratio == approx(0.089 / 0.031)
    assert calc_t_critical(4, 0.95) == approx(2.776445, rel=1e-06)
    assert calc_spearman_rank_correlation([1, 2, 2, 3], [1, 3, 2, 4]) == approx(0.948683, rel=1e-05)
    r = Scenarios.ImageNetV2Shift.result
    assert r.network_rank_correlation == approx(0.996634, rel=1e-05)

def test_token_serving_scenarios():
    """Serving a stream of tokens on one machine: decode batches, speculation, KV budgets, stall,
    schedules, vLLM's pool, disaggregation, energy per token, Little's law on MLPerf (added 2026-10-20)."""
    from mlsysim.core.units import Q_
    g = Scenarios.Gpt2XlChatDecodeBatchesH100
    assert g.kv_per_token.m_as('byte') == pytest.approx(307200)
    assert g.step_time(1).m_as('ms') == pytest.approx(0.989425, rel=1e-05)
    assert g.token_rate(32).m_as('1/s') == pytest.approx(8204.293, rel=1e-06)
    assert g.cache_share(32) == pytest.approx(0.770402, rel=1e-05)
    assert g.rate_ceiling.m_as('1/s') == pytest.approx(10649.36, rel=1e-06)
    assert g.max_batch_within(Q_(20, 'ms')) == 203 and g.max_batch_in_memory == 263
    assert g.prefill_time.m_as('ms') == pytest.approx(3.88271, rel=1e-06)
    assert g.ridge.m_as('flop/byte') == pytest.approx(295.2239, rel=1e-06)
    sp = Scenarios.Gpt2XlSpeculativeGpt2SmallH100
    assert sp.round_time.m_as('ms') == pytest.approx(1.191642, rel=1e-06)
    assert sp.expected_tokens == pytest.approx(2.7731) and sp.tpot_gain == pytest.approx(2.083993, rel=1e-06)
    k8 = Scenarios.Llama3_8BFp16ConversationsH100.result
    assert k8.kv_per_token.m_as('byte') == 131072 and k8.max_requests == 65
    assert k8.kv_budget.m_as('GB') == pytest.approx(69.83935, rel=1e-06)
    k70 = Scenarios.Llama3_70BInt4KvBudgetH100
    assert k70.result.kv_budget.m_as('GB') == pytest.approx(50.59935, rel=1e-06)
    assert k70.context_fit() == pytest.approx(4825.53, rel=1e-06)
    cs = Scenarios.Llama3_8BInt4ChatServiceH100
    assert cs.ttft.m_as('ms') == pytest.approx(41.93186, rel=1e-06)
    assert cs.tpot.m_as('ms') == pytest.approx(1.56765, rel=1e-05)
    assert cs.kv_capacity_tokens == 589824 and cs.resident_requests == 469
    assert cs.request_rate.m_as('1/s') == pytest.approx(10)
    assert cs.cost_per_million_tokens.m_as('dollar') == pytest.approx(0.0663482, rel=1e-05)
    e = Scenarios.H100ChatEnergyPerToken
    assert e.joules_per_token.m_as('J') == pytest.approx(1.096491, rel=1e-06) and e.low_use_ratio == pytest.approx(6)

def test_fairness_gates_price_and_feedback():
    """Credit service rates, price, feedback loop, evaluation budget, COMPAS identity (added 2026-10-20)."""
    from mlsysim import Literature
    from mlsysim.physics import calc_cycles_to_gate, calc_feedback_disparity, calc_fpr_from_ppv, calc_gap_samples_per_group
    c = Scenarios.CreditServiceTwoGroups
    assert (c.size('A'), c.size('B')) == (10000, 2000)
    assert c.positive_rate('A') == pytest.approx(0.55) and c.positive_rate('B') == pytest.approx(0.4)
    assert c.tpr('A') == pytest.approx(0.9) and c.tpr('B') == pytest.approx(0.6)
    assert c.fpr('A') == pytest.approx(0.2) == c.fpr('B')
    assert c.fnr('B') / c.fnr('A') == 4 and c.overall_accuracy == pytest.approx(0.825)
    assert c.share('B') == pytest.approx(1 / 6) and c.base_rate('A') == c.base_rate('B') == 0.5
    p = Scenarios.CreditServiceEqualOpportunityPrice
    assert p.baseline_utility.m_as('dollar') == pytest.approx(150.0)
    assert p.added_repaid_value.m_as('dollar') == pytest.approx(225.0)
    assert p.added_default_cost.m_as('dollar') == pytest.approx(300.0)
    assert p.net_change.m_as('dollar') == pytest.approx(-75.0) and p.loss_fraction == pytest.approx(0.5)
    assert p.population_average_loss.m_as('dollar') == pytest.approx(12.5)
    f = Scenarios.CreditServiceBiasFeedback
    assert f.factor == pytest.approx(1.1) and f.crossing == pytest.approx(5.3596124, rel=1e-07)
    assert f.breach_cycle == 6 and f.gap_at_breach == pytest.approx(5.314683, rel=1e-06)
    assert f.unmitigated_at_horizon == pytest.approx(43.923, rel=1e-06)
    assert f.sampled_gap_at_breach == pytest.approx(1.594323, rel=1e-06)
    assert calc_feedback_disparity(3.0, 1.1, 2) == pytest.approx(3.63)
    assert calc_cycles_to_gate(1.0, 2.0, 8.0) == pytest.approx(3.0)
    b = Scenarios.OnePercentGroupEvaluationBudget
    assert b.samples_needed == 9604 and b.random_collection == pytest.approx(1000000.0) and (b.random_multiple == 100)
    assert b.gap_samples_per_group() == 1568 and b.gap_samples_per_group(0.01) == 19208
    assert b.gap_half_width_at(9604) == pytest.approx(0.0141419, rel=1e-05)
    assert calc_gap_samples_per_group(0.5, 0.01, 1.96) == 19208
    assert calc_fpr_from_ppv(0.5, 0.5, 0.0) == pytest.approx(1.0)
    fr = Scenarios.FairnessAccuracyFrontierSketch
    assert fr.point('A')[1] == pytest.approx(0.9472676, rel=1e-06) and fr.point('B')[1] == pytest.approx(0.9132121, rel=1e-06)
    assert fr.point('C')[1] == pytest.approx(0.85) and fr.disparity_ratio('A', 'B') == pytest.approx(3.6)
    x = Scenarios.RecommenderExplorationExposure
    assert x.impressions == 200000000 and x.catalog_coverage == pytest.approx(2e-05)
    assert x.explore_impressions == pytest.approx(2000000.0)
    F = Literature.Fairness

def test_agent_cost_accounting():
    """Cohort cost per accepted task and the levers on it (added 2026-10-20)."""
    from mlsysim.physics import calc_kingman_wait_ratio, calc_speculative_round_speedup
    assert calc_kingman_wait_ratio(0.5, 1.0, 1.0) == pytest.approx(1.0)
    assert calc_speculative_round_speedup(0.4, 5, 0.08) == pytest.approx(1.0)
    p = Scenarios.PriceAtFixedCapabilityDecline.result
    assert p.series_count == 21 and round(p.rate_low) == 9 and (round(p.rate_median) == 46)
    assert round(p.rate_high, -1) == 860 and round(p.series_rates['MMLU / GPT-3']) == 12
    f = Scenarios.FrontierSweCostVsScore.result
    assert f.configurations == 19 and round(f.cost_ratio) == 10 and (round(f.score_gap) == 3)
    assert round(100 * f.median_spread) == 21
    c = Scenarios.RepairCohortTokenPriceComparison.result
    b = Scenarios.RepairResendCacheBill.result
    assert (b.sent_tokens, b.fresh_tokens) == (892500, 51500)
    k = Scenarios.RepairCompactionPayback.result
    assert Scenarios.RepairCompactionPaybackHalfCachedRate.result.breakeven_calls == 23
    r = Scenarios.CheapRetryAgainstStrongCall.result
    e = Scenarios.SerialRepairEpisode.result
    s = Scenarios.SerialRepairEpisodeSpeculation.result
    q = Scenarios.CodeEditThreeTierCascade.result
    a = Scenarios.RepairTrajectoryQuotaAdmission.result
    w = Scenarios.BurstyModelServerQueuePrice.result
    d = Scenarios.DedicatedCapacityBreakEven.result
    g = Scenarios.RepairBudgetLateBill.result
    assert (g.settled_final, g.reserved_final, g.free_final) == (1972, 0, 3028)
    assert (g.child_free_at_stop, g.child_apparent_free, g.child_free_settled) == (650, 1650, 950)
    h = Scenarios.DependencyUpgradeArchitectures.result
    n = Scenarios.CodeReviewAdaptationPayback.result
    l = Scenarios.LighthouseWorkloadLevers.result.rows

def test_premises_no_sensor_reports():
    """Oil-film budget, film detector, restricted transit, packaging cell, shield bill, latch spring (added 2026-10-20)."""
    from mlsysim import Literature, Sensors
    f = Scenarios.WarehouseFinishedStopOilFilm.result
    assert f.deceleration.m_as('m/s**2') == pytest.approx(0.4903325)
    assert f.required_distance.m_as('m') == pytest.approx(2.13182, rel=1e-05) and (not f.fits)
    d = Scenarios.WarehouseFilmDetectorRange.result
    assert (d.required_distance + d.added_delay_travel).m_as('mm') == pytest.approx(1098.9925)
    z = Scenarios.WarehouseFilmDetectorClosure.result
    assert z.trials_for_target == 2995 and z.rate_upper_exact == pytest.approx(0.00299124955, rel=1e-08)
    r = Scenarios.WarehouseRestrictedTransit
    assert r.speed_cut == pytest.approx(3 / 13) and r.time_increase == pytest.approx(0.3)
    assert r.leg_time_restricted.m_as('s') == pytest.approx(30.0)
    cell = Scenarios.PackagingCellTransit
    assert cell.delay.m_as('ms') == pytest.approx(64.0)
    assert cell.result.stop_distance.m_as('mm') == pytest.approx(171.0)
    assert cell.result.speed_ceiling.m_as('m/s') == pytest.approx(0.5471828, rel=1e-06)
    assert cell.kinetic_energy.m_as('J') == pytest.approx(13.5)
    a_tau, two_a_d, root = cell.root_terms
    assert (a_tau.m_as('m/s'), two_a_d.m_as('m**2/s**2')) == pytest.approx((0.96, 1.35))
    assert (root - a_tau).m_as('m/s') == pytest.approx(cell.result.speed_ceiling.m_as('m/s'))
    mode = Scenarios.PackagingCellRestrictedMode
    assert mode.result.stop_distance.m_as('mm') == pytest.approx(121 / 3) and mode.remaining.m_as('mm') == pytest.approx(59 / 3)
    assert Scenarios.PackagingCellHalfProcessing.result.speed_ceiling.m_as('m/s') == pytest.approx(0.6613327, rel=1e-06)
    assert Scenarios.PackagingCellHalfBrake.result.speed_ceiling.m_as('m/s') == pytest.approx(0.4715251, rel=1e-06)
    q = Scenarios.PackagingCellQualifiedBrake
    assert q.result.speed_ceiling.m_as('m/s') == pytest.approx(0.5242167, rel=1e-06)
    assert q.result.stop_distance.m_as('mm') == pytest.approx(42.416667, rel=1e-06)
    curtain = Sensors.SafetyLightCurtain.SICK_deTec4_Set_30_450
    assert curtain.response_time.m_as('ms') == 10.0 and curtain.pfh_d.m_as('1/hour') == pytest.approx(1.53e-08)
    b = Scenarios.ShieldCoverageBill.result
    assert b.exposure.m_as('hour') == pytest.approx(216000.0)
    assert b.proposal_rate_upper.m_as('1/hour') == pytest.approx(1.386913e-05, rel=1e-06)
    assert b.residual_rate.m_as('1/hour') == pytest.approx(1.386913e-09, rel=1e-06) and (not b.meets_target)
    assert b.required_coverage == pytest.approx(0.9999278974, rel=1e-09)
    assert b.days_needed.m_as('day') == pytest.approx(124.822178, rel=1e-06)
    assert (b.coverage_trials, b.coverage_trials_joint) == (29956, 36887)
    assert b.residual_rate_joint.m_as('1/hour') == pytest.approx(1.7078146e-09, rel=1e-06)
    s = Scenarios.WarehouseLatchSeriesSpring
    assert s.arm_deceleration.m_as('m/s**2') == pytest.approx(2.9508197, rel=1e-06)
    assert s.driven_stroke.m_as('mm') == pytest.approx(1.8944444, rel=1e-06)
    assert s.series_stiffness_max.m_as('N/m') == pytest.approx(44868.035, rel=1e-06)
    assert s.spring_stiffness_max.m_as('N/m') == pytest.approx(50536.746, rel=1e-06)
    assert s.spring_stroke.m_as('mm') == pytest.approx(1.9787582, rel=1e-06)
    assert [p.m_as('N') for p in (s.coasting_rigid_guarded, s.coasting_rigid_fast, s.coasting_spring_fast)] == pytest.approx([67.416615, 219.601913, 74.894354], rel=1e-06)
    assert Literature.RobotDeployments.AgilityDigitTotesMoved == 100000

def test_closing_an_agent_course():
    """Remediation attempt, lost reply, residual violations, passing set, blame search,
    specification bound, passing-patch audit, and METR's 50 against 80 percent horizons (added 2026-10-20)."""
    from mlsysim import Agents
    r = Scenarios.RemediationIncidentAttempt.result
    assert r.attempt_cost == pytest.approx(20.45) and r.cost_per_accepted == pytest.approx(20.45 / 0.7)
    assert r.token_share == pytest.approx(0.15 / 20.45)
    assert r.grown_attempt_cost == pytest.approx(20.6) and r.grown_attempt_rise == pytest.approx(0.15 / 20.45)
    assert r.grown_break_even_acceptance == pytest.approx(0.7 * 20.6 / 20.45)
    w = Agents.Remediation.OnCall
    assert w.attempt_cost.m_as('USD') == pytest.approx(20.45) and w.approval_share == pytest.approx(20 / 20.45)
    assert Scenarios.RemediationCanaryPivotLoss.attempt_cost.m_as('USD') == pytest.approx(20.45)
    b = Scenarios.BlindRetryAgainstStatusLookup
    assert b.result.timeouts == pytest.approx(124.9) and b.result.duplicates == pytest.approx(99.8001)
    assert b.duplicate_cost_per_day.m_as('USD') == pytest.approx(199600.2)
    assert b.lookup_cost_per_day.m_as('USD') == pytest.approx(1.249)
    assert b.unapplied_closures == pytest.approx(25) and b.lost_request_share == pytest.approx(0.001 / 0.004996)
    v = Scenarios.ResidualViolationsFiftyTurns
    assert v.trajectory_violation == pytest.approx(0.0487944, rel=1e-05)
    assert round(v.violating_per_day) == 488
    assert Scenarios.ThirtyPercentValidPassingSet.result.posterior == pytest.approx(0.035 / 0.305)
    assert Scenarios.HalvedHealthWindow.interval_widening == pytest.approx(2 ** 0.5)
    c = Scenarios.CheckedWindowBlameSearch
    assert c.full_paths == 4 ** 50 and round(c.full_exponent) == 30 and (c.window_paths == 1024)
    assert c.window_search_time.m_as('minute') == pytest.approx(20.48)

def test_volume1_synthesis_recall_scenarios():
    """Recalled calculations of the Volume I synthesis (added 2026-10-20)."""
    from mlsysim import Literature
    assert Scenarios.Conv3x3At256Channels.separable_ratio == pytest.approx(9 * 256 / 265)
    d = Scenarios.Llama2_70B_DecodeTwoH100.result
    assert d.memory_time.m_as('ms') == pytest.approx(20.8955224, rel=1e-06)
    assert d.compute_time.m_as('ms') == pytest.approx(0.0707786, rel=1e-05)
    assert d.bound_ratio == pytest.approx(Scenarios.Llama2_70B_DecodeH100.result.bound_ratio)
    assert Scenarios.ChatEndpointTtftMeanVersusP99.gap_high == pytest.approx(40)
    a = Scenarios.HostPreprocessingRequestPath.result
    assert a.stage_fraction == pytest.approx(0.1) and a.upgraded_total_time.m_as('percent') == pytest.approx(91)
    assert a.overall_speedup == pytest.approx(1 / 0.91)

def test_ml_operations_part2_scenarios():
    """Drift confirmation, detection-delay loss, retraining interval, PSI, K-S, alerts, slices,
    rubric audit, severity policy (added 2026-10-20)."""
    from mlsysim import Literature
    from mlsysim.core.units import Q_
    from mlsysim.physics import calc_one_proportion_sample_size, calc_detection_delay_loss, calc_optimal_retraining_interval, calc_retraining_cost_rate
    c = Scenarios.LabeledOutcomeDriftConfirmation
    assert c.outcomes_needed() == pytest.approx(184.77, rel=1e-05)
    assert [c.outcomes_needed(d) for d in c.compared_drops] == pytest.approx([1030.093, 3931.769, 15328.286], rel=1e-05)
    assert c.collection_time('high').m_as('minute') == pytest.approx(1000 / 60)
    assert c.collection_time('low').m_as('day') == pytest.approx(10)
    assert calc_one_proportion_sample_size(0.5, 0.4, 1.0, 0.0) == pytest.approx(25)
    d = Scenarios.RecommenderDetectionDelayLoss
    assert d.slow_loss.m_as('USD') == pytest.approx(191780.82, rel=1e-06)
    assert d.annual_savings.m_as('USD') == pytest.approx(739726.03, rel=1e-06)
    assert d.full_year_loss.m_as('USD') == pytest.approx(2500000.0) and d.slow_over_fast == pytest.approx(28)
    assert calc_detection_delay_loss(Q_(365, 'USD'), 1.0, Q_(1, 'day')).m_as('USD') == pytest.approx(1)
    r = Scenarios.RankingRetrainingInterval
    assert r.staleness_slope.m_as('USD/day**2') == pytest.approx(1107.1516, rel=1e-06)
    assert r.optimal_interval.m_as('day') == pytest.approx(3.00536, rel=1e-05)
    assert r.cost_rate(Q_(3, 'day')).m_as('USD/day') == pytest.approx(3327.394, rel=1e-06)
    assert r.compute_cost.m_as('USD') == pytest.approx(96) and r.compute_share == pytest.approx(0.0192)
    half = r.optimal_interval.m_as('day') / 2
    assert Scenarios.RankingRetrainingIntervalFourfoldTraffic.optimal_interval.m_as('day') == pytest.approx(half)
    t = calc_optimal_retraining_interval(Q_(50, 'USD'), Q_(1, 'USD/day**2')).m_as('day')
    assert t == pytest.approx(10)
    assert calc_retraining_cost_rate(Q_(10, 'day'), Q_(50, 'USD'), Q_(1, 'USD/day**2')).m_as('USD/day') == pytest.approx(10)
    dr = Scenarios.RankingServiceAccuracyDrift
    assert dr.time_to(dr.later_accuracy).m_as('day') == pytest.approx(28)
    assert dr.accuracy_at(Q_(90, 'day')) == pytest.approx(0.760421, rel=1e-05)
    assert dr.trigger_for_floor(0.88, Q_(10, 'day')) == pytest.approx(0.900976, rel=1e-05)
    assert Scenarios.RankingUserAgePsi.psi == pytest.approx(0.029309, rel=0.0001)
    assert Scenarios.OneModelDailyMetricTests.daily_false_alerts == pytest.approx(7.5)
    s = Scenarios.RecommendationAccuracyByDevice
    assert s.total_share == pytest.approx(1) and s.overall_accuracy == pytest.approx(0.906) and (s.worst_segment == 'Tablet')
    rows = load_series('slides/vol1/14b_ml_ops/images/data/rabanser2019_label_free_detection.csv').entries
    assert [r['samples'] for r in rows][-1] == 10000 and rows[-1]['small'] == pytest.approx(0.54)

def test_responsible_engineering_cost_carbon_and_duties():
    """Ranking-service lifecycle cost, operational and embodied carbon, and the machine cost of
    explanation, review, audit logs, and per-example gradients (added 2026-10-20)."""
    t = Scenarios.RankingServiceLifecycleCost
    assert t.inferences_per_day == pytest.approx(200000000.0)
    assert t.gpu_hours_per_day.m_as('hour') == pytest.approx(2000000.0 / 3600)
    assert t.gpus_busy == pytest.approx(2000000.0 / 3600 / 24)
    assert t.cycle_cost.m_as('USD') == pytest.approx(3200) and t.search_time.m_as('hour') == pytest.approx(500)
    assert t.training_cost.m_as('USD') == pytest.approx(38400)
    assert t.inference_cost.m_as('USD') == pytest.approx(2000000.0 / 3600 * 2.5 * 1095)
    assert t.operations_cost.m_as('USD') == pytest.approx(510000)
    assert t.training_carbon.m_as('kg') == pytest.approx(12 * 800 * 0.16)
    assert t.inference_carbon.m_as('kg') == pytest.approx(2000000.0 / 3600 * 1095 * 0.16)
    assert t.serving_saving.m_as('USD') == pytest.approx(0.2 * t.inference_cost.m_as('USD'))
    assert t.training_saving.m_as('USD') == pytest.approx(19200)
    assert Scenarios.Gpt3TrainingCarbon.carbon.m_as('metric_ton') == pytest.approx(552.123)
    hours = 6 * 8760 * 0.85
    x = Scenarios.CreditDenialExplanationLoad
    assert x.denial_share == pytest.approx(1 - 6300 / 12000)
    assert x.sampled_evaluations == pytest.approx(475000000.0) and x.exact_per_explanation == 2 ** 20
    r = Scenarios.CreditAppealReview
    assert r.reviewer_hours_per_day.m_as('hour') == pytest.approx(100) and r.reviewers == pytest.approx(12.5)
    assert r.compare_hours.m_as('hour') == pytest.approx(100)
    g = Scenarios.DpSgdPerExampleGradients
    assert g.per_batch('KWS').m_as('byte') == pytest.approx(200000 * 4 * 256)
    assert g.per_batch('GPT-2').m_as('byte') == pytest.approx(1500000000.0 * 4 * 256)

def test_frontier_training_cost_trend():
    """Frontier training-cost growth fit over Epoch's notable models (added 2026-10-20)."""
    t = Scenarios.FrontierTrainingCostTrend
    assert 2.5 < t.growth_per_year < 3.5
    assert t.priciest['model'] == 'Grok 4' and t.priciest['training_cost_2023usd'] > 100000000.0

def test_swe_bench_verified_leaderboard_one_attempt():
    """SWE-bench Verified leaderboard: the one-attempt frontier and the same-model loop
    comparison of 2 April 2024 (added 2026-10-20)."""
    from mlsysim import Literature
    calls = load_series('vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_django_12050_calls.csv').entries
    assert len(calls) == 22 and calls[0]['context_chars'] == 5499

def test_chat_service_day_scenario():
    """One day of ChatGPT prompts at the Gemini median prompt (added 2026-10-20)."""
    from mlsysim import Literature, ReferenceStats
    assert float(ReferenceStats.TextTokenization.WordsPerToken) == 0.75
    d = Literature.DatacenterElectricity

def test_vercel_gateway_and_repair_trajectory_scenarios():
    """Production gateway token-usage telemetry and recorded repair trajectory (added 2026-10-21)."""
    g = Scenarios.VercelGatewayTokenGrowth
    assert len(g.index_rows) == 6 and len(g.tool_rows) == 2
    assert g.usage_growth == pytest.approx(7.908)
    assert g.spend_growth == pytest.approx(3.427)
    assert g.price_index_end == pytest.approx(43.3)
    assert g.price_drop_fraction == pytest.approx(0.567)
    assert g.tool_request_share_start == pytest.approx(0.114)
    assert g.tool_request_share_end == pytest.approx(0.222)
    assert g.tool_token_share_start == pytest.approx(0.316)
    assert g.tool_token_share_end == pytest.approx(0.589)
    assert g.tool_tokens_per_request_ratio == pytest.approx(0.589 / 0.222 / (0.411 / 0.778))
    c = Scenarios.MiniSweAgentDjango12050Calls
    assert c.calls == 22
    assert c.first_call_chars == 5499 and c.last_call_chars == 29594
    assert c.context_growth == pytest.approx(29594 / 5499)
    assert c.cumulative_sent_chars == 396997 and c.cumulative_new_chars == 29594
    assert c.reread_ratio == pytest.approx(396997 / 29594)
    assert c.cost_usd == pytest.approx(0.147413)
    w = Scenarios.VercelGatewayWorkloadSpend
    assert len(w.use_cases) == 8 and len(w.open_weight_rows) == 10
    assert w.token_share('AI Personal Assistants') == pytest.approx(0.4)
    assert w.cost_share('AI Personal Assistants') == pytest.approx(0.196)
    assert w.token_share('AI Coding Agents') == pytest.approx(0.218)
    assert w.cost_share('AI Coding Agents') == pytest.approx(0.204)
    assert w.token_share('Back Office') == pytest.approx(0.058)
    assert w.cost_share('Back Office') == pytest.approx(0.15)
    assert w.back_office_vs_assistant_price_ratio == pytest.approx(15.0 / 5.8 / (19.6 / 40.0))
    assert w.open_weight_token_share_start == pytest.approx(0.07)
    assert w.open_weight_token_share_end == pytest.approx(0.715)
    assert w.open_weight_spend_share_end == pytest.approx(0.226)
    assert w.closed_vs_open_price_ratio_end == pytest.approx(0.774 / 0.285 / (0.226 / 0.715))
