from physx.constants import *
from physx.formulas import calc_training_time_days, calc_amdahls_speedup
from physx.formatting import fmt


def calc_intro_setup():
    """Chapter-wide intro numbers used in prose and footnotes."""
    google_search_b = fmt(GOOGLE_SEARCHES_PER_DAY / 1e9, precision=1)
    gmail_emails_t = fmt(GMAIL_EMAILS_PER_DAY * 365 / 1e12, precision=0)
    gpt4_gpu_m = fmt(GPT4_TRAINING_GPU_DAYS / 1e6, precision=1)

    gpt3_params_b = f"{GPT3_PARAMS.to(Mparam).magnitude/1000:.0f}"
    gpt3_params_billion = f"{gpt3_params_b} billion"

    gpt3_training_zflops = int(GPT3_TRAINING_OPS.magnitude / 1e21)
    gpt3_training_zflops_str = str(gpt3_training_zflops)

    gpt3_gpus = 1024
    gpt3_gpus_str = f"{gpt3_gpus:,}"

    h100_fp16_tflops_str = fmt(H100_FLOPS_FP16_TENSOR, TFLOPs / second, 0)
    h100_fp8_tflops_str = fmt(H100_FLOPS_FP8_TENSOR, TFLOPs / second, 0)
    cpu_fp32_tflops_str = fmt(CPU_FLOPS_FP32, TFLOPs / second, 0)

    return {
        "google_search_b": google_search_b,
        "gmail_emails_t": gmail_emails_t,
        "gpt4_gpu_m": gpt4_gpu_m,
        "gpt3_params_b": gpt3_params_b,
        "gpt3_params_billion": gpt3_params_billion,
        "gpt3_training_zflops_str": gpt3_training_zflops_str,
        "gpt3_gpus": gpt3_gpus,
        "gpt3_gpus_str": gpt3_gpus_str,
        "h100_fp16_tflops_str": h100_fp16_tflops_str,
        "h100_fp8_tflops_str": h100_fp8_tflops_str,
        "cpu_fp32_tflops_str": cpu_fp32_tflops_str,
    }


def calc_gpt3_training():
    """Iron Law example: GPT-3 training time."""
    num_gpus = 1024
    efficiency_eta = 0.45
    target_eta = 0.60

    training_days = calc_training_time_days(
        GPT3_TRAINING_OPS,
        num_gpus,
        A100_FLOPS_FP16_TENSOR,
        efficiency_eta,
    )
    optimized_days = calc_training_time_days(
        GPT3_TRAINING_OPS,
        num_gpus,
        A100_FLOPS_FP16_TENSOR,
        target_eta,
    )

    ops_mag = f"{GPT3_TRAINING_OPS.magnitude:.2e}"
    coeff, exp = ops_mag.split("e+")

    return {
        "num_gpus": num_gpus,
        "num_gpus_str": f"{num_gpus}",
        "efficiency_eta": efficiency_eta,
        "target_eta": target_eta,
        "efficiency_eta_pct_str": f"{int(efficiency_eta * 100)}",
        "target_eta_pct_str": f"{int(target_eta * 100)}",
        "ops_coeff": coeff,
        "ops_exp": int(exp),
        "peak_tflops": f"{A100_FLOPS_FP16_TENSOR.to(TFLOPs / second).magnitude:.0f}",
        "days_initial": f"{training_days:.0f}",
        "days_optimized": f"{optimized_days:.0f}",
        "days_saved": f"{training_days - optimized_days:.0f}",
    }


def calc_waymo_data_rates():
    """Waymo data rate strings for intro narrative."""
    return {
        "waymo_data_low_str": f"{int(WAYMO_DATA_PER_HOUR_LOW.to(TB / hour).magnitude)}",
        "waymo_data_high_str": f"{int(WAYMO_DATA_PER_HOUR_HIGH.to(TB / hour).magnitude)}",
    }


def calc_amdahls_pitfall():
    """Amdahl's law pitfall example for latency breakdown."""
    t_inference = 45  # ms
    t_pre = 60  # ms
    t_post = 25  # ms
    t_total = t_pre + t_inference + t_post

    s_inf = 3
    t_inf_new = t_inference / s_inf
    t_total_new = t_pre + t_inf_new + t_post

    p_inf = t_inference / t_total
    overall_speedup = calc_amdahls_speedup(p_inf, s_inf)
    improvement_pct = (1 - (1 / overall_speedup)) * 100
    naive_pct = (1 - (1 / s_inf)) * 100

    return {
        "t_inference_str": str(t_inference),
        "t_pre_str": str(t_pre),
        "t_post_str": str(t_post),
        "total_ms": f"{t_total}",
        "new_total_ms": f"{t_total_new:.0f}",
        "improv_pct": f"{improvement_pct:.0f}",
        "naive_p": f"{naive_pct:.0f}",
        "t_inf_new_str": f"{int(t_inf_new)}",
    }
