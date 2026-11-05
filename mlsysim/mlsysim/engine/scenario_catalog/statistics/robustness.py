"""Named statistics scenarios for robustness.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....datasets.registry import Datasets as _Datasets
from ... import scenario_types as _st
from ....core.units import FLOAT_FORMAT_BITS as _FLOAT_FORMAT_BITS
from ....core.units import COLOR_DEPTH_8BIT as _COLOR_DEPTH_8BIT
from ....physics.statistics import calc_ks_coefficient as _calc_ks_coefficient


# --- Robust AI: silent faults, drift tests, per-request answering, attacks (added 2026-10-20) ---
FeatureMeanShiftTest = _st.MeanShiftTestScenario(
    name="One feature's mean moves from 0.50 to 0.55 over the last 1,000 requests",
    description="Spread 0.3; is the move noise? Standard error, z, and the two-sided p-value.",
    reference_mean=0.5,  # scenario assumption
    observed_mean=0.55,  # scenario assumption
    feature_sd=0.3,  # scenario assumption
    n_samples=1000,  # scenario assumption
    significance_bound=0.001,  # scenario assumption: the bound the alert reports
)


FraudCountryPsi = _st.BinShareShiftScenario(
    name="User-country PSI of a fraud model after six months in production",
    description="Baseline and current traffic shares by country bucket; the Other bucket grows.",
    bins=["USA", "UK", "Germany", "France", "Other"],
    baseline_shares=[0.45, 0.20, 0.15, 0.10, 0.10],  # scenario assumption
    current_shares=[0.38, 0.18, 0.14, 0.12, 0.18],  # scenario assumption
)


FraudAmountDrift = _st.GaussianDriftScenario(
    name="Transaction-amount drift of the same fraud model, log-dollars fitted as normals",
    description="KL divergence of the reference fit from the current fit.",
    reference_mean=4.2,  # scenario assumption
    reference_sd=1.1,  # scenario assumption
    current_mean=4.5,  # scenario assumption
    current_sd=1.3,  # scenario assumption
    kl_trigger=0.05,  # scenario assumption: the service's calibrated investigate threshold
)


FraudAmountKs = _st.KsDriftScenario(
    name="K-S test on transaction amount, 10,000 against 10,000, Bonferroni level 0.01",
    description="The observed largest CDF gap against the critical value at the corrected level.",
    coefficient=_calc_ks_coefficient(0.01),
    significance=0.01,  # scenario assumption: Bonferroni-corrected level of the Level 1 rule
    n_reference=10_000,  # scenario assumption
    n_current=10_000,  # scenario assumption
    observed_statistic=0.089,  # scenario assumption: observed largest CDF gap
)


PgdSevenTraining = _st.AdversarialTrainingPassesScenario(
    name="PGD-7 adversarial training: seven attack steps and one update per batch",
    description="Equal-cost work model with no reuse across steps (Madry et al. 2018).",
    attack_passes=7,  # scenario assumption: PGD-7
    update_passes=1,
)


SmoothingRadiusCifar10 = _st.CertifiedRadiusScenario(
    name="Randomized smoothing at sigma 0.5 and p_A 0.999 against the 8/255 box on a CIFAR-10 image",
    description="Binary certified radius against the l2 length of the l-infinity box corner.",
    noise_sd=0.5,  # scenario assumption
    top_class_probability=0.999,  # scenario assumption
    linf_levels=8,  # the RobustBench CIFAR-10 l-infinity budget, 8/255
    pixel_levels=_COLOR_DEPTH_8BIT,
    coordinates=(_Datasets.CIFAR10.image_width * _Datasets.CIFAR10.image_height * _Datasets.CIFAR10.image_channels),
)


Fp32WeightExponentFlip = _st.FloatBitFlipScenario(
    name="The top exponent bit of an FP32 weight of 0.042 flips",
    description="The value is multiplied by 2^128 and stays finite, so NaN and infinity checks pass.",
    value=0.042,  # scenario assumption: an illustrative weight
    flipped_bit=sum(_FLOAT_FORMAT_BITS["fp32"]) - 1,  # bit 30, the top exponent bit
    exponent_bits=_FLOAT_FORMAT_BITS["fp32"][0],
    fraction_bits=_FLOAT_FORMAT_BITS["fp32"][1],
)
