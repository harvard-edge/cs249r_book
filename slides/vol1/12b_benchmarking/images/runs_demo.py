# 95% half-width of a mean of n runs, in percent, from Eos's five run times
# (MLPerf Training v4.0, eos_ngc24.04_mxnet resnet result_0..4; run CSV).
from statistics import stdev, mean
from scipy.stats import t
eos = [13.3062, 13.3178, 13.30162,
       13.20308, 13.18598]
s, m = stdev(eos), mean(eos)
for n in (5, 10, 20):
    hw = t.ppf(0.975, n-1) * s / n**0.5
    print(n, round(100*hw/m, 2))
