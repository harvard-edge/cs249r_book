import numpy as np
import pandas as pd

alpha = 0.05
p_values = np.logspace(-2, -10, 100)
# n = ln(alpha) / ln(1 - p)
n_values = np.log(alpha) / np.log(1 - p_values)

df = pd.DataFrame({
    'target_failure_rate': p_values,
    'required_exposure_hours': n_values,
    'machine_years': n_values / (24 * 365.25)
})
df.to_csv('/Users/VJ/GitHub/MLSysBook-figures/books/vol4/chapters/17-frontier/data/exposure_scaling.csv', index=False)
