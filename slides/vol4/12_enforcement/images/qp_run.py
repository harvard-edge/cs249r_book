# SE 12.6.1: the two-axis projection, by SciPy and by equation 8.
import numpy as np
from scipy.optimize import minimize
u0 = np.array([3.5, 3.0])
a = np.array([2.0, 1.0])
b = 5.0
cost = lambda u: 0.5 * np.sum((u - u0) ** 2)
safe = {"type": "ineq",
        "fun": lambda u: b - a @ u}
r = minimize(cost, u0, method="SLSQP",
             constraints=[safe])
lam = max(0.0, (a @ u0 - b) / (a @ a))
print("SLSQP  u* =", np.round(r.x, 4), " iterations:", r.nit)
print("closed u* =", u0 - lam * a, " lambda* =", lam)
