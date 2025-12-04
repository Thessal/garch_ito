from optimize import loop
from vol_realized import rv_naive
from optimize import optimize_3param
loop(rv_naive, optimize_3param, "garch_ito_unified_naive", (0.5,0.5,1e-3), iter=200)