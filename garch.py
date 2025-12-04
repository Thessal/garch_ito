from optimize import loop
from vol_est import rv_daily
from optimize import optimize_2param
loop(rv_daily, optimize_2param, "garch_daily", (1e-3,1e-3,0.))