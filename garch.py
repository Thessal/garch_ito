from optimize import loop
from vol_realized import rv_daily
from optimize import optimize_3param
loop(rv_daily, rv_daily, optimize_3param, "garch", (0.,0.,1e-3), iter=200)