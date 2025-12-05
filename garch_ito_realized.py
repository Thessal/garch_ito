from optimize import loop
from vol_realized import rv_preaveraged, rv_daily
from optimize import optimize_3param
loop(rv_daily, rv_preaveraged, optimize_3param, "garch_ito_realized", (0.,0.,1e-3), iter=200)

# from optimize import optimize_4param
# loop(rv_preaveraged, optimize_4param, "garch_ito_realized_preaveraging", (-0.0005, -2.0, 1.5, -2.0), iter=50)