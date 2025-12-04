from optimize import loop
from vol_realized import rv_naive
from optimize import optimize_4param
loop(rv_naive, optimize_4param, "garch_ito_realized_naive", (0.1,0.,0.,0.), iter=200)