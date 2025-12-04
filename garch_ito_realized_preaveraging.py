from optimize import loop
from vol_realized import rv_preaveraged
from optimize import optimize_4param
loop(rv_preaveraged, optimize_4param, "garch_ito_realized_preaveraging", (0.1,0.,0.,0.), iter=200)