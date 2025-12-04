from optimize import loop
from vol_realized import rv_preaveraged
from optimize import optimize_3param
loop(rv_preaveraged, optimize_3param, "garch_ito_unified_preaveraging", (0.,0.,1e-3), iter=50)