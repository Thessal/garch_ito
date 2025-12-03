from optimize import loop
from util import rv_preaveraged
from optimize import optimize_3param
loop(rv_preaveraged, optimize_3param, "garch_ito_unified_naive", (0.5,0.5,1e-3), iter=200)