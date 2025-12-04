from optimize import loop
from vol_realized import rv_preaveraged 
from optimize import optimize_2param
loop(rv_preaveraged, optimize_2param, "garch_preaveraging", (1e-3,1e-3,0.), iter=50)