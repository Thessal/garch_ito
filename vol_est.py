import torch
import numpy as np 

# GARCH model

def vol_est(vol_prev, rv, vol_coef, rv_coef, resid=0):
    # GARCH(1,1) information process
    # volatility at t := expectation of return^2(t) at (t-1)   
    return vol_coef*vol_prev + rv_coef*rv + resid

def vol_est_arr(rv_arr, vol_coef, rv_coef, resid, backend=torch, device="cpu"):
    # Chained estimation of GARCH(1,1)
    # backend can be torch or np
    result = [torch.Tensor([0]).to(device)]
    for i in range(len(rv_arr)):
        result.append(vol_est(result[i], rv_arr[i], vol_coef, rv_coef, resid))
    historical = backend.concat(result[:-1])
    historical = backend.maximum(historical, torch.tensor(1e-6).to(device)) # for stability
    estimation = result[-1]
    return historical, estimation