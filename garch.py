import numpy as np 
import pandas as pd 
import torch
from data import get_data
from util import rv_daily, optimize_2param

if __name__ == "__main__":
    backend = torch
    device = torch.device("cpu")
    df = get_data()
    rv_arr_all = rv_daily(df, backend=backend, device=device)
    est_rv = dict()
    lookback = 250
    params = (1e-3,1e-3,1e-3)
    for i in range(lookback,len(rv_arr_all)):
        rv_arr = rv_arr_all[i-lookback:i]
        params, vol_pred, loss = optimize_2param(rv_arr, init_params=params, device=device, iter=200)
        vol_coef, rv_coef, resid = params
        est_rv[i] = {"vol_coef":vol_coef, "rv_coef":rv_coef, "resid":resid, "vol_pred":vol_pred, "vol_real":rv_arr_all[i], "loss":loss}
        if i%10 == 0:
            result = pd.DataFrame(est_rv).T
            result.to_csv("result_garch_garch_daily.csv")
    result = pd.DataFrame(est_rv).T
    result.to_csv("result_garch_daily.csv")