import numpy as np 
import pandas as pd 
import torch
from data import get_data
from util import vol_est_arr, likelihood

def optimize_2param(rv_arr, init_params=(1e-3,1e-3,0.0), iter=1000, device=torch.device("cpu")):
    # Optimze gamma and beta. set omega = 0
    backend=torch
    vol_coef, rv_coef, resid = [torch.tensor(x, requires_grad=True) for x in init_params]
    resid = torch.tensor(0., requires_grad=True)
    # opt = torch.optim.Adam([vol_coef, rv_coef, resid], lr=1e-3)
    opt = torch.optim.Adam([vol_coef, rv_coef], lr=1e-2)
    for epoch in range(iter):
        opt.zero_grad()
        historical_est, next_est = vol_est_arr(rv_arr, vol_coef, rv_coef, resid, backend=backend, device=device) # It looks like a RNN 
        loss = -likelihood(historical_est, rv_arr, backend=backend)
        loss.backward()
        opt.step()
        #print(f"[{epoch}] loss:{loss.item():.2e}, params:{vol_coef.item():.2e}, {rv_coef.item():.2e}, {resid.item():.2e}, est:{next_est.item():.2e}")
    params = (vol_coef.item(), rv_coef.item(), resid.item())
    params_log = {"gamma":vol_coef.item(), "beta_g":rv_coef.item(), "omega_g":resid.item()}
    return params, params_log, next_est.item(), loss.item()

def optimize_3param(rv_arr, init_params=(0,0.5,1e-3), iter=1000, device=torch.device("cpu")):
    # Optimze gamma, beta, and omega
    backend=torch
    _omega, _beta, _gamma = [torch.tensor(x, requires_grad=True) for x in init_params]
    opt = torch.optim.Adagrad([_omega, _beta, _gamma], lr=1e-2)
    for epoch in range(iter):
        opt.zero_grad()
        gamma = torch.sigmoid(_gamma)
        beta = torch.sigmoid(_beta)
        omega = torch.relu(_omega)
        vol_coef = gamma
        rv_coef = (gamma-1)/beta*(torch.exp(beta)-1-beta)+torch.exp(beta)-1
        resid = (torch.exp(beta)-1)*torch.square(omega)/beta
        historical_est, next_est = vol_est_arr(rv_arr, vol_coef, rv_coef, resid, backend=backend, device=device) # It looks like a RNN 
        loss = -likelihood(historical_est, rv_arr, backend=backend)
        loss.backward()
        opt.step()
        # print(f"[{epoch}] loss:{loss.item():.2e}, params:{vol_coef.item():.2e}, {rv_coef.item():.2e}, {resid.item():.2e}, est:{next_est.item():.2e}")
    params = (_omega.item(), _beta.item(), _gamma.item())
    params_log = {"gamma":vol_coef.item(), "beta_g":rv_coef.item(), "omega_g":resid.item()}
    return params, params_log, next_est.item(), loss.item()

def loop(rv_estimate_fn, optimize_fn, save_name, initial_param, iter=200):
    backend = torch
    device = torch.device("cpu")
    df = get_data()
    rv_arr_all = rv_estimate_fn(df, backend=backend, device=device)
    est_rv = dict()
    lookback = 250
    params = initial_param
    for i in range(lookback,len(rv_arr_all)):
        rv_arr = rv_arr_all[i-lookback:i]
        params, params_log, vol_pred, loss = optimize_fn(rv_arr, init_params=params, device=device, iter=iter)
        est_rv[i] = params_log
        est_rv[i].update({"vol_pred":vol_pred, "vol_real":rv_arr_all[i].item(), "loss":loss})
        print(" ".join([f"{k}:{v:.2e}" for k,v in est_rv[i].items()]))
        if i%10 == 0:
            result = pd.DataFrame(est_rv).T
            result.to_csv(f"result_{save_name}.csv")
    result = pd.DataFrame(est_rv).T
    result.to_csv(f"result_{save_name}.csv")