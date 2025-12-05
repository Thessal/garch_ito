import numpy as np 
import pandas as pd 
import torch
from data import get_data
from vol_est import vol_est_arr
from vol_realized import rv_preaveraged, rv_daily

def likelihood(vol_est_arr, rv_arr, backend=torch):
    # Likelihood function for estimated == realized
    ll1 = backend.log(vol_est_arr) + rv_arr / vol_est_arr
    # prevent divergence in vol < epsilon
    epsilon = 1e-12
    # ll2 = (1 / epsilon - rv_arr / epsilon / epsilon) * vol_est_arr + (1 + rv_arr / epsilon)
    ll2 = (1 / epsilon - rv_arr / epsilon / epsilon) * vol_est_arr + (float(np.log(epsilon)) + rv_arr / epsilon)
    ll = backend.where(vol_est_arr>epsilon, ll1, ll2)
    return -backend.sum(ll)

def optimize_2param(r2_arr_fitting, r2_arr_pred, init_params=(1e-3,1e-3,0.0), iter=1000, device=torch.device("cpu")):
    # Optimze gamma and beta. set omega = 0
    backend=torch
    vol_coef, rv_coef, resid = [torch.tensor(x, requires_grad=True) for x in init_params]
    resid = torch.tensor(0., requires_grad=True)
    # opt = torch.optim.Adam([vol_coef, rv_coef, resid], lr=1e-3)
    opt = torch.optim.Adam([vol_coef, rv_coef], lr=1e-2)
    for epoch in range(iter):
        opt.zero_grad()
        historical_est, next_est = vol_est_arr(r2_arr_pred, vol_coef, rv_coef, resid, backend=backend, device=device) # It looks like a RNN 
        loss = -likelihood(historical_est, r2_arr_fitting, backend=backend)
        loss.backward()
        opt.step()
        #print(f"[{epoch}] loss:{loss.item():.2e}, params:{vol_coef.item():.2e}, {rv_coef.item():.2e}, {resid.item():.2e}, est:{next_est.item():.2e}")
    params = (vol_coef.item(), rv_coef.item(), resid.item())
    params_log = {"gamma":vol_coef.item(), "beta_g":rv_coef.item(), "omega_g":resid.item()}
    return params, params_log, next_est.item(), loss.item()

def optimize_3param(r2_arr_fitting, r2_arr_pred, init_params=(0.5,0.5,1e-3), iter=1000, device=torch.device("cpu")):
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
        historical_est, next_est = vol_est_arr(r2_arr_pred, vol_coef, rv_coef, resid, backend=backend, device=device) # It looks like a RNN 
        loss = -likelihood(historical_est, r2_arr_fitting, backend=backend)
        loss.backward()
        opt.step()
        # print(f"[{epoch}] loss:{loss.item():.2e}, params:{vol_coef.item():.2e}, {rv_coef.item():.2e}, {resid.item():.2e}, est:{next_est.item():.2e}")
    params = (_omega.item(), _beta.item(), _gamma.item())
    params_log = {"gamma":vol_coef.item(), "beta_g":rv_coef.item(), "omega_g":resid.item()}
    return params, params_log, next_est.item(), loss.item()


def optimize_4param(r2_arr_fitting, r2_arr_pred, init_params=(0.1,0,0,0), iter=1000, device=torch.device("cpu")):
    # Optimze omega, alpha, gamma, nu
    backend=torch
    # https://arxiv.org/pdf/1907.01175
    _omega, _alpha, _gamma, _nu = [torch.tensor(x, requires_grad=True) for x in init_params]
    opt = torch.optim.Adagrad([_omega, _alpha, _gamma, _nu], lr=1e-2)
    for epoch in range(iter):
        opt.zero_grad()
        # Page 7
        omega = torch.relu(_omega)
        alpha = torch.sigmoid(_alpha)
        gamma = torch.sigmoid(_gamma)
        nu =  torch.sigmoid(_nu)

        exp_alpha = torch.exp(alpha)    
        rho1 = (exp_alpha - 1.) / alpha
        rho2 = (exp_alpha - 1. - alpha) / alpha / alpha
        rho3 = (exp_alpha - 1. - alpha - alpha * alpha * 0.5) / (alpha*alpha*alpha)
        
        alpha_g = (rho1 - rho2 + 2. * gamma * rho3) * alpha
        # beta_g = 0
        og1 = (rho1 - rho2 + 2. * rho3) * omega # omega = gamma omega1 - omega2
        og2 = (1. - gamma) * (rho2 - 2. * rho3) * nu 
        omega_g = og1 + og2

        vol_coef = gamma
        rv_coef = alpha_g
        resid = omega_g
        historical_est, next_est = vol_est_arr(r2_arr_pred, vol_coef, rv_coef, resid, backend=backend, device=device) # It looks like a RNN 
        loss = -likelihood(historical_est, r2_arr_fitting, backend=backend)
        loss.backward()
        opt.step()
        # print(_omega.item(), _alpha.item(), _gamma.item(), _nu.item())
        # print(f"[{epoch}] loss:{loss.item():.2e}, params:{vol_coef.item():.2e}, {rv_coef.item():.2e}, {resid.item():.2e}, est:{next_est.item():.2e}")
    params = (_omega.item(), _alpha.item(), _gamma.item(), _nu.item())
    params_log = {"gamma":vol_coef.item(), "beta_g":rv_coef.item(), "omega_g":resid.item()}
    return params, params_log, next_est.item(), loss.item()


def loop(rv_estimate_fn_fitting, rv_estimate_fn_pred, optimize_fn, save_name, initial_param, iter=200):
    # rv_estimate_fn_fitting # rv used for QMLE
    # rv_estimate_fn_pred # rv used for GARCH prediction
    backend = torch
    device = torch.device("cpu")
    df = get_data().loc["2023-11":"2025-11"]
    rv_arr_all_PRV = rv_preaveraged(df, backend=backend, device=device)
    rv_arr_all_DailyRV = rv_daily(df, backend=backend, device=device)
    rvs = {rv_preaveraged:rv_arr_all_PRV, rv_daily:rv_arr_all_DailyRV}
    rv_arr_all_fitting = rvs[rv_estimate_fn_fitting]
    rv_arr_all_pred = rvs[rv_estimate_fn_pred]
    est_rv = dict()
    lookback = 500
    params = initial_param
    for i in range(lookback,len(rv_arr_all_fitting)):
        rv_arr_fitting = rv_arr_all_fitting[i-lookback:i]
        rv_arr_pred = rv_arr_all_pred[i-lookback:i]
        params, params_log, vol_pred, loss = optimize_fn(rv_arr_fitting, rv_arr_pred, init_params=params, device=device, iter=iter)
        est_rv[i] = params_log
        est_rv[i].update({"vol_pred":vol_pred, "vol_true_PRV":rv_arr_all_PRV[i].item(), "vol_true_DailyRV":rv_arr_all_DailyRV[i].item(), "loss":loss})
        # print(f"prv:{rv_arr_fitting[-1]}, drv:{rv_arr_pred[-1]}") # 1250
        print(" ".join([f"{k}:{v:.2e}" for k,v in est_rv[i].items()]))
        if i%10 == 0:
            result = pd.DataFrame(est_rv).T
            result.to_csv(f"result_{save_name}.csv")
    result = pd.DataFrame(est_rv).T
    result.to_csv(f"result_{save_name}.csv")