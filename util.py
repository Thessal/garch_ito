import torch
import numpy as np 

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

def likelihood(vol_est_arr, rv_arr, backend=torch):
    # Likelihood function for estimated == realized
    ll1 = backend.log(vol_est_arr) + rv_arr / vol_est_arr
    # prevent divergence in vol < epsilon
    epsilon = 1e-12
    ll2 = (1 / epsilon - rv_arr / epsilon / epsilon) * vol_est_arr + (float(np.log(epsilon)) + rv_arr / epsilon)
    ll = backend.where(vol_est_arr>epsilon, ll1, ll2)
    return -backend.sum(ll)

def rv_daily(df, backend=torch, device="cpu"):
    # Daily realized volatility for standard GARCH(1,1) model 
    x_d = df["close"].resample("1D").last()
    xi_d = np.log(x_d).diff()
    xi_d = xi_d - xi_d.mean()
    xi_sq = np.square(xi_d.values[1:])
    if backend==torch:
        return torch.Tensor(xi_sq).to(device)
    elif backend==np:
        return xi_sq

def rv_naive(df, backend=torch, device="cpu"):
    # RV estimation without pre-averaging
    logret = np.log(df["close"]).diff()
    logret = logret - logret.mean()
    logret = logret.resample("1D").transform(lambda x: np.clip(x-x.mean(),-x.std(),x.std()))
    rv = logret.pow(2).resample("1D").mean()
    # adjust to daily vol
    N = 60*60*24
    rv = rv.multiply(N).values
    if backend==torch:
        return torch.Tensor(rv).to(device)
    elif backend==np:
        return rv

def rv_preaveraged(df, backend=torch, device="cpu"):
    # Preaveraged RV
    logprc = np.log(df["close"])
    logret = logprc.diff()
    logret = logret.resample("1D").transform(lambda x: np.clip(x-x.mean(),-x.std(),x.std()))
    eta_hat = logret.pow(2).resample("1D").mean()*0.5
    N = 60*60*24
    K = int(np.sqrt(N))
    weight_function = np.minimum(np.linspace(0,1,K),np.linspace(1,0,K))
    def convolve(x,g):
        assert len(x) > len(g)
        result = np.convolve(x, weight_function, "same")
        result[:len(g)] = np.nan
        result[-len(g):] = np.nan
        return result
    logprc_avg = logprc.resample("1D").transform(lambda x: convolve(x, weight_function))
    ybar = logprc_avg.diff()
    zeta = 1 / K 
    psi = 1/12
    prv = ((ybar.pow(2)).resample("1D").mean() - zeta * eta_hat)/psi
    # adjust to daily vol
    # prv = prv.multiply(float(N)).values
    prv = prv.values
    if backend==torch:
        return torch.Tensor(prv).to(device)
    elif backend==np:
        return prv
