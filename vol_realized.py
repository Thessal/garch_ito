import torch
import numpy as np

# High-frequency or Daily volatility calculation


def rv_daily(df, backend=torch, device="cpu"):
    # Daily realized volatility for standard GARCH(1,1) model
    x_d = df["close"].resample("1D").last()
    xi_d = np.log(x_d).diff()

    # Detrend
    xi_d = xi_d - xi_d.mean()
    xi_sq = np.square(xi_d.values[1:])

    if backend == torch:
        return torch.Tensor(xi_sq).to(device)
    elif backend == np:
        return xi_sq


def rv_naive(df, backend=torch, device="cpu"):
    # RV estimation without pre-averaging
    logret = np.log(df["close"]).diff()
    logret = logret - logret.mean()
    logret = logret.resample("1D").transform(
        lambda x: np.clip(x-x.mean(), -x.std(), x.std()))
    rv = logret.pow(2).resample("1D").mean()
    # # adjust to daily vol
    # N = 60*60*24
    # rv = rv.multiply(N)
    rv = rv.values

    if backend == torch:
        return torch.Tensor(rv).to(device)
    elif backend == np:
        return rv


def rv_preaveraged(df, backend=torch, device="cpu"):
    # Preaveraged RV

    # Detrend
    logprc = np.log(df["close"])
    logret = logprc.diff()
    logret = logret - logret.mean()
    # logret = logret.resample("1D").transform(lambda x: np.clip(x-x.mean(),-x.std(),x.std()))

    eta_hat = logret.pow(2).resample("1D").mean()*0.5
    N = 60*60*24
    K = int(np.sqrt(N))
    weight_function = np.minimum(np.linspace(0, 1, K), np.linspace(1, 0, K))

    def convolve(x, g):
        assert len(x) > len(g)
        result = np.convolve(x, weight_function, "same")
        result[:len(g)] = np.nan
        result[-len(g):] = np.nan
        return result
    logprc_avg = logprc.resample("1D").transform(
        lambda x: convolve(x, weight_function))
    _N_K_1 = logprc_avg.resample("1D").apply(len)
    ybar = logprc_avg.diff()
    zeta = 1 / K
    psi = 1/12
    prv = ((ybar.pow(2)).resample("1D").sum() - zeta * eta_hat * _N_K_1)/K/psi

    prv = prv.values
    if backend == torch:
        return torch.Tensor(prv).to(device)
    elif backend == np:
        return prv
