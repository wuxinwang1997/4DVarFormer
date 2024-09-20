import numpy as np
import torch
import copy

#### 4DVar implementation following Marcel Nonnenmacher's implementation ######
## Deep Emulators for Differentiation, Forecasting, and Parametrization in Earth Science Simulators

def loss_fun(u, dy, std, R_inv):
    R_inv = R_inv.to(u.device, dtype=u.dtype)
    var = (torch.unsqueeze(std, dim=0) ** 2).to(u.device, dtype=u.dtype)  # (24)
    loss_xb = torch.nansum(u ** 2, dim=(-2, -1))  # (B, 24)
    loss_xb = loss_xb * var  # (B, 24)
    loss_xb = torch.nansum(loss_xb)  # (B)

    loss_obs = torch.nansum(dy ** 2, dim=(-2, -1))  # (B, 4, 24)
    loss_obs = torch.nansum(loss_obs, dim=1)  # (B, 24)
    loss_obs = loss_obs * R_inv * var  # (B, 24)
    loss_obs = torch.nansum(loss_obs)  # (B)

    loss = (loss_xb + loss_obs) / 2

    return loss

def Solve_Var4D(x_init, B_half, R_inv, maxIter, model, obs, obs_masks, std, inf):
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache() 
    shape = x_init.shape
    x_init = x_init.detach()
    u = torch.zeros_like(x_init).to(x_init.device, x_init.dtype)
    u.requires_grad = True
    obs = obs_masks * obs
    # for i in range(len(B_half)):
    #     B_half[i] = torch.from_numpy(B_half[i]).to(x_init.device, x_init.dtype)

    optim = torch.optim.LBFGS(params=[u],
                            lr=1e0,
                            max_iter=50,
                            max_eval=None,
                            tolerance_grad=1e-7,
                            tolerance_change=1e-9,
                            history_size=100,
                            line_search_fn='strong_wolfe')

    def closure():
        optim.zero_grad()
        preds = []
        dx = torch.concat([torch.reshape(1 / B_half[0, i] * torch.unsqueeze(torch.flatten(u[0,i], start_dim=0), dim=-1), x_init[:,i:i+1].shape) for i in range(B_half.shape[-1])], dim=1)
        # dx = torch.concat([torch.reshape(B_half[i] @ torch.unsqueeze(torch.flatten(u[0,i], start_dim=0), dim=-1), x_init[:,i:i+1].shape) for i in range(len(B_half))], dim=1)
        preds.append(x_init + dx.reshape(x_init.shape))
        for i in np.arange(1, obs.shape[1]):
            preds.append(model(preds[i - 1]))
        preds = torch.stack(preds, dim=1)
        dy = (preds - obs) * obs_masks
        loss = loss_fun(u, dy, std, R_inv)
        loss.backward()
        print(loss.item())
        return loss

    for _ in range(maxIter):
        optim.step(closure)
        model.zero_grad()
    
    del optim
    del obs
    torch.cuda.empty_cache() 
    x_analysis = x_init + u.detach()
    torch.set_grad_enabled(False)
    return x_analysis.reshape(shape)