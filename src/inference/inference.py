import torch
import numpy as np
import time
import xarray as xr
import torch
# from src.utils.tools import gaussian_perturb
from src.utils.weighted_acc_rmse import weighted_acc_torch, weighted_rmse_torch
from src.da_methods.var4d import Solve_Var4D

def autoregressive_inference(ic, val_dataset, module, dt, prediction_length, clim, mult, device):
    ic = int(ic)
    dt = dt
    prediction_length = int(prediction_length)
    clim = clim
    mult = mult

    seq_pred = torch.zeros((1 + prediction_length // dt, 24, 160, 160)).to(device, dtype=torch.float32)
    seq_real = torch.zeros((1 + prediction_length // dt, 24, 160, 160)).to(device, dtype=torch.float32)
    seq_rmse = torch.zeros((1 + prediction_length // dt, 24)).to(device, dtype=torch.float32)
    seq_acc = torch.zeros((1 + prediction_length // dt, 24)).to(device, dtype=torch.float32)
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+prediction_length//dt+1).tolist())
    # standardize
    init_data = torch.as_tensor(valid_data_all[0][0]).to(device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(1 + prediction_length // dt):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = init_data
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(valid_data_all[i][0])
                seq_pred[i:i+1] = module(seq_pred[i-1:i])  # autoregressive step
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i + 1], seq_pred[i:i + 1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i + 1] - clim, seq_pred[i:i + 1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

def assim_vit_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device):
    new_obs = torch.zeros_like(xb).to(device, dtype=torch.float32)
    new_obs = torch.stack([new_obs, new_obs, new_obs, new_obs], dim=1) # (1, 4, 24, 160, 160)
    for i in range(model.net.out_chans):
        if var_idx[i] == "u10":
            new_obs[:, :, i] = obs_wind[:, :, 0]
        elif var_idx[i] == "v10":
            new_obs[:, :, i] = obs_wind[:, :, 1]
        elif var_idx[i] == "r@50":
            new_obs[:, :, i] = obs_r[:, :, 0]
        elif var_idx[i] == "r@500":
            new_obs[:, :, i] = obs_r[:, :, 1]
        elif var_idx[i] == "r@850":
            new_obs[:, :, i] = obs_r[:, :, 2]
        elif var_idx[i] == "r@1000":
            new_obs[:, :, i] = obs_r[:, :, 3]
        elif var_idx[i] == "t@50":
            new_obs[:, :, i] = obs_t[:, :, 0]
        elif var_idx[i] == "t@500":
            new_obs[:, :, i] = obs_t[:, :, 1]
        elif var_idx[i] == "t@850":
            new_obs[:, :, i] = obs_t[:, :, 2]
        elif var_idx[i] == "t@1000":
            new_obs[:, :, i] = obs_t[:, :, 3]

    xa = model(xb.to(device, dtype=torch.float32),
               new_obs[:,0].detach() * mask[:,0].to(device, dtype=torch.float32).detach(),
               mask[:,0].to(device, dtype=torch.float32).detach())
    xa = xa.cpu().detach()
    torch.cuda.empty_cache()
    return xa

def assim_cycle_vit(ic, val_dataset, model, afnonet, dt, assim_length, clim, mult, mask, var_idx, device):
    ic = int(ic)
    dt = dt
    assim_length = int(assim_length)
    clim = clim  # (1, 24, 160, 160)
    mult = mult  # (1, 24, 1, 1)
    mask = mask  # (1, 4, 24, 1, 1)

    seq_pred = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_real = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_rmse = torch.zeros((assim_length // dt, 24))
    seq_acc = torch.zeros((assim_length // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+assim_length//dt+1).tolist())

    with torch.no_grad():
        for i in range(assim_length // dt):
            # 从ic开始
            xt, xb, obs_wind, obs_r, obs_t = valid_data_all[i]
            if i == 0:  # start of sequence
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_vit_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)
            elif i % 4 != 0:
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()  # autoregressive step
            else:
                seq_real[i:i + 1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i + 1] = assim_vit_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)

            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

def assim_one_step(xb, model, obs_wind, obs_r, obs_t,mask, var_idx, device):
    new_obs = torch.zeros_like(xb).to(device, dtype=torch.float32)
    new_obs = torch.stack([new_obs for i in range(obs_wind.shape[1])], dim=1) # (1, 4, 24, 160, 160)
    for i in range(model.net.phi_r.out_chans):
        if var_idx[i] == "u10":
            new_obs[:, :, i] = obs_wind[:, :, 0]
        elif var_idx[i] == "v10":
            new_obs[:, :, i] = obs_wind[:, :, 1]
        elif var_idx[i] == "r@50":
            new_obs[:, :, i] = obs_r[:, :, 0]
        elif var_idx[i] == "r@500":
            new_obs[:, :, i] = obs_r[:, :, 1]
        elif var_idx[i] == "r@850":
            new_obs[:, :, i] = obs_r[:, :, 2]
        elif var_idx[i] == "r@1000":
            new_obs[:, :, i] = obs_r[:, :, 3]
        elif var_idx[i] == "t@50":
            new_obs[:, :, i] = obs_t[:, :, 0]
        elif var_idx[i] == "t@500":
            new_obs[:, :, i] = obs_t[:, :, 1]
        elif var_idx[i] == "t@850":
            new_obs[:, :, i] = obs_t[:, :, 2]
        elif var_idx[i] == "t@1000":
            new_obs[:, :, i] = obs_t[:, :, 3]
    # need to evaluate grad/backward during the evaluation and training phase for phi_r
    with torch.set_grad_enabled(True):
        state = torch.autograd.Variable(xb, requires_grad=True)
        xa = model(state.to(device, dtype=torch.float32),
                    new_obs.detach() * mask.to(device, dtype=torch.float32).detach(),
                    mask.to(device, dtype=torch.float32).detach(),
                    model.mult)
        xa = xa.cpu().detach()
    torch.cuda.empty_cache()
    return xa

def assim_cycle_inference(ic, val_dataset, model, afnonet, dt, assim_length, clim, mult, mask, var_idx, device):
    ic = int(ic)
    dt = dt
    assim_length = int(assim_length)
    clim = clim  # (1, 24, 160, 160)
    mult = mult  # (1, 24, 1, 1)
    mask = mask  # (1, 4, 24, 1, 1)

    seq_pred = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_real = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_rmse = torch.zeros((assim_length // dt, 24))
    seq_acc = torch.zeros((assim_length // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+assim_length//dt+1).tolist())

    with torch.no_grad():
        for i in range(assim_length // dt):
            # 从ic开始
            xt, xb, obs_wind, obs_r, obs_t = valid_data_all[i]
            if i == 0:  # start of sequence
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_one_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)
            elif i % 4 != 0:
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()  # autoregressive step
            else:
                seq_real[i:i + 1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i + 1] = assim_one_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)

            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

def assim_cycle_inference_diffobs(ic, val_dataset, model, afnonet, dt, assim_length, clim, mult, mask, var_idx, obs_num, device):
    ic = int(ic)
    dt = dt
    assim_length = int(assim_length)
    clim = clim  # (1, 24, 160, 160)
    mult = mult  # (1, 24, 1, 1)
    mask = mask  # (1, 4, 24, 1, 1)

    seq_pred = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_real = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_rmse = torch.zeros((assim_length // dt, 24))
    seq_acc = torch.zeros((assim_length // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+assim_length//dt+1).tolist())

    with torch.no_grad():
        for i in range(assim_length // dt):
            # 从ic开始
            xt, xb, obs_wind, obs_r, obs_t = valid_data_all[i]
            if i == 0:  # start of sequence
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_one_step(xb, model, obs_wind[:,:obs_num], obs_r[:,:obs_num], obs_t[:,:obs_num], mask[:,:obs_num], var_idx, device)
            elif i % 4 != 0:
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()  # autoregressive step
            else:
                seq_real[i:i + 1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_one_step(xb, model, obs_wind[:,:obs_num], obs_r[:,:obs_num], obs_t[:,:obs_num], mask[:,:obs_num], var_idx, device)

            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)


def assim_4dvarnet_step(xb, model, obs_wind, obs_r, obs_t,mask, var_idx, device):
    new_obs = torch.zeros_like(xb).to(device, dtype=torch.float32)
    new_obs = torch.stack([new_obs, new_obs, new_obs, new_obs], dim=1) # (1, 4, 24, 160, 160)
    for i in range(model.net.phi_r.out_chans):
        if var_idx[i] == "u10":
            new_obs[:, :, i] = obs_wind[:, :, 0]
        elif var_idx[i] == "v10":
            new_obs[:, :, i] = obs_wind[:, :, 1]
        elif var_idx[i] == "r@50":
            new_obs[:, :, i] = obs_r[:, :, 0]
        elif var_idx[i] == "r@500":
            new_obs[:, :, i] = obs_r[:, :, 1]
        elif var_idx[i] == "r@850":
            new_obs[:, :, i] = obs_r[:, :, 2]
        elif var_idx[i] == "r@1000":
            new_obs[:, :, i] = obs_r[:, :, 3]
        elif var_idx[i] == "t@50":
            new_obs[:, :, i] = obs_t[:, :, 0]
        elif var_idx[i] == "t@500":
            new_obs[:, :, i] = obs_t[:, :, 1]
        elif var_idx[i] == "t@850":
            new_obs[:, :, i] = obs_t[:, :, 2]
        elif var_idx[i] == "t@1000":
            new_obs[:, :, i] = obs_t[:, :, 3]
    # need to evaluate grad/backward during the evaluation and training phase for phi_r
    with torch.set_grad_enabled(True):
        state = torch.autograd.Variable(xb, requires_grad=True)
        xa, hidden_new, cell_new, normgrad = model(state.to(device, dtype=torch.float32),
                                                   new_obs.detach() * mask.to(device, dtype=torch.float32).detach(),
                                                   mask.to(device, dtype=torch.float32).detach())
        xa = xa.cpu().detach()
    torch.cuda.empty_cache()
    return xa

def assim_cycle_4dvarnet(ic, val_dataset, model, afnonet, dt, assim_length, clim, mult, mask, var_idx, device):
    ic = int(ic)
    dt = dt
    assim_length = int(assim_length)
    clim = clim  # (1, 24, 160, 160)
    mult = mult  # (1, 24, 1, 1)
    mask = mask  # (1, 4, 24, 1, 1)

    seq_pred = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_real = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_rmse = torch.zeros((assim_length // dt, 24))
    seq_acc = torch.zeros((assim_length // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+assim_length//dt+1).tolist())

    with torch.no_grad():
        for i in range(assim_length // dt):
            # 从ic开始
            xt, xb, obs_wind, obs_r, obs_t = valid_data_all[i]
            if i == 0:  # start of sequence
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_4dvarnet_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)
            elif i % 4 != 0:
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()  # autoregressive step
            else:
                seq_real[i:i + 1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i + 1] = assim_4dvarnet_step(xb, model, obs_wind, obs_r, obs_t, mask, var_idx, device)

            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)


def assim_4dvar_step(xb, B_half, R_inv, maxIter,
                    afnonet, obs_wind, obs_r, obs_t,
                    mask, var_idx, mult, inf, device):
    new_obs = torch.zeros_like(xb).to(device, dtype=torch.float32)
    new_obs = torch.stack([new_obs, new_obs, new_obs, new_obs], dim=1) # (1, 4, 24, 160, 160)
    for i in range(xb.shape[1]):
        if var_idx[i] == "u10":
            new_obs[:, :, i] = obs_wind[:, :, 0]
        elif var_idx[i] == "v10":
            new_obs[:, :, i] = obs_wind[:, :, 1]
        elif var_idx[i] == "r@50":
            new_obs[:, :, i] = obs_r[:, :, 0]
        elif var_idx[i] == "r@500":
            new_obs[:, :, i] = obs_r[:, :, 1]
        elif var_idx[i] == "r@850":
            new_obs[:, :, i] = obs_r[:, :, 2]
        elif var_idx[i] == "r@1000":
            new_obs[:, :, i] = obs_r[:, :, 3]
        elif var_idx[i] == "t@50":
            new_obs[:, :, i] = obs_t[:, :, 0]
        elif var_idx[i] == "t@500":
            new_obs[:, :, i] = obs_t[:, :, 1]
        elif var_idx[i] == "t@850":
            new_obs[:, :, i] = obs_t[:, :, 2]
        elif var_idx[i] == "t@1000":
            new_obs[:, :, i] = obs_t[:, :, 3]
    # need to evaluate grad/backward during the evaluation and training phase for phi_r
    with torch.set_grad_enabled(True):
        state = torch.autograd.Variable(xb, requires_grad=True)
        xa = Solve_Var4D(state.to(device, dtype=torch.float32),
                        B_half,
                        R_inv,
                        maxIter,
                        afnonet,
                        new_obs.detach() * mask.to(device, dtype=torch.float32).detach(),
                        mask.to(device, dtype=torch.float32).detach(),
                        mult,
                        inf)
        xa = xa.cpu().detach()
    torch.cuda.empty_cache()
    return xa

def assim_cycle_4dvar(ic, val_dataset, B_half, R_inv, maxIter,
                      afnonet, dt, assim_length, clim, mult, inf,
                      mask, var_idx, device):
    ic = int(ic)
    dt = dt
    assim_length = int(assim_length)
    clim = clim  # (1, 24, 160, 160)
    mult = mult  # (1, 24, 1, 1)
    mask = mask  # (1, 4, 24, 1, 1)

    seq_pred = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_real = torch.zeros((assim_length // dt, 24, 160, 160))
    seq_rmse = torch.zeros((assim_length // dt, 24))
    seq_acc = torch.zeros((assim_length // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+assim_length//dt+1).tolist())

    with torch.no_grad():
        for i in range(assim_length // dt):
            # 从ic开始
            xt, xb, obs_wind, obs_r, obs_t = valid_data_all[i]
            if i == 0:  # start of sequence
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i+1] = assim_4dvar_step(xb, B_half, R_inv, maxIter,
                                                   afnonet, obs_wind, obs_r, obs_t,
                                                   mask, var_idx, mult, inf, device)
            elif i % 4 != 0:
                seq_real[i:i+1] = torch.as_tensor(xt[0:1])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()  # autoregressive step
            else:
                seq_real[i:i + 1] = torch.as_tensor(xt[0:1])  # (1, 24, 160, 160)
                xb = torch.unsqueeze(torch.as_tensor(xb), dim=0)  # (1, 24, 160, 160)
                obs_wind = torch.unsqueeze(torch.as_tensor(obs_wind), dim=0)  # (1, 4, 2, 160, 160)
                obs_r = torch.unsqueeze(torch.as_tensor(obs_r), dim=0)  # (1, 4, 4, 160, 160)
                obs_t = torch.unsqueeze(torch.as_tensor(obs_t), dim=0)  # (1, 4, 4, 160, 160)
                seq_pred[i:i + 1] = assim_4dvar_step(xb, B_half, R_inv, maxIter,
                                                   afnonet, obs_wind, obs_r, obs_t,
                                                    mask, var_idx, mult, inf, device)

            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i+1], seq_pred[i:i+1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i+1] - clim, seq_pred[i:i+1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

def var4d_medium_forecast(ic,
                        val_dataset,
                        B_half,
                        R_inv,
                        maxIter,
                        afnonet,
                        dt,
                        spin_up_length,
                        forecast_length,
                        clim,
                        mult,
                        mask,
                        var_idx,
                        device):
    ic = int(ic)
    dt = dt
    spin_up_length = int(spin_up_length)
    forecast_length = int(forecast_length)
    clim = clim
    mult = mult

    seq_pred = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_real = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_rmse = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    seq_acc = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+(spin_up_length+forecast_length)//dt).tolist())

    # standardize
    seq_real_, seq_assim_, seq_rmse_, seq_acc_ = assim_cycle_4dvar(ic,
                                                                    val_dataset,
                                                                    B_half,
                                                                    R_inv,
                                                                    maxIter,
                                                                    afnonet,
                                                                    dt,
                                                                    spin_up_length,
                                                                    clim,
                                                                    mult,
                                                                    mask,
                                                                    var_idx,
                                                                    device)

    seq_real[:seq_real_.shape[0]] = torch.as_tensor(seq_real_)
    seq_pred[:seq_assim_.shape[0]] = torch.as_tensor(seq_assim_)
    seq_rmse[:seq_rmse_.shape[0]] = torch.as_tensor(seq_rmse_)
    seq_acc[:seq_acc_.shape[0]] = torch.as_tensor(seq_acc_)

    init_data = torch.as_tensor(seq_assim_[-2:-1])

    with torch.no_grad():
        for i in range(spin_up_length // dt, (spin_up_length+forecast_length) // dt):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = init_data
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(valid_data_all[i][0][0])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i + 1], seq_pred[i:i + 1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i + 1] - clim, seq_pred[i:i + 1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

def medium_forecast(ic,
                    val_dataset,
                    da_method,
                    model,
                    afnonet,
                    dt,
                    spin_up_length,
                    forecast_length,
                    clim,
                    mult,
                    mask,
                    var_idx,
                    device):
    ic = int(ic)
    dt = dt
    spin_up_length = int(spin_up_length)
    forecast_length = int(forecast_length)
    clim = clim
    mult = mult

    seq_pred = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_real = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_rmse = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    seq_acc = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+(spin_up_length+forecast_length)//dt).tolist())

    # standardize
    if da_method == "4dvarformer":
        seq_real_, seq_assim_, seq_rmse_, seq_acc_ = assim_cycle_inference(ic,
                                                                       val_dataset,
                                                                       model,
                                                                       afnonet,
                                                                       dt,
                                                                       spin_up_length,
                                                                       clim,
                                                                       mult,
                                                                       mask,
                                                                       var_idx,
                                                                       device)
    elif da_method == "4dvarnet":
        seq_real_, seq_assim_, seq_rmse_, seq_acc_ = assim_cycle_4dvarnet(ic,
                                                                      val_dataset,
                                                                      model,
                                                                      afnonet,
                                                                      dt,
                                                                      spin_up_length,
                                                                      clim,
                                                                      mult,
                                                                      mask,
                                                                      var_idx,
                                                                      device)
    elif da_method == "vit":
        seq_real_, seq_assim_, seq_rmse_, seq_acc_ = assim_cycle_vit(ic,
                                                                     val_dataset,
                                                                     model,
                                                                     afnonet,
                                                                     dt,
                                                                     spin_up_length,
                                                                     clim,
                                                                     mult,
                                                                     mask,
                                                                     var_idx,
                                                                     device)

    seq_real[:seq_real_.shape[0]] = torch.as_tensor(seq_real_)
    seq_pred[:seq_assim_.shape[0]] = torch.as_tensor(seq_assim_)
    seq_rmse[:seq_rmse_.shape[0]] = torch.as_tensor(seq_rmse_)
    seq_acc[:seq_acc_.shape[0]] = torch.as_tensor(seq_acc_)

    init_data = torch.as_tensor(seq_assim_[-2:-1])

    with torch.no_grad():
        for i in range(spin_up_length // dt, (spin_up_length+forecast_length) // dt):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = init_data
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(valid_data_all[i][0][0])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i + 1], seq_pred[i:i + 1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i + 1] - clim, seq_pred[i:i + 1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)


def medium_forecast_diffobs(ic,
                            val_dataset,
                            da_method,
                            model,
                            afnonet,
                            dt,
                            spin_up_length,
                            forecast_length,
                            clim,
                            mult,
                            mask,
                            var_idx,
                            obs_num,
                            device):
    ic = int(ic)
    dt = dt
    spin_up_length = int(spin_up_length)
    forecast_length = int(forecast_length)
    clim = clim
    mult = mult

    seq_pred = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_real = torch.zeros(((spin_up_length+forecast_length) // dt, 24, 160, 160))
    seq_rmse = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    seq_acc = torch.zeros(((spin_up_length+forecast_length) // dt, 24))
    valid_data_all = val_dataset.__getitem__(np.arange(ic,ic+(spin_up_length+forecast_length)//dt).tolist())

    # standardize
    if da_method == "4dvarformer":
        seq_real_, seq_assim_, seq_rmse_, seq_acc_ = assim_cycle_inference_diffobs(ic,
                                                                                   val_dataset,
                                                                                   model,
                                                                                   afnonet,
                                                                                   dt,
                                                                                   spin_up_length,
                                                                                   clim,
                                                                                   mult,
                                                                                   mask,
                                                                                   var_idx,
                                                                                   obs_num,
                                                                                   device)

    seq_real[:seq_real_.shape[0]] = torch.as_tensor(seq_real_)
    seq_pred[:seq_assim_.shape[0]] = torch.as_tensor(seq_assim_)
    seq_rmse[:seq_rmse_.shape[0]] = torch.as_tensor(seq_rmse_)
    seq_acc[:seq_acc_.shape[0]] = torch.as_tensor(seq_acc_)

    init_data = torch.as_tensor(seq_assim_[-2:-1])

    with torch.no_grad():
        for i in range(spin_up_length // dt, (spin_up_length+forecast_length) // dt):
            # 从ic开始
            if i == 0:  # start of sequence
                seq_real[i:i + 1] = init_data
                seq_pred[i:i + 1] = init_data
            else:
                seq_real[i:i + 1] = torch.as_tensor(valid_data_all[i][0][0])
                seq_pred[i:i+1] = afnonet(seq_pred[i-1:i].to(device, dtype=torch.float32)).cpu().detach()
            seq_rmse[i:i + 1] = mult * weighted_rmse_torch(seq_real[i:i + 1], seq_pred[i:i + 1])
            seq_acc[i:i + 1] = weighted_acc_torch(seq_real[i:i + 1] - clim, seq_pred[i:i + 1] - clim)

    seq_pred = seq_pred.cpu().detach().numpy()
    seq_real = seq_real.cpu().detach().numpy()
    seq_rmse = seq_rmse.cpu().detach().numpy()
    seq_acc = seq_acc.cpu().detach().numpy()

    return np.expand_dims(seq_real, 0), np.expand_dims(seq_pred, 0), np.expand_dims(seq_rmse, 0), np.expand_dims(seq_acc, 0)

