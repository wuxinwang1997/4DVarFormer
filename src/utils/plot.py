import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

cmap_z = 'cividis'
cmap_t = 'RdYlBu_r'
cmap_diff = 'bwr'
cmap_error = 'BrBG'

def imcol(ax, fig, data, title='', **kwargs):
    if not 'vmin' in kwargs.keys():
        mx = np.abs(data.max().values)
        kwargs['vmin'] = -mx; kwargs['vmax'] = mx
#     I = ax.imshow(data, origin='lower',  **kwargs)
    I = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False,
                  rasterized=True, **kwargs)
    cb = fig.colorbar(I, ax=ax, orientation='horizontal', pad=0.01, shrink=0.90)
    ax.set_title(title)
    ax.coastlines(alpha=0.5)

def subplot_increment(valid, xb_iter, fc_iter, methods, name, time=0, lead_time=12, idx=0):
    fig, axs = plt.subplots(len(methods), 6, figsize=(35, 5 * len(methods)), subplot_kw={'projection': ccrs.PlateCarree()})
    for i in range(len(methods)):
        # if methods[i] != '4DVar':
        for iax, var, cmap, r, t in zip(
            [i], ['z'], [cmap_z], [[47000, 58000]], [r'Z500 [m$^2$ s$^{-2}$]']):
            imcol(axs[iax,0],
                    fig,
                    valid[var].sel(time=time), cmap=cmap,
                    vmin=r[0], vmax=r[1], title=f'ERA5 {t} t={lead_time}h')
            imcol(axs[iax,1],
                    fig,
                    xb_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time), cmap=cmap,
                    vmin=r[0], vmax=r[1], title=f'Background {t} t={lead_time}h')
            error_b = xb_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time)-valid[var].sel(time=time)
            imcol(axs[iax,2],
                    fig,
                    error_b, cmap=cmap_diff,
                    title=f'Background - ERA5 {t} t={lead_time}h')
            imcol(axs[iax,3],
                    fig,
                    fc_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time), cmap=cmap,
                    vmin=r[0], vmax=r[1], title=f'Analysis of {methods[i]} {t} t={lead_time}h')
            error_a = fc_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time) - valid[var].sel(time=time)
            imcol(axs[iax, 4],
                    fig,
                    error_a, cmap=cmap_diff,
                    title=f'Analysis - ERA5 {t} t={lead_time}h')
            increment = fc_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time)-xb_iter[i][var].isel(init_time=idx).sel(lead_time=lead_time)
            imcol(axs[iax,5],
                    fig,
                    increment, cmap=cmap_error,
                    title=f'Increment of {methods[i]} {t} {lead_time}h')

    plt.savefig(f'{name}.jpg',dpi=300)

# 迭代预报的误差
def plot_iter_result(data, variable, x_label, y_label, title, unit_x, unit_y, name):
    plt.figure()
    plt.plot(data[x_label].values, data[variable].values) #, marker='.', markersize='5')
    plt.title(title)
    plt.xlabel(x_label + unit_x)
    plt.ylabel(y_label + unit_y)
    plt.savefig(f'{name}.jpg',dpi=300)
    plt.show()
#
def subplot_spinup(data, methods, obspartial, variable, x_label, y_label, title, unit_x, unit_y, spin_len, pred_len, x_tick_range, init_length, name):
    fig = plt.figure(figsize=(16, 10))
    pred_len = pred_len + spin_len
    for i in range(3*4):
        ax = fig.add_subplot(3, 4, i+1)
        for j in range(len(data)):
            if methods[j] != 'ERA5':
                # if methods[j] == 'CGAN':
                #     ax.plot(data[j][i][x_label][:pred_len//6+1], data[j][i][variable][init_length:pred_len//6+init_length+1], label=f'Pred w/ {methods[j]}')
                # else:
                ax.plot(data[j][i][x_label][2:pred_len//6+1], data[j][i][variable][2:pred_len//6+1], label=f'Pred w/ {methods[j]}')
            else:
                ax.plot(data[j][i//4][x_label][2:(pred_len-spin_len)//6+1], data[j][i//4][variable][2:(pred_len-spin_len)//6+1], label=f'Pred w/ {methods[j]}')
        y_min, y_max = [350, 0.3, 250], [1250, 1.0, 800]
        y_lim = [900, 0.7, 550]
        # 绘制RMSE和MAE
        if i == 0:
            ax.annotate('', xy=(pred_len,y_min[0]+y_lim[0]*0.1), xytext=(pred_len,y_min[0]+y_lim[0]*0.65), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-30, y_min[0]+y_lim[0]*0.15, 'lower is better', rotation=90, alpha=0.5)
        # 绘制ACC
        elif i == 4:
            ax.annotate('', xy=(pred_len,y_max[1]-y_lim[1]*0.15), xytext=(pred_len,y_min[1]+y_lim[1]*0.3), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-30, y_min[1]+y_lim[1]*0.35, 'higher is better', rotation=90, alpha=0.5)
        elif i == 8:
            ax.annotate('', xy=(pred_len,y_min[2]+y_lim[2]*0.1), xytext=(pred_len,y_min[2]+y_lim[2]*0.65), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-30, y_min[2]+y_lim[2]*0.15, 'lower is better', rotation=90, alpha=0.5)
        # 绘制标题，只显示在第一行
        if i in np.arange(4):
            ax.annotate('', xy=(spin_len, y_min[0] + y_lim[0] / 15), xytext=(0, y_min[0] + y_lim[0] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(8, y_min[0] + y_lim[0] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len - 15, y_min[0] + y_lim[0] / 15), xytext=(spin_len, y_min[0] + y_lim[0] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(160, y_min[0] + y_lim[0] / 10, 'Fourcast', alpha=0.5)
            plt.title(title + f' {obspartial[i % 4]} observation')
            ax.get_xaxis().set_visible(False)
            plt.ylim(y_min[0], y_max[0])
            ax.fill_betweenx(y=[0, y_max[0]], x1=spin_len, color='gray', alpha=0.1)
        elif i in np.arange(4, 8):
            # ax.axhline(0.65, linestyle='--', color='black', alpha=0.2)
            ax.annotate('', xy=(spin_len, y_min[1] + y_lim[1] / 15), xytext=(0, y_min[1] + y_lim[1] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(8, y_min[1] + y_lim[1] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len - 15, y_min[1] + y_lim[1] / 15), xytext=(spin_len, y_min[1] + y_lim[1] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(160, y_min[1] + y_lim[1] / 10, 'Fourcast', alpha=0.5)
            ax.set_ylim(0, y_max[1])
            ax.get_xaxis().set_visible(False)
            ax.fill_betweenx(y=[0, 1], x1=spin_len, color='gray', alpha=0.1)
            plt.ylim(y_min[1], 1)
        else:
            ax.annotate('', xy=(spin_len, y_min[2] + y_lim[2] / 15), xytext=(0, y_min[2] + y_lim[2] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(8, y_min[2] + y_lim[2] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len - 15, y_min[2] + y_lim[2] / 15), xytext=(spin_len, y_min[2] + y_lim[2] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(160, y_min[2] + y_lim[2] / 10, 'Fourcast', alpha=0.5)
            plt.xticks(np.arange(0, int(pred_len+24), x_tick_range * 24), np.arange(0,int(pred_len//24+1),x_tick_range))
            plt.ylim(y_min[2], y_max[2])
            ax.fill_betweenx(y=[0, y_max[2]], x1=spin_len, color='gray', alpha=0.1)
        if i in [0, 4, 8]:
            plt.ylabel(y_label[i%3] + unit_y[i%3])
        else:
            ax.get_yaxis().set_visible(False)
    # 绘制统一的图例
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0))
    fig.text(0.5, 0.075, x_label + unit_x, ha='center')
    plt.tight_layout()
    plt.savefig(f'{name}.jpg',dpi=300)
    plt.show()

def subplot_daloop(data, methods, obspartial, variable, x_label, y_label, title, unit_x, unit_y, spin_len, pred_len, x_tick_range, name):
    fig = plt.figure(figsize=(16, 20))
    pred_len = pred_len + spin_len
    for i in range(4*5):
        ax = fig.add_subplot(4, 5, i+1)
        for j in range(len(data)):
            ax.plot(data[j][i][x_label][4:], data[j][i][variable][4:], label=f'{methods[j]}')
        y_min, y_max = [300, 0.3, 200], [1100, 1.0, 700]
        y_lim = [800, 0.7, 500]
        # 绘制RMSE和MAE
        if i == 0:
            ax.annotate('', xy=(pred_len,y_min[0]+y_lim[0]*0.6), xytext=(pred_len,y_min[0]+y_lim[0]*1.0), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-120, y_min[0]+y_lim[0]*0.7, 'lower\nis better', rotation=90, alpha=0.5)
        # 绘制ACC
        elif i == 4:
            ax.annotate('', xy=(pred_len,y_max[1]-y_lim[1]*0.5), xytext=(pred_len,y_min[1]+y_lim[1]*0.1), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-120, y_min[1]+y_lim[1]*0.2, 'higher\nis better', rotation=90, alpha=0.5)
        elif i == 8:
            ax.annotate('', xy=(pred_len,y_min[2]+y_lim[2]*0.6), xytext=(pred_len,y_min[2]+y_lim[2]*1.0), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
            ax.text(pred_len-120, y_min[2]+y_lim[2]*0.7, 'lower\nis better', rotation=90, alpha=0.5)
        # 绘制标题，只显示在第一行
        if i in np.arange(4):
            # ax.annotate('', xy=(spin_len, y_min[0] + y_lim[0] / 15), xytext=(0, y_min[0] + y_lim[0] / 15),
            #             arrowprops=dict(color='gray', arrowstyle='->'))
            # ax.text(8, y_min[0] + y_lim[0] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len, y_min[0] + y_lim[0] / 15), xytext=(0, y_min[0] + y_lim[0] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(100, y_min[0] + y_lim[0] / 10, 'Assimilation Cycle', alpha=0.5)
            plt.title(title + f' {obspartial[i % 4]} observation')
            ax.get_xaxis().set_visible(False)
            plt.ylim(y_min[0], y_max[0])
            ax.fill_betweenx(y=[0, y_max[0]], x1=spin_len, color='gray', alpha=0.1)
        elif i in np.arange(4, 8):
            # ax.axhline(0.65, linestyle='--', color='black', alpha=0.2)
            # ax.annotate('', xy=(spin_len, y_min[1] + y_lim[1] / 15), xytext=(0, y_min[1] + y_lim[1] / 15),
            #             arrowprops=dict(color='gray', arrowstyle='->'))
            # ax.text(8, y_min[1] + y_lim[1] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len, y_min[1] + y_lim[1] / 15), xytext=(0, y_min[1] + y_lim[1] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(100, y_min[1] + y_lim[1] / 10, 'Assimilation Cycle', alpha=0.5)
            ax.set_ylim(0, y_max[1])
            ax.get_xaxis().set_visible(False)
            ax.fill_betweenx(y=[0, 1], x1=spin_len, color='gray', alpha=0.1)
            plt.ylim(y_min[1], y_max[1])
        else:
            # ax.annotate('', xy=(spin_len, y_min[2] + y_lim[2] / 15), xytext=(0, y_min[2] + y_lim[2] / 15),
            #             arrowprops=dict(color='gray', arrowstyle='->'))
            # ax.text(8, y_min[2] + y_lim[2] / 10, 'Spin Up', alpha=0.5)
            ax.annotate('', xy=(pred_len, y_min[2] + y_lim[2] / 15), xytext=(0, y_min[2] + y_lim[2] / 15),
                        arrowprops=dict(color='gray', arrowstyle='->'))
            ax.text(100, y_min[2] + y_lim[2] / 10, 'Assimilation Cycle', alpha=0.5)
            plt.xticks(np.arange(0, int(pred_len+24), x_tick_range * 24), np.arange(0,int(pred_len//24+1),x_tick_range))
            plt.ylim(y_min[2], y_max[2])
            ax.fill_betweenx(y=[0, y_max[2]], x1=spin_len, color='gray', alpha=0.1)
        if i in [0, 4, 8]:
            plt.ylabel(y_label[i%3] + unit_y[i%3])
        else:
            ax.get_yaxis().set_visible(False)
    # 绘制统一的图例
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0))
    # fig.legend(lines, labels, ncol=1, loc='right center', bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, 0.075, x_label + unit_x, ha='center')
    plt.tight_layout()
    plt.savefig(f'{name}.jpg',dpi=300)
    plt.show()

# def subplot_daloop(data, methods, obspartial, variable, x_label, y_label, title, unit_x, unit_y, spin_len, pred_len, x_tick_range, name):
#     fig = plt.figure(figsize=(16, 10))
#     pred_len = pred_len + spin_len
#     for i in range(4*5):
#         ax = fig.add_subplot(4, 5, i+1)
#         for j in range(len(data)):
#             ax.plot(data[j][i][x_label][4:], data[j][i][variable][4:], label=f'Pred w/ {methods[j]}')
#         y_min, y_max = [300, 0.3, 200], [1100, 1.0, 700]
#         y_lim = [800, 0.7, 500]
#         # 绘制RMSE和MAE
#         if i == 0:
#             ax.annotate('', xy=(pred_len,y_min[0]+y_lim[0]*0.6), xytext=(pred_len,y_min[0]+y_lim[0]*1.0), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
#             ax.text(pred_len-120, y_min[0]+y_lim[0]*0.7, 'lower\nis better', rotation=90, alpha=0.5)
#         # 绘制ACC
#         elif i == 4:
#             ax.annotate('', xy=(pred_len,y_max[1]-y_lim[1]*0.5), xytext=(pred_len,y_min[1]+y_lim[1]*0.1), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
#             ax.text(pred_len-120, y_min[1]+y_lim[1]*0.2, 'higher\nis better', rotation=90, alpha=0.5)
#         elif i == 8:
#             ax.annotate('', xy=(pred_len,y_min[2]+y_lim[2]*0.6), xytext=(pred_len,y_min[2]+y_lim[2]*1.0), arrowprops=dict(color='gray', arrowstyle='->'), rotation=90)
#             ax.text(pred_len-120, y_min[2]+y_lim[2]*0.7, 'lower\nis better', rotation=90, alpha=0.5)
#         # 绘制标题，只显示在第一行
#         if i in np.arange(4):
#             # ax.annotate('', xy=(spin_len, y_min[0] + y_lim[0] / 15), xytext=(0, y_min[0] + y_lim[0] / 15),
#             #             arrowprops=dict(color='gray', arrowstyle='->'))
#             # ax.text(8, y_min[0] + y_lim[0] / 10, 'Spin Up', alpha=0.5)
#             ax.annotate('', xy=(pred_len, y_min[0] + y_lim[0] / 15), xytext=(0, y_min[0] + y_lim[0] / 15),
#                         arrowprops=dict(color='gray', arrowstyle='->'))
#             ax.text(100, y_min[0] + y_lim[0] / 10, 'Assimilation Cycle', alpha=0.5)
#             plt.title(title + f' {obspartial[i % 4]} observation')
#             ax.get_xaxis().set_visible(False)
#             plt.ylim(y_min[0], y_max[0])
#             ax.fill_betweenx(y=[0, y_max[0]], x1=spin_len, color='gray', alpha=0.1)
#         elif i in np.arange(4, 8):
#             # ax.axhline(0.65, linestyle='--', color='black', alpha=0.2)
#             # ax.annotate('', xy=(spin_len, y_min[1] + y_lim[1] / 15), xytext=(0, y_min[1] + y_lim[1] / 15),
#             #             arrowprops=dict(color='gray', arrowstyle='->'))
#             # ax.text(8, y_min[1] + y_lim[1] / 10, 'Spin Up', alpha=0.5)
#             ax.annotate('', xy=(pred_len, y_min[1] + y_lim[1] / 15), xytext=(0, y_min[1] + y_lim[1] / 15),
#                         arrowprops=dict(color='gray', arrowstyle='->'))
#             ax.text(100, y_min[1] + y_lim[1] / 10, 'Assimilation Cycle', alpha=0.5)
#             ax.set_ylim(0, y_max[1])
#             ax.get_xaxis().set_visible(False)
#             ax.fill_betweenx(y=[0, 1], x1=spin_len, color='gray', alpha=0.1)
#             plt.ylim(y_min[1], y_max[1])
#         else:
#             # ax.annotate('', xy=(spin_len, y_min[2] + y_lim[2] / 15), xytext=(0, y_min[2] + y_lim[2] / 15),
#             #             arrowprops=dict(color='gray', arrowstyle='->'))
#             # ax.text(8, y_min[2] + y_lim[2] / 10, 'Spin Up', alpha=0.5)
#             ax.annotate('', xy=(pred_len, y_min[2] + y_lim[2] / 15), xytext=(0, y_min[2] + y_lim[2] / 15),
#                         arrowprops=dict(color='gray', arrowstyle='->'))
#             ax.text(100, y_min[2] + y_lim[2] / 10, 'Assimilation Cycle', alpha=0.5)
#             plt.xticks(np.arange(0, int(pred_len+24), x_tick_range * 24), np.arange(0,int(pred_len//24+1),x_tick_range))
#             plt.ylim(y_min[2], y_max[2])
#             ax.fill_betweenx(y=[0, y_max[2]], x1=spin_len, color='gray', alpha=0.1)
#         if i in [0, 4, 8]:
#             plt.ylabel(y_label[i%3] + unit_y[i%3])
#         else:
#             ax.get_yaxis().set_visible(False)
#     # 绘制统一的图例
#     lines, labels = fig.axes[-1].get_legend_handles_labels()
#     fig.legend(lines, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0))
#     # fig.legend(lines, labels, ncol=1, loc='right center', bbox_to_anchor=(0.5, -0.05))
#     fig.text(0.5, 0.075, x_label + unit_x, ha='center')
#     plt.tight_layout()
#     plt.savefig(f'{name}.jpg',dpi=300)
#     plt.show()

# def plot_iter_result(var4d, cvae_product, cvae_fill, variable, x_label, y_label, title, unit_x, unit_y, name):
#     plt.figure()
#     plt.plot(var4d[x_label].values[4:], var4d[variable].values[4:], label='4DVar') #, marker='.', markersize='5')
#     plt.plot(cvae_product[x_label].values[4:], cvae_product[variable].values[4:], label='CVAE_P')
#     plt.plot(cvae_fill[x_label].values[4:], cvae_fill[variable].values[4:], label='CVAE_F')
#     plt.title(title)
#     plt.xlabel(x_label + unit_x)
#     plt.ylabel(y_label + unit_y)
#     plt.legend()
#     plt.savefig(f'{name}.jpg',dpi=300)
#     plt.show()

# 误差带的绘制代码
def plot_results(data, mode, x_label, y_label, title, unit, rmses):
    plt.figure()
    if mode == 'single':
        plt.plot(data[x_label].values, data['mean'].values, marker='.', markersize='5')
    elif mode == 'ensemble':
        plt.plot(data[x_label].values, data['mean'].values, marker='.',  markersize='5', label='mean')
        plt.fill_between(
            rmses[x_label].values,
            rmses['mean'].values+rmses['std'].values,
            rmses['mean'].values-rmses['std'].values,
            alpha=0.3,
            label='std'
        )
    plt.title(title)
    plt.ylabel(y_label + unit)
    plt.xlabel(x_label)
    plt.show()

def plot_metrics(rmses, accs):
    fig = plt.figure(figsize=(16, 9))
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        if i in [0, 1, 2, 3, 4]:
            plt.title(plot_vars[i])
            ax.plot(np.mean(rmse_fourcastnet, axis=0)[:, plot_idx[i]], color='k', marker="o", markersize=2,
                    label=f"FourcastNet")
            ax.set_xticks(np.arange(0, rmse_fourcastnet.shape[1], 8))
            ax.set_xticklabels(np.arange(0, 6 * rmse_fourcastnet.shape[1], 48))
            plt.ylabel(f"RMSE {unit_y[i]}")
            plt.xlabel("Lead Time (hours)")
        elif i in [5, 6, 7, 8, 9]:
            plt.title(plot_vars[i - 5])
            ax.plot(np.mean(acc_fourcastnet, axis=0)[:, plot_idx[i - 5]], color='k', marker="o", markersize=2,
                    label=f"FourcastNet")
            ax.set_xticks(np.arange(0, acc_fourcastnet.shape[1], 8))
            ax.set_xticklabels(np.arange(0, 6 * acc_fourcastnet.shape[1], 48))
            plt.ylabel(f"ACC")
            plt.xlabel("Lead Time (hours)")
        elif i in [10, 11, 12, 13, 14]:
            plt.title(plot_vars[i - 5])
            ax.plot(np.mean(rmse_fourcastnet, axis=0)[:, plot_idx[i - 5]], color='k', marker="o", markersize=2,
                    label=f"FourcastNet")
            ax.set_xticks(np.arange(0, rmse_fourcastnet.shape[1], 8))
            ax.set_xticklabels(np.arange(0, 6 * rmse_fourcastnet.shape[1], 48))
            plt.ylabel(f"RMSE {unit_y[i - 5]}")
            plt.xlabel("Lead Time (hours)")
        elif i in [15, 16, 17, 18, 19]:
            plt.title(plot_vars[i - 10])
            ax.plot(np.mean(acc_fourcastnet, axis=0)[:, plot_idx[i - 10]], color='k', marker="o", markersize=2,
                    label=f"FourcastNet")
            ax.set_xticks(np.arange(0, acc_fourcastnet.shape[1], 8))
            ax.set_xticklabels(np.arange(0, 6 * acc_fourcastnet.shape[1], 48))
            plt.ylabel(f"ACC")
            plt.xlabel("Lead Time (hours)")

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0))
    plt.tight_layout()
    plt.gcf().subplots_adjust(top=0.08)
    plt.savefig(f'fourcastnet_medium_forecast.jpg', dpi=300)
    plt.savefig(f'fourcastnet_medium_forecast.pdf', dpi=300)
    plt.show()