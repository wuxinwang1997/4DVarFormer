import numpy as np
import netCDF4 as nc
import datetime
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader

def nearest_position(stn_lat, stn_lon, lat2d, lon2d):
    """
    获取最临近个点坐标索引
    :param stn_lat: 站点纬度
    :param stn_lon: 站点经度
    :param lat2d: numpy.ndarray 网格二维纬度坐标
    :param lon2d: numpy.ndarray 网格二维经度坐标
    :return: y_index, x_index
    """
    difflat = stn_lat - lat2d
    difflon = stn_lon - lon2d
    rad = np.multiply(difflat, difflat) + np.multiply(difflon, difflon)
    aa = np.where(rad==np.min(rad))
    ind = np.squeeze(np.array(aa))
    return list(ind)

def tc_track(idx, lat_ini, lon_ini, lat2d, lon2d, times, fields, speed):
    nt = np.min([len(times), len(fields["time"])])
    lon2d, lat2d = np.meshgrid(lon2d, lat2d)
    ny, nx = np.shape(lat2d)

    date0 = datetime.datetime.strptime(str(times[0])[2:-1], "%Y-%m-%d %H:%M:%S")
    date1 = datetime.datetime.strptime(str(times[1])[2:-1], "%Y-%m-%d %H:%M:%S")
    history_interval = (date1 - date0).total_seconds() / 3600

    lonMax = np.max(lon2d)
    lonMin = np.min(lon2d)
    latMax = np.max(lat2d)
    latMin = np.min(lat2d)

    date = []
    lons = []
    lats = []
    pmin = []

    latTc = lat_ini
    lonTc = lon_ini

    for it in range(nt):
        slp = fields.sel(var="msl").isel(start_time=idx).isel(time=it)["tc_pred"].to_numpy()
        if it == 0:
            if lat_ini is not None and lat_ini > - 60:
                latTc = lat_ini
                lonTc = lon_ini
            else:
                slpMin = np.min(slp)
                indexMin = np.argwhere(slp==slpMin)
                jTc = indexMin[0][0]
                iTc = indexMin[0][1]
                lonTc = lon2d[jTc, iTc]
                latTc = lat2d[jTc, iTc]

        ### 1 找到TC中心(lonTc, latTc)的索引(iTc, jTc)
        indexTc = nearest_position(latTc, lonTc, lat2d, lon2d)
        jTc = int(indexTc[0])
        iTc = int(indexTc[1])
        # 避免台风中心选在边界点
        jTc = np.max((1, jTc))
        jTc = np.min((jTc, ny-2))
        iTc = np.max((1, iTc))
        iTc = np.min((iTc, nx-2))

        ### 2 计算TC center附近的网格分辨率dAvg
        dLat = lat2d[jTc, iTc] - lat2d[jTc+1, iTc]
        dLon = lon2d[jTc, iTc+1] - lon2d[jTc, iTc]
        dAvg = (dLat + dLon) / 2

        ### 根据移速计算台风中心最大可能半径
        if latTc < 30:
            radius = speed * history_interval # 0.5 degree / hour
        else:
            radius = 2 * speed * history_interval
        if it == 0:
            radius = speed
        indexRadius = int(radius / dAvg) + 1

        ### 找到最大可能半径内，slp最小值及其位置索引
        iStart = iTc - indexRadius
        iEnd = iTc + indexRadius
        jStart = jTc - indexRadius
        jEnd = jTc + indexRadius
        jStart = np.max([1, jStart])
        jEnd = np.min([jEnd, ny-2])
        iStart = np.max([1, iStart])
        iEnd = np.min([iEnd, ny - 2])

        slpMin = np.min(slp[jStart:jEnd, iStart:iEnd])
        indexMin = np.argwhere(slp[jStart:jEnd, iStart:iEnd]==slpMin)
        jTc = indexMin[0][0] + jStart
        iTc = indexMin[0][1] + iStart
        lonTc = lon2d[jTc, iTc]
        latTc = lat2d[jTc, iTc]
        print("date: ", str(times[it])[2:-1], "TC center: ", round(lonTc, 2), round(latTc, 2))
        date.append(str(times[it])[2:-1])
        lons.append(round(lonTc, 2))
        lats.append(round(latTc, 2))

    return lons, lats