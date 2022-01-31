


# %%
import numpy as np
import xarray as xr
import math
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import calendar
import os
import sys
from matplotlib.dates import drange
from numpy.lib.npyio import load

# ====================初期値===================
meanstart = 2010
meanend = 2019
# year = 2020
year, month, day = 2020, 10, 11
meanmonth = 10

meanlatrange = [-75,-40]
# meanprs = 2.6101572e+01
# meanprs = 5.6234131e+01
meanprs = 100
notUru = 2021 #うるう年以外なら何でもよい
movingDay = 7
grid = 5
stdTimes = 1
wn = 1
# ====================描画値===================
vector_scale = 8.0e+5
lim = 100
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-80,-30]
graphStartDate = [7,1]
graphEndtDate = [12,31]
schalerange = [-40000,140000]

# ====================定数=====================
halfmd = int((movingDay-1)/2)
yearnum = meanend-meanstart+1
defineYear = 2019 #うるう年ではない年（てきとう）
latIndex = ((np.array(meanlatrange) + 90 )/grid).astype(np.int32)
yarray = np.arange(meanlatrange[0],meanlatrange[1]+1,grid)
phiarray = np.radians(yarray)
cosphi = np.cos(phiarray)
# print(yarray)
# print(phiarray)
# print(cosphi)
meanyears = meanend - meanstart + 1
prsfile = f'./text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
try:
    prsInd = np.where(pcord==meanprs)[0][0]
except:
    print('Error')
    print('not exist prs number!!')
    sys.exit()
ycord = np.arange(-90, 90.1, 5)
xcord = np.arange(-180, 180.1, 5)
phicord = ycord*(math.pi/180.)
xr_cosphi = xr.DataArray(np.cos(phicord),dims=['j'],coords={'j':ycord})
a = 6.37e+6
R = 287
Cp = 1004
K = R/Cp
g0 = 9.80665
ps = 100000
omega = 7.29e-5
Ts =  240
H = R*Ts/g0
rhos = ps/R/Ts
f = 2*omega*np.sin(phicord)
xr_f = xr.DataArray(f,dims=['j'],coords={'j':ycord})
rho = rhos*(pcord*100/ps)
xr_rho = xr.DataArray(rho,dims=['k'],coords={'k':pcord})
N_2 = 4.0e-4
zcord = -H*np.log(pcord*100/ps)
# xr_z = xr.DataArray(z,dims=['k'],coords={'k':pcord})
namelist = ['T','U','V','GPH']
# kindlist = ['zonal','dev']
kind = 'dev'

# def makeEPflux(year,month,day):
#     fday = date(year,1,1)
#     dc = (date(year,month,day)-fday).days + 1

#     for name in namelist:
#         savefile = f'D:/data/MLS/zonal_deviation/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
#         globals()[kind + name] = np.load(savefile)

#     vudev_mean = np.mean(devV*devU,axis=2)
#     Fy = (((-1)*a*vudev_mean*np.cos(phicord)).T*rho).T
#     devFy = np.gradient(Fy*np.cos(phicord), phicord,axis=1)

#     vTdev_mean = np.mean(devV*devT,axis=2)
#     Fz = ((a*np.cos(phicord)*f*R*vTdev_mean/(N_2*H)).T*rho).T
#     devFz = np.gradient(Fz,z,axis=0)
#     nablaF = devFy/(a*np.cos(phicord)) + devFz
#     nablaF = ((nablaF/(a*np.cos(phicord))).T/rho).T
#     nF = nablaF*60*60*24    
#     return Fy, Fz, nF

def load_zonalU(year,month,day):
    fday = date(year,1,1)
    dc = (date(year,month,day)-fday).days + 1
    savefile = f'D:/data/MLS/zonalU_from_sheerBalance/{year}/{year}d{str(dc).zfill(3)}_U_zonal.npy'
    zonalU =np.load(savefile)
    # print(zonalU.shape)
    dsU = xr.DataArray(zonalU,dims=['k','j'],
                        coords={'k':pcord, 'j':ycord},
                        name=f'd{str(dc).zfill(3)}zonalU')
    # print(dsU.shape)
    return dsU

def cal_refractive_index(dsU,k):
    # fday = date(year,1,1)
    # dc = (date(year,month,day)-fday).days + 1
    ubar_zdev = xr.DataArray(np.gradient(dsU,zcord,axis=0),
                        dims=['k','j'],
                        coords={'k':pcord, 'j':ycord})

    ubar_2zdev = xr.DataArray(np.gradient(ubar_zdev*xr_rho/N_2,zcord,axis=0),
                        dims=['k','j'],
                        coords={'k':pcord, 'j':ycord})
    ubarCosphi_phidev = xr.DataArray(np.gradient(dsU*xr_cosphi,phicord,axis=1),
                        dims=['k','j'],
                        coords={'k':pcord, 'j':ycord})
    ubar_2phidev = xr.DataArray(np.gradient(ubarCosphi_phidev/xr_cosphi,phicord,axis=1),
                        dims=['k','j'],
                        coords={'k':pcord, 'j':ycord})
    # print(ubar_zdev.shape)
    # print(ubar_2zdev.shape)
    # print(ubarCosphi_phidev.shape)
    # print(ubar_2phidev.shape)
    # print((2*omega*xr_cosphi/a).shape)
    # print((ubar_2phidev/a**2).shape)
    # print((ubar_2zdev*xr_f**2).shape)
    # print((ubar_2zdev*xr_f**2/xr_rho).shape)
    
    qbarphi = (ubar_2phidev/a**2)*(-1) - (ubar_2zdev*xr_f**2/xr_rho) + (2*omega*xr_cosphi/a)
    # print(qbarphi.shape)
    nk_2 = (qbarphi/dsU) - (k/(a*xr_cosphi))**2 - (xr_f/(2*N_2**0.5*H))**2
    return nk_2


def monthYearMean(startYear,endYear):
    # notUru = 2021
    w_lat = np.broadcast_to(cosphi,(yearnum,len(cosphi)))
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    yearMeanFz = np.zeros((dc),dtype=np.float32)
    yearStdFz = np.zeros((dc),dtype=np.float32)
    amonth = meanmonth
    day_count = 0
    for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
        # dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
        fz_year = np.zeros((endYear+1-startYear),np.float32)
        for ayear in range(startYear,endYear+1):
            dsU = load_zonalU(ayear,amonth,aday)
            zk_2 = cal_refractive_index(dsU,wn)
            if day_count == 0:
                sum_zk_2 = zk_2
            else:
                sum_zk_2 =+ zk_2
            day_count = +1
    mean_zk_2 = sum_zk_2/day_count
    print(f'finish YearMean nk!')
    return mean_zk_2


def monthMean(ayear):
    # notUru = 2021
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    oneYearMeanFz = np.zeros((dc),dtype=np.float32)
    w_lat = cosphi
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
            loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
            dataz = np.load(loadfile)
            Fz = dataz["Fz"]
            fz_1d = np.zeros((latIndex[1]-latIndex[0]+1),np.float32)
            for ind in range(latIndex[0],latIndex[1]+1):
                fz_1d[ind-latIndex[0]] = Fz[prsInd,ind]

                # if ind == latIndex[0]:
                #     fz_1d = Fz[prsInd,ind]
                # else:
                #     fz_1d = np.append(fz_1d, Fz[prsInd,ind])
            oneYearMeanFz[dnum-1] = np.nansum(fz_1d*w_lat)/np.sum(w_lat[~np.isnan(fz_1d)])
    print(f'finish Fz {ayear}!')
    return oneYearMeanFz

def draw():
    dsU = load_zonalU(year,month,day)
    zk_2 = cal_refractive_index(dsU,wn)
    zk_2 = zk_2*xr_cosphi**2*a**2
    print(zk_2.shape)
    print(zk_2)
    fig, axes = plt.subplots(1,1,figsize=(7, 6),facecolor='grey',sharex=True,sharey=True)
    axes.set_ylim(lim,1.0)
    axes.set_xlim(latrange[0],latrange[1])
    axes.set_yscale('log')
    axes.set_yticks(yticks)
    axes.set_yticklabels(ylabel)
    axes.set_xlabel('LAT')
    axes.set_ylabel('pressure')
    min_value ,max_value = -100, 100
    div=40      #図を描くのに何色用いるか
    interval=np.linspace(min_value,max_value,div+1)
    X,Y=np.meshgrid(ycord,pcord)
    contf = axes.contourf(X,Y,zk_2,interval,cmap='PuOr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
    cont = axes.contour(X,Y,zk_2,np.arange(-100,101,100),colors='black')
    # cont = axes.contour(X,Y,zk_2,np.arange(-100,101,10),colors='black')
    # axes.clabel(cont,fmt='%1.3f')
    axes.clabel(cont,fmt='%d')
    fig.suptitle(f'nk_2 {year}/{month}/{day}',fontsize=20)
    axpos = axes.get_position()
    cbar_ax = fig.add_axes([0.87, axpos.y0, 0.02, axpos.height])
    fig.colorbar(contf,cax=cbar_ax)
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(wspace=0.15)
    # if not os.path.exists(f'./picture/monthYearMean/{month}'):
    #     os.makedirs(f'./picture/yearsmean_2020/{month}')
    plt.savefig(f'D:/picture/study/MLS/nk_2/{year}{str(month).zfill(2)+str(day).zfill(2)}_nk_2.png')
    print(f'finish drawing!!!')

def main():
    draw()


if __name__ == '__main__':
    main()

# %%

# def load_zonalT(year,month,day):
#     fday = date(year,1,1)
#     dc = (date(year,month,day)-fday).days + 1
#     savefile = f'D:/data/MLS/zonal_deviation/T/{year}/{year}d{str(dc).zfill(3)}_T_zonal.npy'
#     zonalT =np.load(savefile)
#     return zonalT






# def checkDayIndex(datelist1,datelist2):
#     fdate = date(notUru,1,1)
#     date1 = date(notUru,datelist1[0],datelist1[1])
#     date2 = date(notUru,datelist2[0],datelist2[1])
#     startInd = (date1-fdate).days
#     endInd = (date2-fdate).days
#     return startInd, endInd

# def makeXaxis(datelist1,datelist2):
#     cdaylist = np.array([],dtype=np.int32)
#     strDatelist = np.array([],dtype=np.str_)
#     fdate = date(notUru,1,1)
#     for month in range(datelist1[0],datelist2[0]+1):
#         cday = (date(notUru,month,1)-fdate).days+1
#         cdaylist = np.append(cdaylist,cday)
#         strDatelist = np.append(strDatelist,f'{str(month).zfill(2)}/01')
#     if datelist2[1]==calendar.monthrange(notUru,datelist2[0])[1]:
#         cday = (date(notUru,datelist2[0],datelist2[1])-fdate).days+1
#         if datelist2[0]==12:
#             nextM = 1
#         else:
#             nextM = datelist2[0]+1
#         strdate = f'{str(nextM).zfill(2)}/01'
#         cdaylist = np.append(cdaylist,cday)
#         strDatelist = np.append(strDatelist,strdate)
#     return cdaylist, strDatelist

# def movingMean(data):
#     movingdata = np.zeros((len(data)),dtype=np.float32)*np.nan
#     for day in range(len(data)):
#         # slicedata = data[halfmd:-halfmd]
#         if day >= halfmd and day < len(data)-halfmd:
#             # print(day)
#             movingvalue = np.nanmean(data[day-halfmd:day+halfmd+1])
#             # print(movingvalue)
#             movingdata[day] = movingvalue
#     # if len(np.where(movingdata==0)[0])/2==halfmd:
#     #     print('OK')
#     # print(movingdata)
#     return movingdata


# def draw():
#     yMean, stdup, stddown = monthYearMean(meanstart, meanend)
#     y = monthMean(year)
#     print("start yMean")
#     yMean = movingMean(yMean)
#     stdup = movingMean(stdup)
#     stddown = movingMean(stddown)
#     print(yMean)
#     y = movingMean(y)
#     print("start 2020")
#     # print(y)

#     sInd, eInd = checkDayIndex(graphStartDate, graphEndtDate)
#     x = np.arange(sInd+1,eInd+1+1)
#     fig, axes = plt.subplots(1,1,figsize=(9, 6),facecolor='#fff')
#     cdaylist, strDate = makeXaxis(graphStartDate,graphEndtDate)
#     # date1 = date(notUru,graphStartDate[0],graphStartDate[1])
#     # date2 = date(notUru,graphEndtDate[0],graphEndtDate[1])
#     # delta = timedelta(days=1)
#     # x = drange(date1,date2+delta,delta)
#     # print(x.shape)
#     # print(y.shape)
#     # print(yMean.shape)

#     # axes.text(1,1,f'red:{year}')
#     axes.set_ylim(schalerange[0],schalerange[1])
#     # axes[0].set_xlim(latrange[0],latrange[1])
#     # axes[0].set_yscale('log')
#     # axes[0].set_yticks(yticks)
#     # axes[0].set_yticklabels(ylabel)
#     axes.set_xlabel(f'red:{year}')
#     # axes[1].set_xlabel('LAT')
#     # axes[0].set_ylabel('pressure',labelpad=-10)
#     axes.set_ylabel('Fz',fontsize=15)
#     # axes[1].set_ylabel('pressure')
#     axes.fill_between(x,stddown[sInd:eInd+1],stdup[sInd:eInd+1],color='#ddd')
#     axes.plot(x,y[sInd:eInd+1],color='red')
#     axes.plot(x,yMean[sInd:eInd+1],color='blue')
#     axes.plot(x,np.zeros((len(x)),dtype=np.float32),color='black',linewidth=0.7)
#     # from matplotlib.dates import DateFormatter
#     # xaxis_ = axes.xaxis
#     # xaxis_.set_major_formatter(DateFormatter('%m/%d'))
#     axes.set_xticks(cdaylist)
#     axes.set_xticklabels(strDate)
#     axes.set_title(f'Fz intensity  prs={meanprs}hPa red:{year} latWeightMean {meanlatrange[0]}to{meanlatrange[1]}',fontsize=20)
#     plt.grid(True)
#     # plt.savefig(f'D:/picture/study/MLS/Fz_intensity/movingMean_latWeightMean/std/prs{meanprs}_lat{meanlatrange[0]}to{meanlatrange[1]}Mean_E-Pflux_Fz_intensity.png')
#     plt.show()
#     print(f'finish drawing!!!')

# def main():
#     draw()


# if __name__ == '__main__':
#     main()

# %%
import xarray as xr
import numpy as np
# data = xr.DataArray(np.random.randn(2, 3))
# print(data)

#例データ
data_np = np.random.randn(5,4)
x_axis = np.linspace(0.0, 1.0, 5)
t_axis = np.linspace(0.0, 2.0, 4)

data1 = xr.DataArray(data_np, dims=['x','t'], 
                    coords={'x':x_axis, 't':t_axis}, 
                    name='some_measurement')

# print(data1)
# print(data1.loc[0:0.5,:1.0])
# print(data1.isel(x=1))

import xarray as xr
import numpy as np
ds = xr.Dataset({'data1': (['x','t'], np.random.randn(5,4)), 'data2': (['x','t'], np.random.randn(5,4))}, 
                coords={'x': x_axis, 't': t_axis})

# print(ds)
# print(np.random.randn(5,4))

# print(ds['data1'])
lamda = np.array([1,10,100])
da = xr.DataArray(np.arange(1,61).reshape(5,4,3),dims=['x','t','lamda'], 
                    coords={'x':x_axis, 't':t_axis,'lamda':lamda}, 
                    name='T_data')
# print(da)

a = np.arange(2,9,2)
a = xr.DataArray(a,dims=['t'],coords={'t':t_axis})
print(a)
print(a**2)
# print(da*a)
# cal = ds['data1'] + da
# print(cal)
# print(cal.mean(('lamda')))
# dev = np.gradient(da,lamda,axis=2)

# print(type(dev))