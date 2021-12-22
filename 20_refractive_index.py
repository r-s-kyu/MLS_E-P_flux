


# %%
import numpy as np
import xarray
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
year = 2020
meanlatrange = [-75,-40]
# meanprs = 2.6101572e+01
# meanprs = 5.6234131e+01
meanprs = 100
notUru = 2021 #うるう年以外なら何でもよい
movingDay = 7
grid = 5
stdTimes = 1
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
phicord = np.arange(-90,91,5)*(math.pi/180.)
ycord = np.arange(-90, 90.1, 5)
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
rho = rhos*(pcord*100/ps)
N_2 = 4.0e-4
z = -H*np.log(pcord*100/ps)
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

def cal_refractive_index():
    qphi =  2*omega*

def load_zonalT(year,month,day):
    fday = date(year,1,1)
    dc = (date(year,month,day)-fday).days + 1
    savefile = f'D:/data/MLS/zonal_deviation/T/{year}/{year}d{str(dc).zfill(3)}_T_zonal.npy'
    zonalT =np.load(savefile)
    return zonalT


def monthYearMean(startYear,endYear):
    # notUru = 2021
    w_lat = np.broadcast_to(cosphi,(yearnum,len(cosphi)))
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    yearMeanFz = np.zeros((dc),dtype=np.float32)
    yearStdFz = np.zeros((dc),dtype=np.float32)
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            fz_year = np.zeros((endYear+1-startYear),np.float32)
            for ayear in range(startYear,endYear+1):
                fz_lat = np.zeros((latIndex[1]-latIndex[0]+1),np.float32)

                dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
                loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
                dataz = np.load(loadfile)
                Fz = dataz["Fz"]
                for ind in range(latIndex[0],latIndex[1]+1):
                    fz_lat[ind-latIndex[0]] = Fz[prsInd,ind]
                fzlatMean = np.nanmean(fz_lat)
                fz_year[ayear-startYear] = fzlatMean
            # yearMeanFz[dnum-1] = np.nanmean(fz_1d)
            yearMeanFz[dnum-1] = np.nanmean(fz_year)
            yearStdFz[dnum-1] = np.nanstd(fz_year)
            stdup = yearMeanFz+yearStdFz*stdTimes
            stddown = yearMeanFz-yearStdFz*stdTimes
            print(f'{amonth}/{aday}')
    print(f'finish YearMeanFz!')
    return yearMeanFz, stdup, stddown


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

def checkDayIndex(datelist1,datelist2):
    fdate = date(notUru,1,1)
    date1 = date(notUru,datelist1[0],datelist1[1])
    date2 = date(notUru,datelist2[0],datelist2[1])
    startInd = (date1-fdate).days
    endInd = (date2-fdate).days
    return startInd, endInd

def makeXaxis(datelist1,datelist2):
    cdaylist = np.array([],dtype=np.int32)
    strDatelist = np.array([],dtype=np.str_)
    fdate = date(notUru,1,1)
    for month in range(datelist1[0],datelist2[0]+1):
        cday = (date(notUru,month,1)-fdate).days+1
        cdaylist = np.append(cdaylist,cday)
        strDatelist = np.append(strDatelist,f'{str(month).zfill(2)}/01')
    if datelist2[1]==calendar.monthrange(notUru,datelist2[0])[1]:
        cday = (date(notUru,datelist2[0],datelist2[1])-fdate).days+1
        if datelist2[0]==12:
            nextM = 1
        else:
            nextM = datelist2[0]+1
        strdate = f'{str(nextM).zfill(2)}/01'
        cdaylist = np.append(cdaylist,cday)
        strDatelist = np.append(strDatelist,strdate)
    return cdaylist, strDatelist

def movingMean(data):
    movingdata = np.zeros((len(data)),dtype=np.float32)*np.nan
    for day in range(len(data)):
        # slicedata = data[halfmd:-halfmd]
        if day >= halfmd and day < len(data)-halfmd:
            # print(day)
            movingvalue = np.nanmean(data[day-halfmd:day+halfmd+1])
            # print(movingvalue)
            movingdata[day] = movingvalue
    # if len(np.where(movingdata==0)[0])/2==halfmd:
    #     print('OK')
    # print(movingdata)
    return movingdata


def draw():
    yMean, stdup, stddown = monthYearMean(meanstart, meanend)
    y = monthMean(year)
    print("start yMean")
    yMean = movingMean(yMean)
    stdup = movingMean(stdup)
    stddown = movingMean(stddown)
    print(yMean)
    y = movingMean(y)
    print("start 2020")
    # print(y)

    sInd, eInd = checkDayIndex(graphStartDate, graphEndtDate)
    x = np.arange(sInd+1,eInd+1+1)
    fig, axes = plt.subplots(1,1,figsize=(9, 6),facecolor='#fff')
    cdaylist, strDate = makeXaxis(graphStartDate,graphEndtDate)
    # date1 = date(notUru,graphStartDate[0],graphStartDate[1])
    # date2 = date(notUru,graphEndtDate[0],graphEndtDate[1])
    # delta = timedelta(days=1)
    # x = drange(date1,date2+delta,delta)
    # print(x.shape)
    # print(y.shape)
    # print(yMean.shape)

    # axes.text(1,1,f'red:{year}')
    axes.set_ylim(schalerange[0],schalerange[1])
    # axes[0].set_xlim(latrange[0],latrange[1])
    # axes[0].set_yscale('log')
    # axes[0].set_yticks(yticks)
    # axes[0].set_yticklabels(ylabel)
    axes.set_xlabel(f'red:{year}')
    # axes[1].set_xlabel('LAT')
    # axes[0].set_ylabel('pressure',labelpad=-10)
    axes.set_ylabel('Fz',fontsize=15)
    # axes[1].set_ylabel('pressure')
    axes.fill_between(x,stddown[sInd:eInd+1],stdup[sInd:eInd+1],color='#ddd')
    axes.plot(x,y[sInd:eInd+1],color='red')
    axes.plot(x,yMean[sInd:eInd+1],color='blue')
    axes.plot(x,np.zeros((len(x)),dtype=np.float32),color='black',linewidth=0.7)
    # from matplotlib.dates import DateFormatter
    # xaxis_ = axes.xaxis
    # xaxis_.set_major_formatter(DateFormatter('%m/%d'))
    axes.set_xticks(cdaylist)
    axes.set_xticklabels(strDate)
    axes.set_title(f'Fz intensity  prs={meanprs}hPa red:{year} latWeightMean {meanlatrange[0]}to{meanlatrange[1]}',fontsize=20)
    plt.grid(True)
    # plt.savefig(f'D:/picture/study/MLS/Fz_intensity/movingMean_latWeightMean/std/prs{meanprs}_lat{meanlatrange[0]}to{meanlatrange[1]}Mean_E-Pflux_Fz_intensity.png')
    plt.show()
    print(f'finish drawing!!!')

def main():
    draw()


if __name__ == '__main__':
    main()

# %%

a = np.array(np.arange(1,13).reshape(3,4),dtype=np.float32)
# a[1,1] = np.nan
b = np.arange(1,5)
print(a)
print(b)
# c = np.average(a,axis=0,weights=b)
d = np.broadcast_to(b,(3,4))
print(d)
c = np.average(a,weights=b,axis=1)
print(c)
c = np.average(a,weights=d)
print(c)
e = np.nansum(a*d)/np.sum(d)
print(np.sum(d))
print(np.sum(d[~np.isnan(a)]))

# つまり、np.nanを含む配列の加重平均は
data = np.array(np.arange(1,13).reshape(3,4),dtype=np.float32)
data[1,1] = np.nan 

# 平均するdataと同じ配列の重み配列を用意
w = np.arange(1,5)
w_samedimen = np.broadcast_to(w,(3,4))

# np.nan以外の場所で加重平均
ava = np.nansum(data*w_samedimen)/np.sum(w_samedimen[~np.isnan(data)])
print(ava)
# print(e)