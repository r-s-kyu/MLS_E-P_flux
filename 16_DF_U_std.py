
# %%
import numpy as np
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
month = 8
stdnum = 2

meanlatrange = [-75,-40]
# meanprs = 1.4677993e+00
meanprs = 10
notUru = 2021 #うるう年以外なら何でもよい
grid = 5
# [1.0000000e+03 8.2540417e+02 6.8129205e+02 5.6234131e+02 4.6415887e+02
#  3.8311868e+02 3.1622775e+02 2.6101572e+02 2.1544347e+02 1.7782794e+02
#  1.4677992e+02 1.2115276e+02 1.0000000e+02 8.2540421e+01 6.8129204e+01
#  5.6234131e+01 4.6415890e+01 3.8311867e+01 3.1622776e+01 2.6101572e+01
#  2.1544348e+01 1.7782795e+01 1.4677993e+01 1.2115276e+01 1.0000000e+01
#  8.2540417e+00 6.8129206e+00 5.6234131e+00 4.6415887e+00 3.8311868e+00
#  3.1622777e+00 2.6101573e+00 2.1544347e+00 1.7782794e+00 1.4677993e+00
#  1.2115277e+00 1.0000000e+00 6.8129206e-01 4.6415889e-01 3.1622776e-01
#  2.1544346e-01 1.4677992e-01 1.0000000e-01 4.6415888e-02 2.1544347e-02
#  9.9999998e-03 4.6415888e-03 2.1544348e-03 1.0000000e-03 4.6415889e-04
#  2.1544346e-04 9.9999997e-05 4.6415887e-05 2.1544347e-05 9.9999997e-06]
# ====================描画値===================
scalerange = [-50,50]

graphStartDate = [7,1]
graphEndtDate = [12,31]
# vector_scale = 8.0e+5
lim = 100
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-80,-30]

# ====================定数=====================
defineYear = 2019 #うるう年ではない年（てきとう）
latIndex = ((np.array(meanlatrange) + 90 )/grid).astype(np.int32)
yarray = np.arange(meanlatrange[0],meanlatrange[1]+1,grid)
phiarray = np.radians(yarray)
cosphi = np.cos(phiarray)
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
# xcord = np.arange(-90, 90.1, 5)
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
def load_zonalU(year,month,day):
    fday = date(year,1,1)
    dc = (date(year,month,day)-fday).days + 1
    savefile = f'D:/data/MLS/zonalU_from_sheerBalance/{year}/{year}d{str(dc).zfill(3)}_U_zonal.npy'
    zonalU =np.load(savefile)
    return zonalU

def monthYearMean(startYear,endYear):
    # notUru = 2021
    w_lat = np.broadcast_to(cosphi,(len(cosphi)))
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    yearMeanDF = np.zeros((dc),dtype=np.float32)
    yearMeanU = np.zeros((dc),dtype=np.float32)
    yearstdDF = np.zeros((dc),dtype=np.float32)
    yearstdU = np.zeros((dc),dtype=np.float32)
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            nf_year = np.zeros((meanyears),np.float32)
            u_year = np.zeros((meanyears),np.float32)
            for ayear in range(startYear,endYear+1):
                dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
                nf_lat = np.zeros((latIndex[1]-latIndex[0]+1),dtype=np.float32)
                u_lat = np.zeros((latIndex[1]-latIndex[0]+1),dtype=np.float32)
                loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
                dataz = np.load(loadfile)
                nF = dataz["nablaF"]
                zonalU = load_zonalU(ayear,amonth,aday)
                for ind in range(latIndex[0],latIndex[1]+1):
                    nf_lat[ind-latIndex[0]] = nF[prsInd,ind]
                    u_lat[ind-latIndex[0]] = zonalU[prsInd,ind]
                nflatmean = np.nansum(nf_lat*w_lat)/np.sum(w_lat[~np.isnan(nf_lat)])
                ulatmean = np.nansum(u_lat*w_lat)/np.sum(w_lat[~np.isnan(u_lat)])
                nf_year[ayear-startYear] = nflatmean
                u_year[ayear-startYear] = ulatmean
            yearMeanDF[dnum-1] = np.nanmean(nf_year)
            yearstdDF[dnum-1] = np.nanstd(nf_year)
            yearMeanU[dnum-1] = np.nanmean(u_year)  
            yearstdU[dnum-1] = np.nanstd(u_year)
            print(f'{amonth}/{aday}')
    print(f'finish YearMeanFz!')
    return yearMeanDF, yearMeanU, yearstdDF,yearstdU


def monthMean(ayear):
    # notUru = 2021
    w_lat = cosphi
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    oneYearMeanDF = np.zeros((dc),dtype=np.float32)
    oneYearMeanU = np.zeros((dc),dtype=np.float32)
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
            nf_1d = np.zeros((latIndex[1]-latIndex[0]+1),np.float32)
            u_1d = np.zeros((latIndex[1]-latIndex[0]+1),np.float32)
            loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
            dataz = np.load(loadfile)
            nF = dataz["nablaF"]
            zonalU = load_zonalU(ayear,amonth,aday)
            for ind in range(latIndex[0],latIndex[1]+1):
                nf_1d[ind-latIndex[0]] = nF[prsInd,ind]
                u_1d[ind-latIndex[0]] = zonalU[prsInd,ind]
            oneYearMeanDF[dnum-1] = np.nansum(nf_1d*w_lat)/np.sum(w_lat[~np.isnan(nf_1d)])
            oneYearMeanU[dnum-1] = np.nansum(u_1d*w_lat)/np.sum(w_lat[~np.isnan(u_1d)])
    print(f'finish {ayear}!')
    return oneYearMeanDF, oneYearMeanU

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


meanDF, meanU ,stdDF,stdU = monthYearMean(meanstart, meanend)
upstdDF = meanDF+stdDF*stdnum
downstdDF = meanDF-stdDF*stdnum
upstdU = meanU+stdU*stdnum
downstdU = meanU-stdU*stdnum

Df,U = monthMean(year)
print("start yMean")
# yMean = movingMean(yMean)
print(meanU)
# y = movingMean(y)
print("start 2020")
print(Df.shape)

sInd, eInd = checkDayIndex(graphStartDate, graphEndtDate)
x = np.arange(sInd+1,eInd+1+1)
fig, axes = plt.subplots(2,1,figsize=(9, 6),facecolor='#fff')
cdaylist, strDate = makeXaxis(graphStartDate,graphEndtDate)
# date1 = date(notUru,graphStartDate[0],graphStartDate[1])
# date2 = date(notUru,graphEndtDate[0],graphEndtDate[1])
# delta = timedelta(days=1)
# x = drange(date1,date2+delta,delta)
# print(x.shape)
# print(y.shape)
# print(yMean.shape)


axes[0].set_ylim(scalerange[0],scalerange[1])
# axes[0].set_xlim(latrange[0],latrange[1])
# axes[0].set_yscale('log')
# axes[0].set_yticks(yticks)
# axes[0].set_yticklabels(ylabel)
# axes.set_xlabel('day')
# axes[1].set_xlabel('LAT')
# axes[0].set_ylabel('pressure',labelpad=-10)
axes[0].set_ylabel('DF',fontsize=15)
axes[1].set_ylabel('U',fontsize=15)
# axes[1].set_ylabel('pressure')
axes[0].fill_between(x,downstdDF[sInd:eInd+1],upstdDF[sInd:eInd+1],color='#ddd')
axes[0].plot(x,Df[sInd:eInd+1],color='red',linewidth=1,)
axes[0].plot(x,meanDF[sInd:eInd+1],color='blue',linestyle='solid')
# axes2 = axes.twinx()
axes[1].fill_between(x,downstdU[sInd:eInd+1],upstdU[sInd:eInd+1],color='#ddd')
axes[1].plot(x,U[sInd:eInd+1],color='red',linestyle='dashed')
axes[1].plot(x,meanU[sInd:eInd+1],color='blue',linestyle='dashed')


# from matplotlib.dates import DateFormatter
# xaxis_ = axes.xaxis
# xaxis_.set_major_formatter(DateFormatter('%m/%d'))
axes[0].set_xticks(cdaylist)
axes[0].set_xticklabels(strDate)
axes[1].set_xticks(cdaylist)
axes[1].set_xticklabels(strDate)
axes[0].set_title(f'prs={meanprs:.2f}hPa  {meanlatrange[0]}to{meanlatrange[1]}',fontsize=15)
axes[0].grid(True)
axes[1].grid(True)
plt.savefig(f'D:/picture/study/MLS/e-p_flux/DF_U/prs{meanprs:.07f}_lat{meanlatrange[0]}to{meanlatrange[1]}Mean_E-Pflux_DF.png')
plt.show()
print(f'finish drawing!!!')

print(f'finish calculation')
# %%

# def main():
#     draw()


# if __name__ == '__main__':
#     main()

