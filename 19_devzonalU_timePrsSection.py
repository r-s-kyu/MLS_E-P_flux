
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
meanlatrange = [-75,-40]
# meanprs = 100 
notUru = 2021 #うるう年以外なら何でもよい
grid = 5

# ====================描画値===================
graphStartDate = [7,1]
graphEndtDate = [12,31]
# vector_scale = 8.0e+5
# lim = 100
lim = 3.1622775e+02
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
    w_lat = np.broadcast_to(cosphi,(meanyears,len(cosphi)))
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    yearMeanFz = np.zeros((dc,55),dtype=np.float32)
    yearMeanU = np.zeros((dc,55),dtype=np.float32)
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            nf_3d = np.zeros((55,meanyears,latIndex[1]-latIndex[0]+1),np.float32)
            u_3d = np.zeros((55,meanyears,latIndex[1]-latIndex[0]+1),np.float32)
            for ayear in range(startYear,endYear+1):
                dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
                loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
                dataz = np.load(loadfile)
                nF = dataz["nablaF"]
                zonalU = load_zonalU(ayear,amonth,aday)
                for ind in range(latIndex[0],latIndex[1]+1):
                    nf_3d[:,ayear-startYear,ind-latIndex[0]] = nF[:,ind]
                    u_3d[:,ayear-startYear,ind-latIndex[0]] = zonalU[:,ind]
            for prsInd in range(55):
                yearMeanFz[dnum-1,prsInd] = np.nansum(nf_3d[prsInd]*w_lat)/np.sum(w_lat[~np.isnan(nf_3d[prsInd])])
                yearMeanU[dnum-1,prsInd] = np.nansum(u_3d[prsInd]*w_lat)/np.sum(w_lat[~np.isnan(u_3d[prsInd])])
            print(f'{amonth}/{aday}')
    print(f'finish YearMeanFz!')
    yearMeanFz = yearMeanFz.T
    yearMeanU = yearMeanU.T
    return yearMeanFz, yearMeanU


def monthMean(ayear):
    # notUru = 2021
    w_lat = cosphi
    dc = (date(notUru,12,31)-date(notUru,1,1)).days+1
    oneYearMeanFz = np.zeros((dc,55),dtype=np.float32)
    oneYearMeanU = np.zeros((dc,55),dtype=np.float32)
    for amonth in range(1,13):
        for aday in range(1,calendar.monthrange(notUru,amonth)[1]+1): #うるう年でなければ何でもよい
            dnum = (date(notUru,amonth,aday)-date(notUru,1,1)).days+1
            dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
            nf_2d = np.zeros((55,latIndex[1]-latIndex[0]+1),np.float32)
            u_2d = np.zeros((55,latIndex[1]-latIndex[0]+1),np.float32)
            loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
            dataz = np.load(loadfile)
            nF = dataz["nablaF"]
            zonalU = load_zonalU(ayear,amonth,aday)
            for ind in range(latIndex[0],latIndex[1]+1):
                nf_2d[:,ind-latIndex[0]] = nF[:,ind]
                u_2d[:,ind-latIndex[0]] = zonalU[:,ind]
            for prsInd in range(55):
                oneYearMeanFz[dnum-1,prsInd] = np.nansum(nf_2d[prsInd]*w_lat)/np.sum(w_lat[~np.isnan(nf_2d[prsInd])])
                oneYearMeanU[dnum-1,prsInd] = np.nansum(u_2d[prsInd]*w_lat)/np.sum(w_lat[~np.isnan(u_2d[prsInd])])
    print(f'finish Fz {ayear}!')
    oneYearMeanFz = oneYearMeanFz.T
    oneYearMeanU = oneYearMeanU.T
    return oneYearMeanFz, oneYearMeanU

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

def caldata():
    nablaF1,zonalU1 = monthYearMean(meanstart,meanend)
    print(f'nablaF:{nablaF1.shape}')
    print(f'zonalU:{zonalU1.shape}')
    return nablaF1, zonalU1


nablaF0,zonalU0 = caldata()
print(f'finish calculation')
# %%
def caldata2(ayear):
    nablaF2,zonalU2 = monthMean(ayear)
    print(f'nablaF:{nablaF2.shape}')
    print(f'zonalU:{zonalU2.shape}')
    return nablaF2,zonalU2
year = 2020
# for year in range(2010,2021):
nablaF1,zonalU1 = caldata2(year)
# def draw():

devDF = nablaF1 - nablaF0
devU = zonalU1 - zonalU0

fig, axes = plt.subplots(1,2,figsize=(11, 6),facecolor='#ddd',sharex=True,sharey=True)
axes[0].set_ylim(lim,0.1)
axes[0].set_xlim(latrange[0],latrange[1])
axes[0].set_yscale('log')
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(ylabel)
# axes[0].set_xlabel('')
# axes[1].set_xlabel('LAT')
# axes[0].set_ylabel('pressure',labelpad=-10)
axes[0].set_ylabel('pressure')
# axes[1].set_ylabel('pressure')

for prsnum in range(len(pcord)):
    if pcord[prsnum] == lim:
        num = prsnum
sInd, eInd = checkDayIndex(graphStartDate, graphEndtDate)
xcord = np.arange(sInd+1,eInd+1+1)
cdaylist, strDate = makeXaxis(graphStartDate,graphEndtDate)
axes[0].set_xticks(cdaylist)
axes[0].set_xticklabels(strDate)
axes[0].set_xlim(cdaylist[0],cdaylist[-1])
title0 = f'deviation DF {year}-10Mean'
title1 = f'deviation U {year}-10Mean'   


min_value ,max_value = -50, 50
div=40      #図を描くのに何色用いるか
interval=np.linspace(min_value,max_value,div+1)

min_value ,max_value = -30, 30
div=40      #図を描くのに何色用いるか
interval2=np.linspace(min_value,max_value,div+1)
X,Y=np.meshgrid(xcord,pcord)

# cont = axes[axnum].contour(X,Y,zonalhgt,colors='black')
# cont = axes[y].contour(X,Y,zonalU,levels=np.linspace(-100,100,25),linewidths=0.75, cmap='plasma_r')
cont0 = axes[0].contour(X,Y,nablaF1[:,sInd:eInd+1],linewidths=0.75, colors='black',alpha=0.65)
contf0 = axes[0].contourf(X,Y,devDF[:,sInd:eInd+1],interval,cmap='bwr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
cont1 = axes[1].contour(X,Y,zonalU1[:,sInd:eInd+1],levels=np.array(np.arange(-100,121,10),dtype=np.int32),linewidths=0.75, colors='black',alpha=0.65)
contf1 = axes[1].contourf(X,Y,devU[:,sInd:eInd+1],interval2,cmap='bwr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
# q = axes[y].quiver(X[num:,2::mabiki], Y[num:,2::mabiki], Fy[num:,2::mabiki], Fz[num:,2::mabiki]*100,pivot='middle',
#             scale_units='xy', headwidth=5,scale=vector_scale, color='#5c6',width=0.0065,alpha=0.70)
axes[0].set_title(f'{title0}',fontsize=15)
axes[0].clabel(cont0, cont0.levels[::1], fmt='%d', inline=True, fontsize=12)
axes[1].set_title(f'{title1}',fontsize=15)
axes[1].clabel(cont1, cont1.levels[::1], fmt='%d', inline=True, fontsize=12)

# cbar_ax = fig.add_axes([])
# fig.colorbar(contf0)
# fig.colorbar(contf1)
# fig.colorbar()

def colorbar(fig, contf, index1=0.95, index2=0.1, index3=0.02, index4=0.8, direction='vertical', label='test'):
        cbaxes = fig.add_axes([index1, index2, index3, index4]) #x位置, y位置, 太さ, 長さ 
        fig.colorbar(contf, cax = cbaxes, orientation=direction)
        return fig
fig = colorbar(fig,contf0,index1=0.91)
fig = colorbar(fig,contf1,index1=0.45)
fig.suptitle(f'E-Pflux and U',fontsize=20)
fig.subplots_adjust(wspace=0.33)
# axpos = axes[0].get_position()
# cbar_ax = fig.add_axes([0.81, axpos.y0, 0.02, axpos.height])
# fig.colorbar(contf0,cax=cbar_ax)
# fig.text(0.77,0.90,'∇F[m/s/d]',size=14.5)
# cbar_ax2 = fig.add_axes([0.50,axpos.y0, 0.02,axpos.height])
# fig.colorbar(contf1,cax=cbar_ax2)
# fig.text(0.89,0.90,'U[m/s]',size=14.5)
plt.subplots_adjust(right=0.88)
plt.subplots_adjust(left=0.1)
# plt.subplots_adjust(wspace0.15)
# if not os.path.exists(f'./picture/monthYearMean/{month}'):
#     os.makedirs(f'./picture/yearsmean_2020/{month}')
plt.savefig(f'D:/picture/study/MLS/e-p_flux/timePrsSection/latWeightMean/deviation/e-p_flux_timePrsSection_10Mean_and_{year}_lat{meanlatrange[0]}to{meanlatrange[1]}Mean.png')
plt.show()

print(f'finish drawing!!!')

# def main():
#     draw()


# if __name__ == '__main__':
#     main()

