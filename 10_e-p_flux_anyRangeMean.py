
# %%
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt
import calendar
import os

from numpy.lib.npyio import load

# ====================初期値===================
meanstart = 2010
meanend = 2019
year = 2020
month = 10
dateRange = np.array([[9,1],[11,30]],dtype=np.int32)

# ====================描画値===================
vector_scale = 8.0e+5
lim = 100
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-80,-30]

# ====================定数=====================
defineYear = 2019 #うるう年ではない年（てきとう）
# fday = date(defineYear,1,1)
# sdatecount = (date(defineYear,sdate[0],sdate[1])-fday).days + 1
# edatecount = (date(defineYear,edate[0],edate[1])-fday).days + 1
# allDateCount = (date(defineYear,edate[0],edate[1]) - date(defineYear,sdate[0],sdate[1])).days +1
meanyears = meanend - meanstart + 1
# pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
#         650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
#         50,30,20,10,7,5,3,2,1])
prsfile = f'./text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
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


def changednum(year,month,day):
    dc = (date(year,month,day)-date(year,1,1)).days+1
    return dc

def changedate(year,dc):
    month = 1
    while (date(year,month,calendar.monthrange(year,month)[1])-date(year,1,1)).days + 1 < dc:
        month += 1
    day = 1
    while dc > (date(year,month,day)-date(year,1,1)).days + 1:
        day += 1
    return month, day

def load_zonalU(year,month,day):
    fday = date(year,1,1)
    dc = (date(year,month,day)-fday).days + 1
    savefile = f'D:/data/MLS/zonalU_from_sheerBalance/{year}/{year}d{str(dc).zfill(3)}_U_zonal.npy'
    zonalU =np.load(savefile)
    return zonalU

def monthYearMean(startYear,endYear):
    for ayear in range(startYear,endYear+1):
        sdc = changednum(ayear,dateRange[0,0],dateRange[0,1])
        edc = changednum(ayear,dateRange[1,0],dateRange[1,1])
        for dnum in range(sdc,edc+1):
            amonth, aday = changedate(ayear,dnum)
            dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
            loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
            dataz = np.load(loadfile)
            Fy, Fz, nF = dataz["Fy"], dataz["Fz"], dataz["nablaF"]
            zonalU = load_zonalU(ayear,amonth,aday)
            if ayear == startYear and aday ==1:
                fy_3d, fz_3d, nf_3d = Fy[np.newaxis], Fz[np.newaxis], nF[np.newaxis]
                zonalU_3d = zonalU[np.newaxis]
            else:
                fy_3d = np.append(fy_3d,Fy[np.newaxis],axis=0)
                fz_3d = np.append(fz_3d,Fz[np.newaxis],axis=0)
                nf_3d = np.append(nf_3d,nF[np.newaxis],axis=0)
                zonalU_3d = np.append(zonalU_3d,zonalU[np.newaxis],axis=0)
        print(f'complete to add {ayear}!')  
    FyMean = np.nanmean(fy_3d,axis=0)
    FzMean = np.nanmean(fz_3d,axis=0)
    nFMean = np.nanmean(nf_3d,axis=0)
    zonalUMena = np.nanmean(zonalU_3d,axis=0)
    print(f'finish mean!')
    return FyMean, FzMean, nFMean, zonalUMena


def monthMean(ayear):
    sdc = changednum(ayear,dateRange[0,0],dateRange[0,1])
    edc = changednum(ayear,dateRange[1,0],dateRange[1,1])
    for dnum in range(sdc,edc+1):
        amonth, aday = changedate(ayear,dnum)
        zonalU = load_zonalU(ayear,amonth,aday)
        dates = f'{str(ayear).zfill(4)+str(amonth).zfill(2)+str(aday).zfill(2)}'
        loadfile = f'D:/data/MLS/e-p_flux/{ayear}/e-p_flux.{dates}.npz'
        dataz = np.load(loadfile)
        Fy, Fz, nF = dataz["Fy"], dataz["Fz"], dataz["nablaF"]
        if aday == 1:
            fy_3d, fz_3d, nf_3d = Fy[np.newaxis], Fz[np.newaxis], nF[np.newaxis]
            zonalU_3d = zonalU[np.newaxis]
        else:
            fy_3d = np.append(fy_3d,Fy[np.newaxis],axis=0)
            fz_3d = np.append(fz_3d,Fz[np.newaxis],axis=0)
            nf_3d = np.append(nf_3d,nF[np.newaxis],axis=0)
            zonalU_3d = np.append(zonalU_3d,zonalU[np.newaxis],axis=0)
    FyMean = np.nanmean(fy_3d,axis=0)
    FzMean = np.nanmean(fz_3d,axis=0)
    nFMean = np.nanmean(nf_3d,axis=0)
    zonalUMean = np.nanmean(zonalU_3d,axis=0)
    return FyMean, FzMean, nFMean, zonalUMean

    
def draw():
    fig, axes = plt.subplots(1,2,figsize=(9, 6),facecolor='#fff',sharex=True,sharey=True)
    axes[0].set_ylim(lim,1.0)
    axes[0].set_xlim(latrange[0],latrange[1])
    axes[0].set_yscale('log')
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabel)
    axes[0].set_xlabel('LAT')
    axes[1].set_xlabel('LAT')
    # axes[0].set_ylabel('pressure',labelpad=-10)
    axes[0].set_ylabel('pressure')
    # axes[1].set_ylabel('pressure')

    for prsnum in range(len(pcord)):
        if pcord[prsnum] == lim:
            num = prsnum

    min_value ,max_value = -20, 20
    div=80      #図を描くのに何色用いるか
    interval=np.linspace(min_value,max_value,div+1)
    X,Y=np.meshgrid(ycord,pcord)
    for y in range(2):
        if y == 0:
            Fy, Fz, nablaF,zonalU = monthYearMean(meanstart,meanend)
            title = f'{meanstart}-{meanend}'
        else:
            Fy, Fz, nablaF,zonalU = monthMean(year)
            title = str(year)
        # cont = axes[axnum].contour(X,Y,zonalhgt,colors='black')
        # cont = axes[y].contour(X,Y,zonalU,levels=np.linspace(-100,100,25),linewidths=0.75, cmap='plasma_r')
        cont = axes[y].contour(X,Y,zonalU,levels=np.arange(-100,101,10),linewidths=0.75, colors='black',alpha=0.65)
        contf = axes[y].contourf(X,Y,nablaF,interval,cmap='bwr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
        q = axes[y].quiver(X[num:,2::mabiki], Y[num:,2::mabiki], Fy[num:,2::mabiki], Fz[num:,2::mabiki]*100,pivot='middle',
                    scale_units='xy', headwidth=5,label='8e+5',scale=vector_scale, color='#5c6',width=0.0065,alpha=1.)
        axes[y].set_title(f'{title}',fontsize=15)
        axes[y].clabel(cont, cont.levels[::1], fmt='%d', inline=True, fontsize=12)
    str_range = f'{str(dateRange[0,0]).zfill(2)}/{str(dateRange[0,1]).zfill(2)}-{str(dateRange[1,0]).zfill(2)}/{str(dateRange[1,1]).zfill(2)}'
    str_range2 = f'{str(dateRange[0,0]).zfill(2)}{str(dateRange[0,1]).zfill(2)}to{str(dateRange[1,0]).zfill(2)}{str(dateRange[1,1]).zfill(2)}'
    fig.suptitle(f'{str_range} Mean E-Pflux and U',fontsize=20)
    axpos = axes[0].get_position()
    # axpos2 = axes[0].get_position()

    cbar_ax = fig.add_axes([0.81, axpos.y0, 0.02, axpos.height])
    fig.colorbar(contf,cax=cbar_ax)
    fig.text(0.783,0.895,'DF[m/s/d]',size=13.5)
    fig.text(0.77,0.04,f'Fy:{vector_scale:.1E}',size=12.0,color='#060')
    axes[1].annotate('->', (-30,150),annotation_clip=True)
    # arrowprops=dict(shrink=0,width=1,headwidth=8,
    #                 headlength=10, connectionstyle='arc3',
    #                 facecolor='gray', edgecolor='gray'))
    # cbar_ax2 = fig.add_axes([0.90,axpos.y0, 0.02,axpos.height])
    # fig.colorbar(cont,cax=cbar_ax2)
    # fig.text(0.89,0.90,'U[m/s]',size=14.5)
    plt.subplots_adjust(right=0.78)
    plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(wspace=0.15)
    # if not os.path.exists(f'./picture/monthYearMean/{month}'):
    #     os.makedirs(f'./picture/yearsmean_2020/{month}')
    plt.savefig(f'D:/picture/study/MLS/anyRangeMean/zonalU/range{str_range2}Mean_10YearsMean_and_{year}_E-Pflux_zonalU.png')
    plt.show()
    print(f'finish drawing!!!')

def main():
    draw()


if __name__ == '__main__':
    main()

