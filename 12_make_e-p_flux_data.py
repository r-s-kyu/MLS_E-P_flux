
# %%
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt
import calendar
import os

# ====================初期値===================
startyear = 2010
endyear = 2020

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
# pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
#         650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
#         50,30,20,10,7,5,3,2,1])
prsfile = f'./text/prs_values.npy'
parent = f'D:/data/MLS/e-p_flux'
child = np.array([],dtype=np.str_) #ここの配列に作りたいディレクトリの階層を高い順で記入
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

def makeEPflux(year,month,day):
    fday = date(year,1,1)
    dc = (date(year,month,day)-fday).days + 1

    for name in namelist:
        savefile = f'D:/data/MLS/zonal_deviation/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
        globals()[kind + name] = np.load(savefile)

    vudev_mean = np.mean(devV*devU,axis=2)
    Fy = (((-1)*a*vudev_mean*np.cos(phicord)).T*rho).T
    devFy = np.gradient(Fy*np.cos(phicord), phicord,axis=1)

    vTdev_mean = np.mean(devV*devT,axis=2)
    Fz = ((a*np.cos(phicord)*f*R*vTdev_mean/(N_2*H)).T*rho).T
    devFz = np.gradient(Fz,z,axis=0)
    nablaF = devFy/(a*np.cos(phicord)) + devFz
    nablaF = ((nablaF/(a*np.cos(phicord))).T/rho).T
    nF = nablaF*60*60*24   # [/s/d]
    return Fy, Fz, nF

# def load_zonalU(year,month,day):
#     fday = date(year,1,1)
#     dc = (date(year,month,day)-fday).days + 1
#     savefile = f'D:/data/MLS/zonalU_from_sheerBalance/{year}/{year}d{str(dc).zfill(3)}_U_zonal.npy'
#     zonalU =np.load(savefile)
#     return zonalU

def saveData(file,Fy,Fz,nF):
    np.savez(file,Fy=Fy,Fz=Fz,nablaF=nF)
    return

def makeDir(parent_path,child_path):
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    for add_path in child_path:
        parent_path += f'/{add_path}'
        if not os.path.exists(parent_path):
            os.mkdir(parent_path)
    return parent_path

def main(child):
    for year in range(startyear,endyear+1):
        add_year_child_path = np.append(child ,str(year).zfill(4))
        dirpath = makeDir(parent,add_year_child_path)
        for month in range(1,13):
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                Fy, Fz, nF = makeEPflux(year,month,day)
                dates = f'{str(year).zfill(4)+str(month).zfill(2)+str(day).zfill(2)}'
                filename = f'e-p_flux.{dates}.npz'
                savefile =  f'{dirpath}/{filename}'
                saveData(savefile, Fy, Fz, nF)
                print(dates)


if __name__ == '__main__':
    main(child)

