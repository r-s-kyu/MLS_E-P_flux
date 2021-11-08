# %%
import numpy as np
import math
from datetime import date
import os
import sys

# ===================初期値======================
moveMeanDay = 3
year = 2020
startyear = 2010
endyear = 2020
NAMElist = np.array(['GPH','T'])
# NAMElist = np.array(['GPH'])
namelist = np.array(['hgt','tmp'])
# ===================定数=====================

md = moveMeanDay
pq = 'hgt'

latphicord = np.arange(-90, 90.1, 1.25)*(math.pi/180.)
a = 6.37e+6
g = 9.80665										# 重力加速度
omega = 7.29e-5
lon_cord = np.arange(0, 360, 1.25)
lat_cord = np.arange(-90, 90.1, 1.25)
print(lon_cord.shape)
print(lat_cord.shape)

lonlamdacord = np.radians(lon_cord)
f = 2*omega*np.sin(latphicord)
# y = a**2*math.pi/(360/1.25)
# x = (a*np.cos(latphicord))**2*math.pi/(360/5)
# M = a*omega*np.cos(latphicord)


def gWind(year):
    dayc = (date(year,12,31)-date(year,1,1)).days + 1

    filename = f'D:/data/JRA55/zonal_deviation'
    if not os.path.exists(filename):
        os.mkdir(filename)
    # nlist = ['U','V']
    nlist = ['ugrd','vgrd']
    for na in nlist:
        if not os.path.exists(filename+f'/{na}'):
            os.mkdir(filename+f'/{na}')         
        if not os.path.exists(filename+f'/{na}/{year}'):
            os.mkdir(filename+f'/{na}/{year}')
    
    for l in range(dayc):
        file = f'D:/data/JRA55/hgt/{year}/{year}d{str(l+1).zfill(3)}_hgt_dev.npy'
        phidev = np.load(file)
            # print(devphi_devx[:,:,i].shape)
        devphi_devx = np.gradient(phidev*g, latphicord, axis=1)
        devphi_devy = np.gradient(phidev*g, lonlamdacord, axis=2)

        ugdev = np.transpose((-1*np.transpose(devphi_devy,(0,2,1))/f),(0,2,1))/a
        vgdev = np.transpose((1*np.transpose(devphi_devx,(0,2,1))/(f*np.cos(latphicord))),(0,2,1))/a

        savefile = f'D:/data/JRA55/zonal_deviation/ugrd/{year}/{year}d{str(l+1).zfill(3)}_ugrd_dev.npy'
        np.save(savefile, ugdev)
        print(f'complete to make ugrd-dev {year}d{str(l+1).zfill(3)}')

        savefile = f'D:/data/JRA55/zonal_deviation/vgrd/{year}/{year}d{str(l+1).zfill(3)}_vgrd_dev.npy'
        np.save(savefile, vgdev)
        print(f'complete to make vgrd-dev {year}d{str(l+1).zfill(3)}')

    return 


for year in range(startyear,endyear+1):
    for name in namelist:
        gWind(year)

print('finish program!!!!!!!')

