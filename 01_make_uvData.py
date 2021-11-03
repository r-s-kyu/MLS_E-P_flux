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
namelist = np.array(['ght','tmp'])
# ===================定数=====================

md = moveMeanDay
pq = 'GPH'

latphicord = np.arange(-90, 90.1, 5)*(math.pi/180.)
a = 6.37e+6
g = 9.80665										# 重力加速度
omega = 7.29e-5
lon_cord = np.arange(-180, 180.1, 5)
lat_cord = np.arange(-90, 90.1, 5)
lonlamdacord = np.radians(lon_cord)
f = 2*omega*np.sin(latphicord)
y = a**2*math.pi/(360/5)
x = (a*np.cos(latphicord))**2*math.pi/(360/5)
M = a*omega*np.cos(latphicord)


def gWind(phidev,year):
    # print(phidev[264,24,6])
    # print(x[6])
    dayc = (date(year,12,31)-date(year,1,1)).days + 1
    devphi_devx = np.zeros((dayc,55,37,73))
        # print(devphi_devx[:,:,i].shape)
    devphi_devx = np.gradient(phidev*g, latphicord, axis=2)
    devphi_devy = np.gradient(phidev*g, lonlamdacord, axis=3)
    # print(devphi_devy.shape)
    # devzonalphi_devy = np.gradient(phizonal, y)

    # sys.exit()

    ugdev = np.transpose((-1*np.transpose(devphi_devy,(0,1,3,2))/f),(0,1,3,2))/a
    vgdev = np.transpose((1*np.transpose(devphi_devx,(0,1,3,2))/(f*np.cos(latphicord))),(0,1,3,2))/a

    print(ugdev[264,24,6])
    print(vgdev[264,24,6])
    # ugdev = (-1*devphi_devy.T/f).T
    # vgdev = (1*devphi_devx.T/f).T
    filename = f'D:/data/MLS/zonal_deviation'
    if not os.path.exists(filename):
        os.mkdir(filename)
    nlist = ['U','V']
    for na in nlist:
        if not os.path.exists(filename+f'/{na}'):
            os.mkdir(filename+f'/{na}')         
        if not os.path.exists(filename+f'/{na}/{year}'):
            os.mkdir(filename+f'/{na}/{year}')
    for i in range(dayc):
        savefile = f'D:/data/MLS/zonal_deviation/U/{year}/{year}d{str(i+1).zfill(3)}_U_dev.npy'
        np.save(savefile, ugdev[i])
        print(f'complete to make U-dev {year}d{str(i+1).zfill(3)}')

        savefile = f'D:/data/MLS/zonal_deviation/V/{year}/{year}d{str(i+1).zfill(3)}_V_dev.npy'
        np.save(savefile, vgdev[i])
        print(f'complete to make V-dev {year}d{str(i+1).zfill(3)}')

    # ugzonal =  -1*devzonalphi_devy/f
    # uzonal = -M

    return 

def hensa(NAME,year):
    sdate = date(year, 1, 1)
    edate = date(year, 12, 31)
    allcday = (edate-sdate).days + 1

    # if NAME == 'hgt':
    #     kind ='zonal'
    # else:
    filename = f'D:/data/MLS/zonal_deviation/{NAME}/{year}'
    if not os.path.exists(filename[:11]):
        os.mkdir(filename[:11])
    if not os.path.exists(filename[:27]):
        os.mkdir(filename[:27])
    if not os.path.exists(filename[:27]+f'/{NAME}'):
        os.mkdir(filename[:27]+f'/{NAME}')         
    if not os.path.exists(filename[:27]+f'/{NAME}/{year}'):
        os.mkdir(filename[:27]+f'/{NAME}/{year}')
    dayc = (date(year,12,31)-date(year,1,1)).days + 1
    savefile = f'D:/dataMLS/MLS_griddata/move_and_complement/{NAME}/MLS-Aura_{NAME}_Mov{md}daysCom_griddata_{year}.npy'
    print(f'{NAME} 読み込み中')
    data = np.load(savefile)
    print(f'{NAME} 読み込み完了')
    zonal = np.nanmean(data,axis=3)
    dev = (data.T - zonal.T).T

    # np.save(f'../../dataJRA55/{NAME}/{NAME}_one_day_data.npy', data)
    # np.save(f'../../dataJRA55/{NAME}/{NAME}_one_day_zonal.npy', zonal)
    for i in range(dayc):
        kind = 'zonal'
        savefile = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(i+1).zfill(3)}_{NAME}_{kind}.npy'
        np.save(savefile, zonal[i])
        print(f'complete to make {NAME}-{kind} {year}d{str(i+1).zfill(3)}')

        kind = 'dev'
        savefile = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(i+1).zfill(3)}_{NAME}_{kind}.npy'
        np.save(savefile, dev[i])
        print(f'complete to make {NAME}-{kind} {year}d{str(i+1).zfill(3)}')
    return dev   
            

for year in range(startyear,endyear+1):
    for NAME in NAMElist:
        dev = hensa(NAME,year)
        print(f'finish to make zonal and deviation at {NAME} {year}')
        if NAME == 'GPH':
            gWind(dev, year)

print('finish program!!!!!!!')

