import numpy as np
import math
from datetime import date
import os

# ===================初期値======================
moveMeanDay = 3
year = 2020
startyear = 2010
endyear = 2020
NAMElist = np.array(['GPH','T'])
namelist = np.array(['ght','tmp'])
# ===================定数=====================

md = moveMeanDay
pq = 'GPH'

latphicord = np.arange(-90, 90.1, 5)*(math.pi/180.)
a = 6.37e+6

omega = 7.29e-5
lon_cord = np.arange(-180, 180.1, 5)
lat_cord = np.arange(-90, 90.1, 5)
f = 2*omega*np.sin(latphicord)
y = a**2*math.pi/(360/5)
x = (a*np.cos(latphicord))**2*math.pi/(360/5)
M = a*omega*np.cos(latphicord)


def gWind(phidev, phizonal,year):
    devphi_devx = np.zeros((37,73))
    for i in range(len(x)):
        devphi_devx[i] = np.gradient(phidev, x[i], axis=1)
    devphi_devy = np.gradient(phidev, y, axis=0)
    devzonalphi_devy = np.gradient(phizonal, y)
    ugdev = (-1*devphi_devy.T/f).T
    vgdev = (1*devphi_devx.T/f).T
    dayc = (date(year,12,31)-date(year,1,1)).days + 1
    filename = f'D:/data/MLS'
    if not os.path.exists(filename):
        os.mkdir(filename)
    nlist = ['U','V']
    for na in nlist:
        if not os.path.exists(filename+f'/{na}'):
            os.mkdir(filename+f'/{na}')         
        if not os.path.exists(filename+f'/{na}/{year}'):
            os.mkdir(filename+f'/{na}/{year}')
    for i in range(dayc):
        savefile = f'D:/data/MLS/U/{year}/{year}d{str(i+1).zfill(3)}_U_dev.npy'
        np.save(savefile, ugdev[i])
        print(f'complete to make U-dev {year}')

        savefile = f'D:/data/MLS/V/{year}/{year}d{str(i+1).zfill(3)}_V_dev.npy'
        np.save(savefile, vgdev[i])
        print(f'complete to make V-dev {year}')

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
    if not os.path.exists(filename):
        dayc = (date(year,12,31)-date(year,1,1)).days + 1
        savefile = f'D:/dataMLS/MLS_griddata/move_and_complement/{NAME}/MLS-Aura_{NAME}_Mov{md}daysCom_griddata_{year}.npy'
        print(f'{NAME} 読み込み中')
        data = np.load(savefile)
        print(f'{NAME} 読み込み完了')
        zonal = np.mean(data,axis=3)
        dev = (data.T - zonal.T).T

        # np.save(f'../../dataJRA55/{NAME}/{NAME}_one_day_data.npy', data)
        # np.save(f'../../dataJRA55/{NAME}/{NAME}_one_day_zonal.npy', zonal)
        for i in range(dayc):
            kind = 'zonal'
            savefile = f'D:/data/MLS/{NAME}/{year}/{year}d{str(i+1).zfill(3)}_{NAME}_{kind}.npy'
            np.save(savefile, zonal[i])
            print(f'complete to make {NAME}-{kind} {year}')

            kind = 'dev'
            savefile = f'D:/data/MLS/{NAME}/{year}/{year}d{str(i+1).zfill(3)}_{NAME}_{kind}.npy'
            np.save(savefile, dev[i])
            print(f'complete to make {NAME}-{kind} {year}')
        return zonal, dev   
            
    else:
        print(f'already exist {NAME} {year}')
        return None, None

for year in range(startyear,endyear+1):
    for NAME in NAMElist:
        zonal,dev = hensa(NAME,year)
        if not zonal == None and not dev == None:
            print(f'finish to make zonal and deviation at {NAME} {year}')
            if NAME == 'HGT':
                gWind(zonal,dev, year)


