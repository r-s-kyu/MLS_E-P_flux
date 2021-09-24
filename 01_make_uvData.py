import numpy as np
import math
from datetime import date
import os

# ===================初期値======================
moveMeanDay = 3
year = 2020
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


def gWind(phidev, phizonal):
    devphi_devx = np.zeros((37,73))
    for i in range(len(x)):
        devphi_devx[i] = np.gradient(phidev, x[i], axis=1)
    devphi_devy = np.gradient(phidev, y, axis=0)
    devzonalphi_devy = np.gradient(phizonal, y)
    ugdev = (-1*devphi_devy.T/f).T
    vgdev = (1*devphi_devx.T/f).T
    ugzonal =  -1*devzonalphi_devy/f
    uzonal = -M
    return ugdev, vgdev, 

def hensa(name,year):
    sdate = date(year, 1, 1)
    edate = date(year, 12, 31)
    allcday = (edate-sdate).days + 1

    # if name == 'hgt':
    #     kind ='zonal'
    # else:
    filename = f'D:/data/MLS/zonal_deviation/{name}/{year}'
    if not os.path.exists(filename[:11]):
        os.mkdir(filename[:11])
    if not os.path.exists(filename[:27]):
        os.mkdir(filename[:27])
    if not os.path.exists(filename[:27]+f'/{name}'):
        os.mkdir(filename[:27]+f'/{name}')         
    if not os.path.exists(filename[:27]+f'/{name}/{year}'):
        os.mkdir(filename[:27]+f'/{name}/{year}')
    if not os.path.exists(filename):
        dayc = (date(year,12,31)-date(year,1,1)).days + 1
        savefile = f'D:/dataMLS/MLS_griddata/move_and_complement/{pq}/MLS-Aura_{pq}_Mov{md}daysCom_griddata_{year}.npy'
        print(f'{name} 読み込み中')
        data = np.load(savefile)
        print(f'{name} 読み込み完了')
        zonal = np.mean(data,axis=3)
        dev = (data.T - zonal.T).T

        
        # np.save(f'../../dataJRA55/{name}/{name}_one_day_data.npy', data)
        # np.save(f'../../dataJRA55/{name}/{name}_one_day_zonal.npy', zonal)
        for i in range(dayc):
            kind = 'zonal'
            savefile = f'D:/data/MLS/{name}/{year}/{year}d{str(i+1).zfill(3)}_{name}_{kind}.npy'
            np.save(savefile, zonal[i])
            print(f'complete to make {name}-{kind} {year}')

            kind = 'dev'
            savefile = f'D:/data/MLS/{name}/{year}/{year}d{str(i+1).zfill(3)}_{name}_{kind}.npy'
            np.save(savefile, dev[i])
            print(f'complete to make {name}-{kind} {year}')
            
    else:
        print(f'already exist {name}-{kind} {year}')
    return 


for year in range(startyear,endyear+1):
    for name in namelist:
        hensa(name,year)

print('finish!')
# %%

a = f'D:/data/MLS/zonal_deviation'
print(a[:11])
print(a[:27])

# %%
import numpy as np
apple = np.arange(10).reshape(2,5)
print(apple)

print()