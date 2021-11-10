# %%
# program [make zonalU]
# import module 
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt
import calendar
import os
from numpy.core.fromnumeric import transpose

from numpy.lib.npyio import save


# =================初期値================
# moveMeanDay = 3
# year = 2020
startyear = 2010
endyear = 2020
# NAMElist = np.array(['GPH','T'])
# NAMElist = np.array(['GPH'])
# namelist = np.array(['hgt','tmp'])
# ====================================



# ================定数==================
latphicord = np.arange(-90, 90.1, 5)*(math.pi/180.)
a = 6.37e+6
g = 9.80665										# 重力加速度
omega = 7.29e-5
lon_cord = np.arange(-180, 180.1, 5)
lat_cord = np.arange(-90, 90.1, 5)
lat = np.radians(lat_cord)
lonlamdacord = np.radians(lon_cord)
f = 2*omega*np.sin(latphicord)
y = a**2*math.pi/(360/5)
x = (a*np.cos(latphicord))**2*math.pi/(360/5)
M = a*omega*np.cos(latphicord)  #(37,)
# ====================================


def make_zonalU(year):
    NAME = 'GPH'
    kind = 'zonal'
    dayc = (date(year,12,31)-date(year,1,1)).days + 1
    for i in range(dayc):
        file = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(i+1).zfill(3)}_{NAME}_{kind}.npy'
        zonalGPH = np.load(file)
        # print(np.where(~np.isnan(zonalGPH)))
        # print(zonalGPH.shape)
        devZonalPhi_devy = np.gradient(zonalGPH*g, latphicord, axis=1)
        # print(np.where(~np.isnan(devZonalPhi_devy)))
        # print(devZonalPhi_devy.shape)
        UgZonal = np.zeros((55,37),dtype=np.float32)*np.nan
        UgZonal[:,f!=0] = -1*devZonalPhi_devy[:,f!=0]/f[f!=0]/a
        # print(np.where(~np.isnan(UgZonal)))
        zonalU1 = -M+(M**2+2*M*UgZonal)**0.5
        # zonalU2 = -M-(M**2+2*M*UgZonal)**0.5
        # print(zonalU1)
        # print(zonalU2)
        # print(zonalGPH[20:30,10:20])
        # print(UgZonal[20:30,10:20])
        # print(zonalU1[20:30,10:20])
        # print(zonalU2[20:30,10:20])
        # print(np.where(~np.isnan(zonalU1)))
        path = f'D:/data/MLS/zonalU_from_sheerBalance'
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(path+f'/{year}'):
            os.mkdir(path+f'/{year}')
        savefile = f'D:/data/MLS/zonalU_from_sheerBalance/{year}/{year}d{str(i+1).zfill(3)}_U_zonal.npy'
        np.save(savefile,zonalU1)
        print(f'complete zonalU d{year}{str(i+1).zfill(3)}')
    print(f'complete {year}!!')

def main():
    for year in range(startyear,endyear+1):
        make_zonalU(year)
    print(f'finish program!!!')

if __name__ == '__main__':
    main()

# %%

import numpy as np

a = np.arange(30).reshape(2,3,5)
b = np.arange(15).reshape(3,5)
d = np.arange(1,11,2)
print('a')
print(a)
# print('b')
# print(b)
print(d)
c = a+b
# print(c)
print(a+d)