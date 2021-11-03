# %%
from matplotlib.pyplot import savefig
import numpy as np
import math
from datetime import date
import os

year = 2020
name = 'hgt'
startyear =2010
endyear =2010

def hensa(name,year):

    sdate = date(year, 1, 1)
    edate = date(year, 12, 31)
    allcday = (edate-sdate).days + 1

    kind ='dev'
    filename = f'D:/data/JRA55/{name}/{year}/{year}d{str(1).zfill(3)}_{name}_{kind}.npy'
    if not os.path.exists(filename[:13]):
        os.mkdir(filename[:13])
    if not os.path.exists(filename[:13]+f'/{name}'):
        os.mkdir(filename[:13]+f'/{name}')         
    if not os.path.exists(filename[:13]+f'/{name}/{year}'):
        os.mkdir(filename[:13]+f'/{name}/{year}')
    if not os.path.exists(filename):
        dayc = (date(year,12,31)-date(year,1,1)).days + 1
        file = f'D:/data/JRA55/{name}/anl_p_{name}.{year}.bin'
        f = open(file, 'rb')
        print(f'{name} 読み込み中')
        array = np.fromfile(f,dtype='>f').reshape(allcday,37,145,288)
        print(f'{name} 読み込み完了')
        f.close()
        data = array[:,:,::-1]
        zonal = np.mean(data,axis=3)
        dev = (data.T - zonal.T).T
        # np.save(f'../../dataJRA55/{name}/{name}_one_day_data.npy', data)
        # np.save(f'../../dataJRA55/{name}/{name}_one_day_zonal.npy', zonal)
        for i in range(dayc):
            savefile = f'D:/data/JRA55/{name}/{year}/{year}d{str(i+1).zfill(3)}_{name}_{kind}.npy'
            np.save(savefile, dev[i])
        print(f'complete to make {name}-{kind} {year}')
            
    else:
        print(f'already exist {name}-{kind} {year}')
    return 
for year in range(startyear,endyear+1):
    hensa(name,year)

print('finish!')