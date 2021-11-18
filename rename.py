


import calendar
import os

for year in range(2010,2021):
    for month in range(1,13):
        for day in range(1,calendar.monthrange(year,month)[1]+1):
            old = f'D:/data/MLS/e-p_flux/{year}/e-p_flux.{year}{str(month).zfill(2)+str(day).zfill(2)}.nyz.npz'
            os.remove(old)
# new = f'D:/data/MLS/e-p_flux/{year}/e-p_flux.20100101.nyz.npz'