# %%
# compare JRA55 and MLS GPH deviation

import numpy as np
from datetime import date
import matplotlib.pyplot as plt

# ================初期値========================
pressure = 10 # 基本的に100か10か1
year = 2020
month = 10
day = 20
fday = date(year,1,1)
lim = 100
min_value ,max_value = -1000, 1000
div=100      #図を描くのに何色用いるか
# ===========================================

dc = (date(year,month,day)-fday).days + 1

# namelist = ['T','U','V','GPH']
NAME = 'GPH'
name = 'hgt'
# kindlist = ['zonal','dev']
kind = 'dev'
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-90,90]
lonrange = [-180,180]
# for kind in kindlist:
pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])
hgtJRA = f'D:/data/JRA55/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
devJRA = np.load(hgtJRA) #[37,145,288]
# jra2d = devJRA[:,:,jraLonIndex]
xcord = np.arange(-180,180,1.25)
ycord = np.arange(-90, 91,1.25)
prsfile = f'./text/prs_values.npy'
with open(prsfile,'rb') as r:
    Pcord = np.load(r)
Xcord = np.arange(-180,181,5)
Ycord = np.arange(-90, 90.1,5)
GPHMLS = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(dc+1).zfill(3)}_{NAME}_{kind}.npy'
devMLS = np.load(GPHMLS) #[55,37,73]
# mls2d = devMLS[:,:,mlsLonIndex]

c = 0
while True:
    if Pcord[c] == pressure:
        mlsprsInd = c
        break
    c += 1
d = 0
while True:
    if pcord[d] == pressure:
        jraprsInd = d
        break
    d += 1
mls2d = devMLS[jraprsInd]
jra2d = devJRA[jraprsInd]
jraw = jra2d[:,144:]
jrae = jra2d[:,:144]
jra2d = np.concatenate([jraw,jrae],1)

fig, axes = plt.subplots(1,2,figsize=(9,6),facecolor='grey',sharey=True,sharex=True)
axes[0].set_xlim(lonrange[0],lonrange[1])
axes[0].set_ylim(latrange[0],latrange[1])
# axes[0].set_yscale('log')
# axes[0].set_yticks(yticks)
# axes[0].set_yticklabels(ylabel)
axes[0].set_xticks(np.arange(-150,151,30))
axes[0].set_xticklabels(np.array(np.arange(-150,151,30),np.str_))
axes[0].set_ylabel('LAT')
axes[0].set_xlabel('LON')



interval=np.linspace(min_value,max_value,div+1)

x,y =np.meshgrid(xcord,ycord)
X,Y =np.meshgrid(Xcord,Ycord)
contf = axes[0].contourf(x,y,jra2d,interval,cmap='bwr',extend='both')
axes[0].set_title(f'JRA55',fontsize=15)

contf = axes[1].contourf(X,Y,mls2d,interval,cmap='bwr',extend='both')
axes[1].set_title(f'MLS',fontsize=15)
# contf = axes[0].contourf(X,Y,devJRA,interval,cmap='bwr',extend='both')
fig.suptitle(f'{month}/{day}/{year} prs={pressure} deviation of GeoPotentialHight ',fontsize=15)
axpos = axes[0].get_position()
cbar_ax = fig.add_axes([0.91, axpos.y0, 0.02, axpos.height])
fig.colorbar(contf,cax=cbar_ax)
plt.savefig(f'D:/picture/study/MLS/compare/GPH/pressure/d{year}{str(month).zfill(2)+str(day).zfill(2)}_prs={pressure}_devhgt_in_JRA_MLS.png')
plt.show()
print(f'finsh drawing')


# [1.0000000e+03 8.2540417e+02 6.8129205e+02 5.6234131e+02 4.6415887e+02
#  3.8311868e+02 3.1622775e+02 2.6101572e+02 2.1544347e+02 1.7782794e+02
#  1.4677992e+02 1.2115276e+02 1.0000000e+02 8.2540421e+01 6.8129204e+01
#  5.6234131e+01 4.6415890e+01 3.8311867e+01 3.1622776e+01 2.6101572e+01
#  2.1544348e+01 1.7782795e+01 1.4677993e+01 1.2115276e+01 1.0000000e+01
#  8.2540417e+00 6.8129206e+00 5.6234131e+00 4.6415887e+00 3.8311868e+00
#  3.1622777e+00 2.6101573e+00 2.1544347e+00 1.7782794e+00 1.4677993e+00
#  1.2115277e+00 1.0000000e+00 6.8129206e-01 4.6415889e-01 3.1622776e-01
#  2.1544346e-01 1.4677992e-01 1.0000000e-01 4.6415888e-02 2.1544347e-02
#  9.9999998e-03 4.6415888e-03 2.1544348e-03 1.0000000e-03 4.6415889e-04
#  2.1544346e-04 9.9999997e-05 4.6415887e-05 2.1544347e-05 9.9999997e-06]

