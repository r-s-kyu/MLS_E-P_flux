# %%
# compare JRA55 and MLS U deviation

import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import os

# ================初期値========================
longitude = -20 #5の倍数を指定
year = 2020
month = 8
day = 10
fday = date(year,1,1)
lim = 100
min_value ,max_value = -50, 50
div=40      #図を描くのに何色用いるか
# ===========================================
if longitude>=0:
    jraLonIndex = int(longitude/1.25-1)
else:
    jraLonIndex = int((360+longitude)/1.25-1)
mlsLonIndex = int((longitude+180)/5)
dc = (date(year,month,day)-fday).days + 1

NAME = 'U'
name = 'ugrd'
# kindlist = ['zonal','dev']
kind = 'dev'
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-90,90]
# for kind in kindlist:
Pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])
ugrdJRA = f'D:/data/JRA55/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
devJRA = np.load(ugrdJRA) #[37,145,288]
jra2d = devJRA[:,:,jraLonIndex]

Ycord = np.arange(-90, 91,1.25)
prsfile = f'./text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
ycord = np.arange(-90, 90.1,5)
UMLS = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(dc+1).zfill(3)}_{NAME}_{kind}.npy'
devMLS = np.load(UMLS) #[55,37,73]
mls2d = devMLS[:,:,mlsLonIndex]


fig, axes = plt.subplots(1,2,figsize=(7,6),facecolor='grey',sharex=True,sharey=True)
axes[0].set_ylim(lim,1.0)
axes[0].set_xlim(latrange[0],latrange[1])
axes[0].set_yscale('log')
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(ylabel)
axes[0].set_xlabel('LAT')
axes[0].set_ylabel('pressure')



interval=np.linspace(min_value,max_value,div+1)

X,Y =np.meshgrid(Ycord,Pcord)
x,y =np.meshgrid(ycord,pcord)
contf = axes[0].contourf(X,Y,jra2d,interval,cmap='bwr',extend='both')
axes[0].set_title(f'JRA55',fontsize=15)

contf = axes[1].contourf(x,y,mls2d,interval,cmap='bwr',extend='both')
axes[1].set_title(f'MLS',fontsize=15)
# contf = axes[0].contourf(X,Y,devJRA,interval,cmap='bwr',extend='both')
fig.suptitle(f'{month}/{day}/{year} lon={longitude} deviation of U ',fontsize=15)
axpos = axes[0].get_position()
cbar_ax = fig.add_axes([0.91, axpos.y0, 0.02, axpos.height])
fig.colorbar(contf,cax=cbar_ax)
path = f'D:/picture/study/MLS/compare'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(path+f'/U'):
    os.mkdir(path+f'/U')
if not os.path.exists(path+f'/U'+f'/longitude'):
    os.mkdir(path+f'/U'+f'/longitude')

plt.savefig(f'D:/picture/study/MLS/compare/U/longitude/d{year}{str(month).zfill(2)+str(day).zfill(2)}_lon={longitude}_devugrd_in_JRA_MLS.png')
plt.show()

print(f'finsh drawing')
