
# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import date

name = 'vgrd'
NAME = 'V'
year = 2020
dc = 265 # 2020/9/20
kind = 'dev'
sdate = date(year, 1, 1)
edate = date(year, 12, 31)
allcday = (edate-sdate).days + 1

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

mlsgrid = f'D:/dataMLS/MLS_griddata/move_and_complement/{NAME}/MLS-Aura_{NAME}_Mov3daysCom_griddata_{year}.npy'
mls = open(mlsgrid,'rb')
data = np.load((mls))
data1 = np.mean(data[dc-1],axis=2)
# print(data.shape)
prsfile = f'../text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
phicord = np.arange(-90,91,5)*(math.pi/180.)
ycord = np.arange(-90, 90.1, 5)
print(data1.shape)


jragrid = f'D:/data/JRA55/tmp/anl_p_{name}.{year}.bin'
f = open(jragrid, 'rb')
# dt = np.dtype(('>f',(37,145,288)))
array = np.fromfile(f,dtype='>f').reshape(allcday,37,145,288)[:,:,::-1]
f.close()
data2 = np.mean(array[dc-1],axis=2)
print(data2.shape)
phicord = np.arange(-90,91,1.25)*(math.pi/180.)
pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])
ycord = np.arange(-90, 90.1, 1.25)

for i in range(37):
    # print(data1[31,i,40],data2[24,i,40],data3[31,i,40])
    print(data1[31,i],data2[24,i])

# mlsfile = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(dc).zfill(3)}_{NAME}_{kind}.npy'
# mls = open(mlsfile,'rb')
# data = np.load((mls))
# data = np.mean(data,axis=2)
# prsfile = f'../text/prs_values.npy'
# with open(prsfile,'rb') as r:
#     pcord = np.load(r)
# phicord = np.arange(-90,91,5)*(math.pi/180.)
# ycord = np.arange(-90, 90.1, 5)

# jrafile = f'D:/data/JRA55/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
# jra = open(jrafile,'rb')
# data = np.load(jra)
# data = np.mean(data,axis=2)
# phicord = np.arange(-90,91,1.25)*(math.pi/180.)
# pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
#         650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
#         50,30,20,10,7,5,3,2,1])
# ycord = np.arange(-90, 90.1, 1.25)

# %%
fig, ax = plt.subplots(figsize=(6, 6),facecolor='white')
lim=1000
ylon=([1000, 500, 100, 50, 10, 5, 1,0.1])
chei=(["1000", "500", "100", "50", "10", "5", "1","0.1"])
# ylon=([100, 50, 10, 5, 1, 0.1])
# chei=(["100", "50", "10", "5", "1", "0.1"])
ax.set_ylim(lim,0.1)
ax.set_xlim(-80,-30)
ax.set_yscale('log')
ax.set_yticks(ylon)
ax.set_yticklabels(chei)
ax.set_xlabel('LAT')
ax.set_ylabel('pressure')
# ax.imshow(vmin=1e-6,vmax=1e+6)

# num = 0
for a in range(len(pcord)):
    if pcord[a] == lim:
        num = a

min_value ,max_value = 160, 320
div=80      #図を描くのに何色用いるか
delta=(max_value-min_value)/div
interval=np.linspace(min_value,max_value,div+1)

# print(interval)
# interval=np.arange(min_value,abs(max_value)*2+delta,delta)[0:int(div)+1]
X,Y=np.meshgrid(ycord,pcord)
# cont = plt.contour(X,Y,zonalhgt,colors='black')
# contf = plt.contourf(X,Y,data,interval,cmap='bwr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
contf = plt.contourf(X,Y,data,interval,cmap='jet',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
# contf = plt.contourf(X,Y,data,extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
# q = plt.quiver(X[num:,2::mabiki], Y[num:,2::mabiki], Fy[num:,2::mabiki], Fz[num:,2::mabiki]*100,pivot='middle',
#                 scale_units='xy', headwidth=5,scale=vector_scale, color='green',width=0.005)
# plt.title(f'{month}/{day}/{year} E-Pflux and ∇',fontsize=20)
plt.colorbar(contf)
# if not os.path.exists(f'./picture'):
#     os.mkdir(f'./picture')
# if not os.path.exists(f'./picture/day'):
#     os.mkdir(f'./picture/day')
# file =  f'./picture/day/{year}/{year}{str(month).zfill(2)+str(day).zfill(2)}_E-Pflux_from_MLS.png'
# if not os.path.exists(file[:13]+f'/{year}'):
#     os.makedirs(file[:13]+f'/{year}')
# plt.savefig(file)