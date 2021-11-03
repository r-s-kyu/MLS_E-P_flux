# %%
import numpy as np
import math
import matplotlib.pyplot as plt


name = 'hgt'
NAME = 'GPH'
year = 2020
dc = 265 # 2020/9/20
kind = 'dev'



# mlsgrid = f'D:/dataMLS/MLS_griddata/move_and_complement/{NAME}/MLS-Aura_{NAME}_Mov3daysCom_griddata_{year}.npy'
# f = open(mlsgrid,'rb')
# d = np.load(f)[dc-1]
# print(d[30])
# print(d.shape)
# zonald = np.mean(d,axis=2)
# print(zonald[30])
# print(zonald.shape)
# print(zonald)
# data = zonald
# print(len(np.where(np.isnan(zonald)[np.isnan(zonald)==True])[0]))
# print(55*37)

# data3 = (d.T-zonald.T).T
# print(dev[30,2:20])
# print(dev[30,2])
# print(dev.shape)

# data = np.mean(dev,axis=2)
# print(data[30])
# print(data.shape)
# phicord = np.arange(-90,91,5)*(math.pi/180.)
# ycord = np.arange(-90, 90.1, 5)
# prsfile = f'../text/prs_values.npy'
# with open(prsfile,'rb') as r:
#     pcord = np.load(r)

mlsfile = f'D:/data/MLS/zonal_deviation/{NAME}/{year}/{year}d{str(dc+1).zfill(3)}_{NAME}_{kind}.npy'
mls = open(mlsfile,'rb')
data1 = np.load((mls))
# data = np.mean(data,axis=2)
prsfile = f'../text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
phicord = np.arange(-90,91,5)*(math.pi/180.)
ycord = np.arange(-90, 90.1, 5)

jrafile = f'D:/data/JRA55/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
jra = open(jrafile,'rb')
data2 = np.load(jra)
# data = np.mean(data,axis=2)
phicord = np.arange(-90,91,1.25)*(math.pi/180.)
pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])
ycord = np.arange(-90, 90.1, 1.25)

# for i in range(18):
    # print(data1[31,i,40],data2[24,i,40])
print(data1[31,18,0],data2[24,72,0])
    # print(np.mean(data1,axis=2)[31,i],np.mean(data2,axis=2)[24,i])
# print(np.mean(data1,axis=2)[31,18],np.mean(data2,axis=2)[24,72])


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

min_value ,max_value = -5e-5, 5e-5
# min_value ,max_value = 160, 320
div=80      #図を描くのに何色用いるか
delta=(max_value-min_value)/div
interval=np.linspace(min_value,max_value,div+1)

# print(interval)
# interval=np.arange(min_value,abs(max_value)*2+delta,delta)[0:int(div)+1]
X,Y=np.meshgrid(ycord,pcord)
# cont = plt.contour(X,Y,zonalhgt,colors='black')
contf = plt.contourf(X,Y,data,interval,cmap='bwr',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
# contf = plt.contourf(X,Y,data,interval,cmap='jet',extend='both') #cmap='bwr_r'で色反転, extend='both'で範囲外設定
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