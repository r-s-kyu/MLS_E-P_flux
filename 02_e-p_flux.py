
# %%
import numpy as np
import math
from datetime import date

year = 2020
month = 9
day = 20
fday = date(year,1,1)

dc = (date(year,month,day)-fday).days + 1

namelist = ['temp','ugrd','vgrd']
# kindlist = ['data','zonal','dev']
kindlist = ['dev']

for kind in kindlist:
    for name in namelist:
        savefile = f'D:/data/MLS/{name}/{year}/{year}d{str(i+1).zfill(3)}_{name}_{kind}.npy'
        globals()[kind + name] = np.load(savefile)

pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])*100

# print(pcord)

phicord = np.arange(-90,91,1.25)*(math.pi/180.)

a = 6.37e+6
R = 287
Cp = 1004
K = R/Cp
g0 = 9.80665
ps = 100000
omega = 7.29e-5
Ts =  240
H = R*Ts/g0
rhos = ps/R/Ts
Fy = np.zeros((145,37),dtype=np.float64)*np.nan
Fz = np.zeros((145,37),dtype=np.float64)*np.nan
N_2 = 4.0e-4
f = 2*omega*np.sin(phicord)

vudev_mean = np.mean(devvgrd*devugrd,axis=2)
rho = rhos*(pcord/ps)
Fy = (((-1)*a*vudev_mean*np.cos(phicord)).T*rho).T

vTdev_mean = np.mean(devvgrd*devtemp,axis=2)
Fz = ((a*np.cos(phicord)*f*R*vTdev_mean/(N_2*H)).T*rho).T

# theta = (datatmp.T*(ps/pcord)**K).T # 3次元
# thetadev = (theta.T - np.mean(theta).T).T
# vthetadev_mean = np.mean(devvgrd*thetadev,axis=2)
# theta_z_dev = theta*K/H
# Fz = ((a*np.cos(phicord)*f*vthetadev_mean/np.mean(theta_z_dev,axis=2)).T*rho).T

vector_scale = 1.0e+6
# vector_scale = 5.0e+4
lim = 100
mabiki = 5
test1 = np.mean(Fy[-11:,:],axis=1)
test2 = np.mean(Fz[-11:,:],axis=1)
i = 5
j = 20
# print((Fy[-1*i,2::7]**2+(Fz[-1*i,2::7]*100)**2)**0.5/vector_scale)
# print((Fy[-1*j,2::7]**2+(Fz[-1*j,2::7]*100)**2)**0.5/vector_scale)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10),facecolor='white')

pcord = np.array([1000,975,950,925,900,875,850,825,800,775,750,700,
        650,600,550,500,450,400,350,300,250,225,200,175,150,125,100,70,
        50,30,20,10,7,5,3,2,1])

ycord = np.arange(-90, 90.1, 1.25)
# ylon=([ 50, 10, 5, 1])
# chei=([ "50", "10", "5", "1"])
ylon=([1000, 500, 100, 50, 10, 5, 1])
chei=(["1000", "500", "100", "50", "10", "5", "1"])
ax.set_ylim(lim,1.0e-1)
ax.set_xlim(30,80)
ax.set_yscale('log')
ax.set_yticks(ylon)
ax.set_yticklabels(chei)

for a in range(len(pcord)):
    if pcord[a] == lim:
        num = a

X,Y=np.meshgrid(ycord,pcord)
q = plt.quiver(X[num:,2::mabiki], Y[num:,2::mabiki], Fy[num:,2::mabiki], Fz[num:,2::mabiki]*100,pivot='middle',
                scale_units='xy', headwidth=5,scale=vector_scale, color='green',width=0.005)
plt.savefig('./warota.png')
# %%

