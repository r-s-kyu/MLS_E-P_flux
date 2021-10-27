
# %%
import numpy as np
import math
from datetime import date

year = 2020
month = 9
day = 20
fday = date(year,1,1)

dc = (date(year,month,day)-fday).days + 1
pcord = 

namelist = ['T','U','V']
# kindlist = ['data','zonal','dev']
kind = 'dev'

def epFlux(dc,year):
    for name in namelist:
        savefile = f'D:/data/JRA55/{name}/{year}/{year}d{str(dc).zfill(3)}_{name}_{kind}.npy'
        globals()[kind + name] = np.load(savefile)

    dp = np.array([])
    for i in range(len(pcord)-1):
        dp = np.append(dp, pcord[i]-pcord[i+1])
    dp = np.append(dp,1)

    vudev_mean = np.mean(devV*devU,axis=2)
    Fy = (((-1)*a*vudev_mean*np.cos(phicord)).T*rho).T
    devFy = np.gradient(Fy*np.cos(phicord), phicord,axis=1)
    z = -H*np.log(pcord*100/ps)
    vTdev_mean = np.mean(devV*devtmp,axis=2)
    Fz = ((a*np.cos(phicord)*f*R*vTdev_mean/(N_2*H)).T*rho).T
    devFz = np.gradient(Fz,z,axis=0)
    nablaF = devFy/(a*np.cos(phicord)) + devFz
    # fzmean = np.mean(np.mean(nablaF))
    nablaF = ((nablaF/(a*np.cos(phicord))).T/rho).T
    nablaF = nablaF*60*60*24
    # fmean = np.mean(np.mean(nablaF))
    return Fy, Fz, nablaF

# theta = (datatmp.T*(ps/pcord)**K).T # 3次元
# thetadev = (theta.T - np.mean(theta).T).T
# vthetadev_mean = np.mean(devV*thetadev,axis=2)
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

