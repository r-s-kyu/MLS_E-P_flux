
# calculate EP-flux for MLS
# 地衡風平衡より東西風を求めてからEPfluxを計算する
#%%
import numpy as np
from datetime import date
import os

syear = 2020
fyear = 2020
for year in range(syear,fyear+1) :
	print(year)
    # dayc = (date(year,12,31)-date(year,1,1)).days + 1
	dayc = (date(year,12,31)-date(year,1,1)).days + 1

	############################ 初期値
	a = 6.371e6 									# 地球の半径
	omega = 7.292e-5							# 自転速度
	lat = np.arange(-90,90.1,5)		# 緯度(degrees)
	lon = np.arange(-180,180.1,5)	# 経度(degrees)
	phi = np.radians(lat)				# 緯度(radians)
	print(phi.shape)
	ramda = np.radians(lon)				# 経度(radians)
	cosine = np.cos(phi)					# 各緯度におけるcos値
	sin = np.sin(phi)							# 各緯度におけるsin値
	tan = np.tan(phi)							# 各緯度におけるtan値
	cori = 2*omega*sin						# コリオリパラメータ
	M = a*omega*cosine						# a*omegea*cos
	g = 9.80665										# 重力加速度
	H = 7000.											# スケールハイト
	p0 = 1e5											# 標準圧力
	R = 287.											# 気体定数
	cp = 1004.										# 定圧比熱
	kappa = R/cp									# 比熱比
	Ts = 240.											# 標準温度
	rhos = p0/(Ts*R)							# 標準密度
	N = 2e-2 											# 浮力定数

	# if np.mod(year,4) == 0 :			# 閏日
	# 	day = 366
	# else :
	# 	day = 365

	nz = 55												# 鉛直層数
	ny = 37												# 南北格子点数
	nx = 73												# 東西格子点数

	f = np.broadcast_to(cori,(dayc,nz,ny))
	f4 = np.broadcast_to(f.T,(nx,ny,nz,dayc)).T
	cos = np.broadcast_to(cosine,(dayc,nz,ny))
	cos4 = np.broadcast_to(cos.T,(nx,ny,nz,dayc)).T
	print('f:',f.shape, f4.shape)
	print('cos:',cos.shape, cos4.shape)

	############################	load file
	data_GPH = f'D:/dataMLS/MLS_griddata/move_and_complement/GPH/MLS-Aura_GPH_Mov3daysCom_griddata_{year}.npy'
	data_T = f'D:/dataMLS/MLS_griddata/move_and_complement/T/MLS-Aura_T_Mov3daysCom_griddata_{year}.npy'

	f_gph = np.load(data_GPH)
	f_temp = np.load(data_T)
	print(f_gph.shape)
	print(f_temp.shape)

	gph = f_gph			# GeoPotentialHeight
	temp = f_temp			# Temperature
	# pcord = f_gph['pre']			# pressure

	#### 高度
	# z = -H*np.log(pcord*100/p0)
	# z2 = np.broadcast_to(z,(day,nz)).T
	# zcord = np.broadcast_to(z2,(nx,ny,nz,day)).T

	#### 密度
	# r0 = pcord*100/(Ts*R)
	# r2 = np.broadcast_to(r0,(day,nz)).T
	# rho = np.broadcast_to(r2,(ny,nz,day)).T

	############################	Geopotential = GPH * g
	geop = gph * g
	# print(f'geop:{geop.shape}')

	############################ U & V from Geostrophic balance
	#### U
	dphi_gph = np.gradient(geop,phi,axis=2)
	U = -dphi_gph/(f4*a)
	# U = U.T
	print(f'U:{U.shape}')

	#### V
	dram_gph = np.gradient(geop,ramda,axis=3)
	V = dram_gph/(f4*a*cos4)
	# V = V.T
	print(f'V:{V.shape}')
	filename = f'D:/data/MLS/zonal_deviation/test'
	nlist = ['U','V']

	if not os.path.exists(filename):
		os.mkdir(filename)
	for na in nlist:
		if not os.path.exists(filename+f'/{na}'):
			os.mkdir(filename+f'/{na}')         
		if not os.path.exists(filename+f'/{na}/{year}'):
			os.mkdir(filename+f'/{na}/{year}')
	for i in range(dayc):
		savefile = f'D:/data/MLS/zonal_deviation/test/U/{year}/{year}d{str(i+1).zfill(3)}_U_dev.npy'
		np.save(savefile, U[i])
		print(f'complete to make U-dev {year}d{str(i+1).zfill(3)}')

		savefile = f'D:/data/MLS/zonal_deviation/test/V/{year}/{year}d{str(i+1).zfill(3)}_V_dev.npy'
		np.save(savefile, V[i])
		print(f'complete to make V-dev {year}d{str(i+1).zfill(3)}')

	############################ T & potential temperature
	#### T
	# print(f'T:{temp.shape}')

	#### theta
	# theta = temp*np.exp(kappa*zcord/H)
	# print(f'theta:{theta.shape}')

	# ############################ zonal mean
	# zoU = np.mean(U,axis=3).T
	# zoV = np.mean(V,axis=3).T
	# zoT = np.mean(temp,axis=3).T
	# zoTh = np.mean(theta,axis=3).T

	# zonalU = np.broadcast_to(zoU,(nx,ny,nz,day)).T
	# zonalV = np.broadcast_to(zoV,(nx,ny,nz,day)).T
	# zonalT = np.broadcast_to(zoT,(nx,ny,nz,day)).T
	# zonalTh = np.broadcast_to(zoTh,(nx,ny,nz,day)).T

	# print(f'zonalU:{zonalU.shape},zonalV: {zonalV.shape},zonalT:{zonalT.shape},zonalTh:{zonalTh.shape}')

	# ############################ zonal mean deviation
	# deviU = U - zonalU
	# deviV = V - zonalV
	# deviT = temp - zonalT
	# deviTh =  theta - zonalTh
	# print(f'deviU:{deviU.shape},deviV:{deviV.shape},deviT:{deviT.shape},deviTh:{deviTh.shape}')

	# ############################	caluculate flux
	# ####	momentum
	# mflux = np.mean(deviU*deviV,axis=3)
	# ####	heat
	# hflux_T = np.mean(deviV*deviT,axis=3)
	# hflux_th = np.mean(deviV*deviTh,axis=3)

	# print(f'momentum flux:{mflux.shape}')
	# print(f'heat flux(T):{hflux_T.shape}')
	# print(f'heat flux(theta):{hflux_th.shape}')

	# ############################ z-derivative to calculate EPflux using theta
	# dz = np.gradient(zoTh,z,axis=1).T
	# print(dz.shape)

	# ############################	EPflux
	# ####  Fy / Fz
	# Fy = -rho*a*cos*mflux
	# Fz_T = rho*a*cos*f*hflux_T*R/(H*N**2)
	# Fz_th = rho*a*cos*f*hflux_th/dz
	# print(f'Fy:{Fy.shape},Fz(T):{Fz_T.shape},Fz(theta):{Fz_th.shape}')

	# #### div(F)
	# divF_T = np.gradient(Fy*cos,phi,axis=2)/(a*cos) + np.gradient(Fz_T,z,axis=1)
	# divF_th = np.gradient(Fy*cos,phi,axis=2)/(a*cos) + np.gradient(Fz_th,z,axis=1)
	# print(f'divF(T):{divF_T.shape},divF(theta):{divF_th.shape}')

	# #### DF
	# DF_T = (divF_T/(rho*a*cos)) * 24*60*60 # [m/s/day]
	# DF_th = (divF_th/(rho*a*cos)) * 24*60*60 # [m/s/day]
	# print(f'DF(T):{DF_T.shape},DF(theta):{DF_th.shape}')

	# ############################	savefile
	# save_T = f'/Users/kitagawa/desktop/data/MLS/EPflux/EPflux_T_{year}.npz'
	# save_th = f'/Users/kitagawa/desktop/data/MLS/EPflux/EPflux_theta_{year}.npz'
	# np.savez_compressed(save_T,DF=DF_T,Fy=Fy,Fz=Fz_T,pcord=pcord,xcord=lat)
	# np.savez_compressed(save_th,DF=DF_th,Fy=Fy,Fz=Fz_th,pcord=pcord,xcord=lat)

	# print(f'{year}...calculate complete')
	# print()
# %%
