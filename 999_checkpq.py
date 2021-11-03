# %%
import h5py
from numpy.core.fromnumeric import sort
from scipy.interpolate import griddata
import numpy as np
from myModule import my_func as mf
from myModule import mlsdata_screeming as ms
from datetime import datetime
import os


# ===================初期値======================
# 繰り返し年
startyear = 2020
endyear = 2020
# 物理量選択　0:すべて　    1～4:各物理量のみ
pqchoice = 2
# 移動平均日数
movingday_num = 3 
# ====================定数====================
# 抽出物理量定数
phisical_quantity = ['O3', 'GPH', 'Temperature', 'H2O']
# その他定数
md_1 = movingday_num-1
hmd = int((movingday_num-1) /2)
# =============================================


def make_griddata(pq):
    if pq == "Temperature":
        pq2 = "T"
    else:
        pq2 = pq
    print(f'start {pq2}')
    for year in range(startyear,endyear+1):
        oldfile = f'D:/dataMLS/MLS_griddata/move_and_complement/{pq2}/MLS-Aura_{pq2}_Mov{movingday_num}daysCom_griddata_{year}.npy'
        if not os.path.exists(oldfile):
            print(f'start {year}')
            # 定数定義
            start_day = datetime(year, 1, 1)
            end_day = datetime(year,12,31)
            day_max = (end_day - start_day).days + 1
            
            pq_4d = np.zeros((day_max,55,37,73))*np.nan
            pq_4d_comp = np.zeros((day_max,55,37,73))*np.nan
            pq_4d_make = np.zeros((day_max, 55, 37, 73)) 
            pq_4d_count = np.zeros((day_max, 55, 37, 73)) 
            xcord = np.arange(-180, 180.1, 5)
            ycord = np.arange(-90, 90.1, 5)
            X, Y = np.meshgrid(xcord, ycord)

            day_ind = 0 # day = str(day_ind + 1)
            error_count = 0
            move_count = np.zeros((movingday_num-1),dtype=np.int64)
            while day_ind < day_max:
                str_day = str(day_ind+1).zfill(3)
                #データ読み込み
                file = f'D:/dataMLS/MLS/{pq2}/{year}/{year}d{str_day}.he5'
                try:    
                    print('s')
                except:
                    print(f'error {year}d{str_day}')
                    move_count = np.append(move_count,[0])
                    move_count = np.delete(move_count,0)
                    error_count += 1
                    day_ind += 1
                    continue

                move_count = np.append(move_count,[1])

                # LONの近似
                lon_origin = np.copy(lon)
                lon_near = np.arange(-180, 180.1, 5)
                for i in range(73):
                    lon_origin[np.abs(lon_origin - lon_near[i]) < 2.5] = lon_near[i]
            
                # LATの近似
                lat_origin = np.copy(lat)
                lat_near = np.arange(-90, 90.1, 5)
                for j in range(37):
                    lat_origin[np.abs(lat_origin - lat_near[j]) < 2.5] = lat_near[j]
            
                # データを各条件でscreeming
                ms.screeming(pq, data_L2gpValue, data_Status, data_L2gpPrecision, data_Quality, data_Convergence, prs)
            
                # 各時刻、各高度（等圧面）のデータを（経度5）*（緯度5）で近似したグリッドデータに入れ込む
                for n in range(len(nTimes)):
                    j = int((lat_origin[n]+90)/5)
                    i = int((lon_origin[n]+180)/5)
                    if j >= 0 or i >= 0:
                        for k in range(55):
                            if not np.isnan(data_L2gpValue[n,k]):  #np.nanではない時の条件式
                                pq_4d_make[day_ind, k, j, i] += data_L2gpValue[n, k]
                                pq_4d_count[day_ind, k, j, i] += 1
                
                # 指定日数で移動平均
                if np.sum(move_count) == movingday_num:
                    if day_ind >= md_1 :
                        countsum = np.nansum(pq_4d_count[day_ind-md_1:day_ind+1],axis=0)
                        countsum[countsum==0] = np.nan
                        pq_4d[day_ind-hmd] = np.nansum(pq_4d_make[day_ind-md_1:day_ind+1],axis=0) / countsum

                        int_mo, int_da = mf.allDayChangeToDate(year,day_ind-hmd+1)
                        print(f'complete to movingmean {int_mo}/{int_da}/{year}')
                
                        # データが欠損している部分を補完
                        for k in range(55):
                            pq_2d_m = np.ma.masked_invalid(pq_4d[day_ind-hmd,k])
                            if isinstance(np.mean(pq_2d_m), float):
                                X_valid = X[~pq_2d_m.mask]
                                Y_valid = Y[~pq_2d_m.mask]
                                pq_2d_valid = pq_2d_m[~pq_2d_m.mask]
                                pq_4d_comp[day_ind-hmd,k] = griddata((X_valid, Y_valid), pq_2d_valid, (X,Y), method='cubic')
                            else:
                                pq_4d_comp[day_ind-hmd,k] = np.nan
                        print(f"complete to complement {int_mo}/{int_da}/{year}")
                
                move_count = np.delete(move_count,0)
                day_ind += 1
            
            if error_count == day_ind:
                print(f'fail to make gridfile {pq2} {year}....')
            
            else:
                savefile = f'D:/dataMLS/MLS_griddata/move_and_complement/{pq2}/MLS-Aura_{pq2}_Mov{movingday_num}daysCom_griddata_{year}.npy'
                # dirname = f'D:/TestDir/mls/{pq}/{year}'
                if not os.path.exists(savefile[:10]):
                    os.mkdir(savefile[:10])
                if not os.path.exists(savefile[:23]):
                    os.mkdir(savefile[:23])
                if not os.path.exists(savefile[:43]):
                    os.mkdir(savefile[:43])
                if not os.path.exists(savefile[:43] + f'/{pq2}'):
                    os.mkdir(savefile[:43] + f'/{pq2}')
                np.save(savefile, pq_4d_comp)
                print(f'finish {pq2} {year}')
                print(' ')

        else:
            print(f'Already exist {pq2} {year}')
            print(' ')

if pqchoice == 0:
    for pq in phisical_quantity:
        make_griddata(pq)
else:
    pq = phisical_quantity[pqchoice-1]
    make_griddata(pq)
            

# 端の日付[1/1, 12/31]などが移動平均したことによって欠損扱いになっていることに注意


print(f'finish program')

# %%
import h5py
import numpy as np

pq = 'O3'
pq2 = 'O3'
year = 2020
str_day = '300'
file = f'D:/dataMLS/MLS/{pq2}/{year}/{year}d{str_day}.he5'
with h5py.File(file, 'r') as f:
    # 非推奨  data = f['HDFEOS']['SWATHS']['pq']['Data Fields']['pq'].value 
    # 推奨  data = f['HDFEOS']['SWATHS']['pq']['Data Fields']['pq'][()] 
    # 以前は[dataset].valueでdetasetの中身のリストや変数を取り出していたが、
    # 最近は[dataset][()]が推奨されており、.valueにすると、警告文が出る。だが、プログラムは回る。
    
    # datasetから取り出した配列は自動的にnumpy.ndarrayに納められる
    # Data Fields
    data = f['HDFEOS']['SWATHS'][pq]['Data Fields'][pq][()]
    data_L2gpValue = f['HDFEOS']['SWATHS'][pq]['Data Fields']['L2gpValue'][()]
    data_Convergence = f['HDFEOS']['SWATHS'][pq]['Data Fields']['Convergence'][()]               
    data_Precision = f['HDFEOS']['SWATHS'][pq]['Data Fields'][''+pq+'Precision'][()]
    data_L2gpPrecision = f['HDFEOS']['SWATHS'][pq]['Data Fields']['L2gpPrecision'][()]
    data_Quality = f['HDFEOS']['SWATHS'][pq]['Data Fields']['Quality'][()]
    data_Status = f['HDFEOS']['SWATHS'][pq]['Data Fields']['Status'][()]   
    # Geolocation Fields
    lat = f['HDFEOS']['SWATHS'][pq]['Geolocation Fields']['Latitude'][()]
    lon = f['HDFEOS']['SWATHS'][pq]['Geolocation Fields']['Longitude'][()]  
    prs = f['HDFEOS']['SWATHS'][pq]['Geolocation Fields']['Pressure'][()] 
    nLevels = f['HDFEOS']['SWATHS'][pq]['nLevels'][()]
    nTimes = f['HDFEOS']['SWATHS'][pq]['nTimes'][()]
    # print(data)

print(prs)
savefile = f'./text/prs_values.npy'
with open(savefile,'wb') as f:
    np.save(savefile,prs)
# print(lon.shape)

# # LONの近似
# lon_origin = np.copy(lon)
# lon_near = np.arange(-180, 180.1, 5)
# for i in range(73):
#     lon_origin[np.abs(lon_origin - lon_near[i]) < 2.5] = lon_near[i]

# print(lon_origin)
# print(lon_origin.shape)


# sort_lon = np.sort(lon_origin)
# print(sort_lon)
# %%
prsfile = f'./text/prs_values.npy'
with open(prsfile,'rb') as r:
    pcord = np.load(r)
print(pcord)
# %%
