# %%
import os
import calendar
# ====================初期値===================
startyear = 2010
endyear = 2020

# ====================描画値===================
vector_scale = 8.0e+5
lim = 100
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-80,-30]

def main():
    for year in range(startyear,endyear+1):
        for month in range(1,13):
            for day in range(1,calendar.monthrange(year,month)[1]+1):
                dates = f'{str(year).zfill(4)+str(month).zfill(2)+str(day).zfill(2)}'
                newname = f'D:/data/MLS/e-p_flux/{year}/e-p_flux.{dates}.npz'
                oldname = f'D:/data/MLS/e-p_flux/{year}/e-p_flux.{dates}.nyz.npz'
                os.rename(oldname,newname)

main()