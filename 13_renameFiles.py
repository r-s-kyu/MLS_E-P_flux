# %%
import os
import calendar
# ====================初期値===================
year = 2020

# ====================描画値===================
vector_scale = 8.0e+5
lim = 100
mabiki = 1
yticks=([100, 50, 10, 5, 1, 0.1])
ylabel=(["100", "50", "10", "5", "1", "0.1"])
latrange = [-80,-30]

def main():
    for month in range(1,13):
        newname = f'D:/picture/study/MLS/monthYearMean/zonalU/month{month}Mean_2010to2019and{year}_E-Pflux_zonalU.png'
        oldname = f'D:/picture/study/MLS/monthYearMean/zonalU/month{month}Mean_2010to2019and{year}_E-Pflux_zonalT.png'
        os.rename(oldname,newname)

main()