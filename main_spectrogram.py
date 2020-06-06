#-*-coding: utf-8 -*-

#import sys
#sys.path.append('/Users/yuki/Library/Mobile Documents/com~apple~CloudDocs/uki_研究用/音声解析')
import function
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
#import csv

path = 'ff_A.wav'                           #ファイルパス指定
data, samplerate = function.wavload(path)   #wavファイルを読み込み
x = np.arange(0, len(data)) / samplerate    #波形生成のための時間軸を作成

#print(samplerate)
# Fsとoverlapでスペクトログラムの分解能を調整する。
Fs = 4096                                   # フレームサイズ
overlap = 75                                # オーバーラップ率

# オーバーラップ抽出された時間波形配列
time_array, N_ave, final_time = function.ov(data, samplerate, Fs, overlap)

# ハニング窓関数をかける
time_array, acf = function.hanning(time_array, Fs, N_ave)

# FFTをかける
fft_array, fft_mean, fft_axis = function.fft_ave(time_array, samplerate, Fs, N_ave, acf)

# スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
fft_array = fft_array.T


# フィッティング直線
fh = function.fit(fft_mean)


# グラフ描画
fig = plt.figure(figsize=(7,5))

mpl.rcParams['axes.xmargin'] = 0

ax  = fig.add_subplot(311)
plt.plot(x, data, marker=",", linewidth = 0.3)

ax1 = fig.add_subplot(312)

# 3次元データをプロット
im1 = ax1.imshow(fft_array, \
    vmin = -100, vmax = 70,
    extent = [0, final_time, 0, samplerate], \
    aspect = 'auto',\
    cmap = 'gnuplot')



# カラーバーを設定
cbar1 = fig.colorbar(im1)
cbar1.set_label('SPL [dB]')


ax2 = fig.add_subplot(313)

# データをプロット
for j in range(len(fft_array[:,1])):
    plt.plot(fft_array[:,j], marker=",", linestyle = "", color="k")

plt.plot(fft_mean, marker=",", color="r", linewidth = 0.3)

#近似直線をプロット
plt.plot(fh, label="fh")

# 軸設定
ax.set_xlabel('Time [s]')
ax.set_ylabel('Power')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('SPL [dB]')

# スケール設定
ax.set_xticks(np.arange(0, 400, 50))
plt.xlim(0, 400)
ax1.set_xticks(np.arange(0, 400, 50))
plt.xlim(0, 400)
#ax1.set_yticks(np.arange(0, 20000, 2000))
#ax1.set_xlim(0, 400)
ax1.set_ylim(0, samplerate/2)
plt.xticks([0, (int(len(fft_mean)/2)*(5000/(samplerate/2))), int((len(fft_mean)/2)*(10000/(samplerate/2))), int((len(fft_mean)/2)*(15000/(samplerate/2))), int((len(fft_mean)/2)*(20000/(samplerate/2)))],
          [r'$0$', r'$5000$', r'$10000$', r'$15000$', r'$20000$'])
#ax2.set_yticks(np.arange(0, 60, 10))
ax2.set_xlim(0, len(fft_mean)/2)
#ax2.set_ylim(0, 60)

fig.tight_layout()

fig.canvas.draw()
axpos = ax.get_position() # 上の図の描画領域
axpos1 = ax1.get_position() # 下の図の描画領域
axpos2 = ax2.get_position()
#幅をax1と同じにする
ax.set_position([axpos.x0, axpos.y0, axpos1.width, axpos1.height])
ax2.set_position([axpos2.x0, axpos2.y0, axpos1.width, axpos1.height])
#fig.subplots_adjust(hspace=0.6, wspace=0.4)

#グラフ画像保存
fig.savefig("600dpi_fft_array.png",format="png", dpi=600)

# グラフ表示
plt.show()
plt.close()
