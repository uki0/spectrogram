#-*-coding: utf-8 -*-

import numpy as np
import scipy
from scipy import signal
from scipy import fftpack
import soundfile as sf

def wavload(path):
    data, samplerate = sf.read(path)
    return data, samplerate

def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate                     # 全データ長
    Fc = Fs / samplerate                            # フレーム周期
    x_ol = Fs * (1 - (overlap / 100))               # オーバーラップ時のフレームずらし幅
    N_ave = int((Ts - (Fc * (overlap / 100))) /\
                (Fc * (1 - (overlap / 100))))       # 抽出するフレーム数（平均化に使うデータ個数）

    time_array = []                                      # 抽出したデータを入れる空配列の定義

    # forループでデータを抽出
    for i in range(N_ave):
        ps = int(x_ol * i)                          # 切り出し位置をループ毎に更新
        time_array.append(data[ps:ps + Fs:1])            # 切り出し位置psからフレームサイズ分抽出して配列に追加
        final_time = (ps + Fs)/samplerate           #切り出したデータの最終時刻
    return time_array, N_ave, final_time                 # オーバーラップ抽出されたデータ配列とデータ個数、最終時間を戻り値にする


def hanning(time_array, Fs, N_ave):
    han = signal.hann(Fs)                           # ハニング窓作成
    acf = 1 / (sum(han) / Fs)                       # 振幅補正係数(Amplitude Correction Factor)

    # オーバーラップされた複数時間波形全てに窓関数をかける
    for i in range(N_ave):
        time_array[i] = time_array[i] * han         # 窓関数をかける

    return time_array, acf


def db(x, dBref):
    y = 20 * np.log10(x / dBref)                   # 変換式
    return y                                       # dB値を返す


def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    fft_axis = np.linspace(0, samplerate, Fs)      # 周波数軸を作成
    #a_scale = aweightings(fft_axis)               # 聴感補正曲線を計算

    # FFTをして配列にdBで追加、窓関数補正値をかけ、(Fs/2)の正規化を実施。
    for i in range(N_ave):
        fft_array.append(db\
                        (acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2))\
                        , 2e-5))

    fft_array = np.array(fft_array)                # 型をndarrayに変換
    fft_mean = np.mean(fft_array, axis=0)          # 全てのFFT波形の平均を計算

    return fft_array, fft_mean, fft_axis


def fit(fft_mean):
    #xの値を生成
    x = np.arange(len(fft_mean)/2)

    #　フィッティング
    #線形回帰

    fft_mean1 = np.delete(fft_mean, np.s_[:int(len(x))])
    fft_mean1 = fft_mean1[::-1]
    #print(fft_mean1)
    #print(len(fft_mean1))
    #print(len(x))
    a, b = np.polyfit(x, fft_mean1, 1)
    #print(a,b)

    # フィッティング直線
    fh = a * x + b

    return fh
