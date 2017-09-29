import numpy as np
import analyze_func.analyze as anly
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.fftpack import rfft
from scipy.ndimage.filters import gaussian_filter1d as gau1d
from scipy import signal
from scipy.ndimage import uniform_filter1d as meanfilt

import os
import numpy as np
from tkinter import Tk, filedialog
from analyze_func.analyze_funcs import eemd_mt
import analyze_func.analyze as anly
import analyze_func.analyze_funcs as anfn
import configuration.analyze_conf as anco
import record_func.dmc as dmc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import rfft
from scipy.ndimage import uniform_filter1d as meanfilt


def bandpassFilter(fs, highCutoff, lowCutoff, data):
    nyq = 0.5 * fs
    highCutoffNorm = highCutoff / nyq
    lowCutoffNorm = lowCutoff / nyq
    b, a = signal.butter(3, [highCutoffNorm, lowCutoffNorm], 'band', analog=False)
    filtered_data = signal.lfilter(b, a, data, 0)
    return filtered_data


def notchFilter(fs, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    nyq = fs / 2.0
    low = (freq - band / 2.0) / nyq
    high = (freq + band / 2.0) / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


# Tk().withdraw()
# folderpath = filedialog.askopenfilename()
folderpath = 'C:/github/eeg_master_thesis/Python/eeg_data/pulse/1_pulse_B_2017_05_22_09_42.csv'
fsamp = 250
noise_std = 10
channel = 3
max_imfs = 7
datarange = [3000, 4000]
data_length = datarange[1] - datarange[0]
plot_start = int(0.2 * data_length)
plot_end = int(0.7 * data_length)
plot_length = plot_end - plot_start
low = 1
high = 80
data = np.loadtxt(folderpath, delimiter=',')[datarange[0]:datarange[1], channel].T

data_mean = np.mean(data[0:int(data_length / 3)])
data_raw = data - data_mean
data_raw = notchFilter(fsamp, 8, 50, 1, 4, 'butter', data_raw)
data_raw = notchFilter(fsamp, 8, 100, 1, 4, 'butter', data_raw)

delta = bandpassFilter(fsamp, 0.9, 3, data_raw)
theta = bandpassFilter(fsamp, 5, 7, data_raw)
alpha = bandpassFilter(fsamp, 9, 12, data_raw)
beta = bandpassFilter(fsamp, 12, 25, data_raw)
gamma = bandpassFilter(fsamp, 30, 40, data_raw)

t = np.linspace(0, plot_length / fsamp, plot_length)

color_list = [[141, 211, 199], [253, 180, 98], [190, 186, 218], [251, 128, 114], [128, 177, 211]]
color_list = [[n / 256 for n in color] for color in color_list]

fig, axes = plt.subplots(5, 1)
axes[0].plot(t, delta[plot_start:plot_end] / np.max(delta[plot_start:plot_end]), label="δ, 0.5-4 Hz",
             color=color_list[0])
axes[1].plot(t, theta[plot_start:plot_end] / np.max(theta[plot_start:plot_end]), label="θ,   4-8 Hz",
             color=color_list[1])
axes[2].plot(t, alpha[plot_start:plot_end] / np.max(alpha[plot_start:plot_end]), label="α,  8-12 Hz",
             color=color_list[2])
axes[3].plot(t, beta[plot_start:plot_end] / np.max(beta[plot_start:plot_end]), label="β, 12-30 Hz", color=color_list[3])
axes[4].plot(t, gamma[plot_start:plot_end] / np.max(gamma[plot_start:plot_end]), label="ɣ,   >30 Hz",
             color=color_list[4])

for ax in axes:
    ax.set_xlim(xmin=0, xmax=t[-1])
    ax.set_ylim(ymin=-1.2, ymax=1.2)
    ax.tick_params(axis='y', which='both', labelleft='off', left='off')
    ax.legend(loc='upper right', framealpha=1, bbox_to_anchor=(1.01, 1.09), edgecolor='black',
              fancybox=False, shadow=False)
    if ax is not axes[-1]:
        ax.tick_params(axis='x',
                       which='both',
                       labelbottom='off')
fig.text(0.5, 0.02, "Time [s]", verticalalignment='center', horizontalalignment='center')
fig.text(0.02, 0.54, "Amplitude [µV]", rotation='vertical', verticalalignment='center', horizontalalignment='center')
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.99, wspace=0.01, hspace=0)
path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
filename = 'brain_waves' + '.pdf'
fig.savefig(path + filename, dpi=400)
plt.show()
