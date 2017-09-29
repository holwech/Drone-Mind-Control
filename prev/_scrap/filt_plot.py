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


def notchFilter(fs, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter
    nyq = fs / 2.0
    low = (freq - band / 2.0) / nyq
    high = (freq + band / 2.0) / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def bandpassFilter(fs, highCutoff, lowCutoff, data):
    nyq = 0.5 * fs
    highCutoffNorm = highCutoff / nyq
    lowCutoffNorm = lowCutoff / nyq
    b, a = signal.butter(3, [highCutoffNorm, lowCutoffNorm], 'band', analog=False)
    filtered_data = signal.lfilter(b, a, data, 0)
    return filtered_data


# Tk().withdraw()
# folderpath = filedialog.askopenfilename()
folderpath = 'C:/github/eeg_master_thesis/Python/eeg_data/pulse/1_pulse_B_2017_05_22_09_42.csv'

channel = 3
datarange = [3000, 4000]
data_length = datarange[1] - datarange[0]
plot_start = int(0.15 * data_length)
plot_end = int(0.35 * data_length)
plot_length = plot_end - plot_start
low = 1
high = 40
data = np.loadtxt(folderpath, delimiter=',')[datarange[0]:datarange[1], channel].T

data_mean = np.mean(data[0:int(data_length / 3)])
data_raw = data - data_mean
data_filt = bandpassFilter(250, low, high, data_raw)
data_filt = notchFilter(250, 3, 50, 1, 3, 'butter', data_filt)
data_filt = bandpassFilter(250, low, high, data_filt)

t_vec = np.linspace(0, plot_length / 250, plot_length)
fig, ax = plt.subplots(1, 1)
ax.plot(t_vec, data_raw[plot_start:plot_end], label='Raw')
ax.plot(t_vec, data_filt[plot_start:plot_end], label='Filtered')
ax.set_xlim(xmin=0, xmax=t_vec[-1])
ax.legend()
fig.text(0.53, 0.023, "Time [s]", verticalalignment='center', horizontalalignment='center')
fig.text(0.02, 0.57, "Amplitude [µV]", rotation='vertical', verticalalignment='center', horizontalalignment='center')
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.99)
fig.set_size_inches(6, 3, forward=True)
plt.show()
path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
filename = 'Raw_filtered_eeg' + '.pdf'
# fig.savefig(path + filename, dpi=200)
