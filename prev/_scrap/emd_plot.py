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


def main():
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
    plot_end = int(0.8 * data_length)
    plot_length = plot_end - plot_start
    low = 1
    high = 80
    data = np.loadtxt(folderpath, delimiter=',')[datarange[0]:datarange[1], channel].T

    data_mean = np.mean(data[0:int(data_length / 3)])
    data_raw = data - data_mean + np.linspace(220, 550, data_length)
    data_filt = notchFilter(fsamp, 8, 50, 1, 4, 'butter', data_raw)
    data_filt = notchFilter(fsamp, 8, 100, 1, 4, 'butter', data_filt)
    # data_filt = bandpassFilter(fsamp, low, high, data_filt)
    # data_filt = bandpassFilter(fsamp, low, high, data_filt)



    # FOURIER
    pad_mult = 100
    samp = data_length
    xf = np.linspace(0, fsamp / 2, samp * pad_mult)
    ypad = np.zeros(samp * pad_mult)
    yzero = np.zeros(samp * pad_mult)
    ypad[:samp] = data_filt
    yf = (2 * abs(rfft(ypad)) / samp)
    yf = meanfilt(yf, 10)
    yf = meanfilt(yf, 5)
    # plt.fill_between(xf, yf, yzero, linewidth=0.65, alpha=1, zorder=-3)
    # plt.show()
    # quit()

    # EEMD
    imfs = anfn.eemd_mt(data_filt, noise_std, max_imfs)
    freq_mean = np.zeros(max_imfs)
    freq_std = np.zeros(max_imfs)
    for imf in range(max_imfs):
        amp, freq = anfn.hilbert_transform(imfs[imf], fsamp)
        freq_mean[imf] = np.mean(freq)
        freq_std[imf] = np.std(freq)


    color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
    color_list = [[n / 256 for n in color] for color in color_list]

    t_vec = np.linspace(0, plot_length / fsamp, plot_length)
    fig, ax = plt.subplots(max_imfs + 1, 1, sharex='all')
    ax[0].plot(t_vec, data_filt[plot_start:plot_end], color=color_list[0], linewidth=0.9)
    ax[0].set_xlim(xmin=0, xmax=t_vec[-10])
    for imf in range(max_imfs):
        ax[imf + 1].plot(t_vec, imfs[imf, plot_start:plot_end], color=color_list[2],linewidth=0.9)
        ax[imf].tick_params(axis='x',
                            which='both',
                            labelbottom='off')

    # ax.set_xlim(xmin=0, xmax=t_vec[-1])
    # fig.text(0.53, 0.023, "Time [s]", verticalalignment='center', horizontalalignment='center')
    # fig.text(0.02, 0.57, "Amplitude [µV]", rotation='vertical', verticalalignment='center', horizontalalignment='center')


    y_start = 0.945
    y_stop = 0.11
    x_pos = 0.965
    signal_labels = ['x(t)', 'IMF 1', 'IMF 2', 'IMF 3', 'IMF 4', 'IMF 5', 'IMF 6', 'IMF 7']
    num_labels = len(signal_labels)
    y_pos = np.linspace(y_start, y_stop, num_labels)
    signal_labels = ['x(t)', 'IMF 1', 'IMF 2', 'IMF 3', 'IMF 4', 'IMF 5', 'IMF 6', 'IMF 7']
    for i in range(num_labels):
        # Mean val
        fig.text(x_pos, y_pos[i], signal_labels[i], rotation=270, verticalalignment='center',
                 horizontalalignment='center')
    fig.text(0.55, 0.02, "Time [s]", verticalalignment='center', horizontalalignment='center')
    fig.text(0.02, 0.5, "Amplitude [µV]", rotation='vertical', verticalalignment='center', horizontalalignment='center')
    fig.subplots_adjust(wspace=0.1, left=0.11, right=0.95, bottom=0.06, top=0.99)
    fig.set_size_inches(6, 8, forward=True)
    plt.show()
    path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
    filename = 'EEMD_example_plot' + '.pdf'
    fig.savefig(path + filename, dpi=200)


if __name__ == '__main__':
    main()
