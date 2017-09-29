import matplotlib.pyplot as plt
import configuration.record_conf as conf
from matplotlib.patches import Rectangle
from scipy import signal
import numpy as np
import tkinter as tki
from tkinter.filedialog import askopenfilename

tki.Tk().withdraw()
filename = askopenfilename()
*rest, file = filename.split('/')
print(f"Loading file: {file}")

BOARD_FREQUENCY = 250

channels = 8

x_start_plot = 1000
y_max = 150
y_min = -y_max



# Filters
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
    b, a = signal.butter(1, [highCutoffNorm, lowCutoffNorm], 'band', analog=False)
    filtered_data = signal.lfilter(b, a, data, 0)
    return filtered_data


def add_RGB_indication():
    color_list = ['red', 'green', 'blue']
    currentAxis = plt.gca()
    pulse_started = 0
    # Add dotted lines at timestamp of the external trigger
    for index_column, aux_column in enumerate(aux_data):
        for index, value in enumerate(aux_column):
            if pulse_started:
                if value == 0:
                    pulse_started = 0
                    rectangle_end = index
                    # print(rectangle_end)
                    currentAxis.add_patch(
                        Rectangle((rectangle_start, y_min), rectangle_end - rectangle_start, abs(y_min) + abs(y_max),
                                  alpha=0.2, facecolor=color_list[index_column]))
            if value > 0 and pulse_started == 0:
                aux_data[index_column, index] = t_vec[index]
                pulse_started = 1
                rectangle_start = index
                # print(rectangle_start)


data = np.loadtxt(filename, delimiter=',').T

ch_data = data[:8, :]
aux_data = data[8:, :]
ch_filt_data = np.zeros(ch_data.shape)

samples = len(data[0, :])
# t_vec = np.linspace(0, samples / BOARD_FREQUENCY, samples)
t_vec = range(samples)

for i in range(channels):
    ch_filt_data[i, :] = bandpassFilter(BOARD_FREQUENCY, 1, 25, ch_data[i, :])
    ch_filt_data[i, :] = notchFilter(BOARD_FREQUENCY, 3, 50, 1, 3, 'butter', ch_filt_data[i, :])


# ch_filt_data = ch_data
line_color = ['sienna', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'grey']
for i in range(channels):
    plt.subplot(ch_data.shape[0], 1, i + 1)
    plt.plot(t_vec[x_start_plot:], ch_filt_data[i, x_start_plot:], linewidth=1.0, color=line_color[i])
    if i < channels - 1:
        plt.tick_params(axis='x',
                        which='both',
                        bottom='off',
                        labelbottom='off')

    plt.tight_layout()
    axis = plt.gca()
    add_RGB_indication()
    # axis.set_ylim([y_min, y_max])
    axis.set_xlim([0, t_vec[-1]])
    # axis.set_xlim([10000, 10200])


plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
