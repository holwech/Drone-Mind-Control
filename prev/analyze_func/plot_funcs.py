# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.patches import Rectangle
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
import configuration.analyze_conf as anco
import analyze_func.analyze_funcs as anfn

plt.rc('text', usetex=False)


def power_plot(freq_data, amp_data, imfs):
    start_time = -anco.prefetch / anco.BOARD_FREQUENCY
    stop_time = (anco.event_length + anco.postfetch) / anco.BOARD_FREQUENCY
    t_vec = np.linspace(start_time, stop_time, freq_data.shape[-1])
    epoch_length = amp_data.shape[-1]
    color_list = [(0.95, 0.35, 0.4), (0.45, 0.9, 0.11), (0.2, 0.5, 0.9)]
    amp_imf_avg = np.zeros((len(color_list), epoch_length))
    fig, ax = plt.subplots()
    for color in range(len(color_list)):
        for sample in range(epoch_length):
            amp_imf_avg[color, sample] = np.average(amp_data[color, imfs, sample])
        amp_imf_avg[color] = amp_imf_avg[color] ** 2
        # amp_imf_avg[color] = amp_imf_avg[color] / np.max(amp_imf_avg[color])
        ax.plot(t_vec, amp_imf_avg[color], color=color_list[color])
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel("Time [$s$]")
    ax.set_ylabel("Power [$\mu V^2$]")
    plt.show()


def hilbert_spectrum_subj_full_plot(freq_data, amp_data, subj, max_scale):
    log_min_freq = np.log10(anco.hht_freq_min)
    log_max_freq = np.log10(anco.hht_freq_max)
    freq_bins = np.logspace(log_min_freq, log_max_freq, anco.num_freq_bins, base=10)
    colors = freq_data.shape[0]
    channels = freq_data.shape[1]
    spectrum = np.zeros((colors, channels, anco.num_freq_bins, freq_data.shape[-1]))
    for color in range(colors):
        for ch in range(channels):
            spectrum[color, ch] = hilbert_spectrum_v2(freq_data[color, ch],
                                                      amp_data[color, ch],
                                                      freq_bins,
                                                      [0, 7, 8, 9])
    # Time vector
    start_time = -anco.prefetch / anco.BOARD_FREQUENCY
    stop_time = (anco.event_length + anco.postfetch) / anco.BOARD_FREQUENCY
    t_vec = np.linspace(start_time, stop_time, freq_data.shape[-1])

    # Plot scale and array
    scale_max = np.max(spectrum)
    scale_max = max_scale
    scale = np.logspace(np.log10(10), np.log10(scale_max), 9, dtype='int')
    spectrum = anfn.hilbert_power_scale_limit_full(spectrum, scale_max)

    # Single HHT plot code
    # fig, axis = plt.subplots(nrows=1, ncols=1, sharex='all', sharey='all')
    # im = axis.contourf(t_vec, freq_bins, spectrum[1, 5], scale, cmap='magma_r', zorder=-3)
    # axis.set_yscale('log')
    # axis.axvline(x=0, color='grey', linestyle='--', linewidth=1)
    # axis.axvline(x=1, color='grey', linestyle='--', linewidth=1)
    # axis.set_rasterization_zorder(-2)
    #
    # fig.text(0.02, 0.5, r"Frequency [$Hz$]", rotation='vertical', verticalalignment='center',
    #          horizontalalignment='center')
    # fig.text(0.5, 0.02, "Time [$s$]", verticalalignment='center', horizontalalignment='center')
    # fig.text(0.96, 0.5, r"Power [$\mu V^2$]", rotation=-90, verticalalignment='center',
    #          horizontalalignment='center')
    # cbar_ax = fig.add_axes([0.89, 0.1, 0.013, 0.85])  # [left, bottom, width, height]
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.solids.set_rasterized(True)
    # fig.subplots_adjust(left=0.085, right=0.87, bottom=0.1, top=0.95)
    # fig.set_size_inches(7, 4, forward=True)
    # path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
    # filename = 'HHT_subject_SINGLE_' + subj + '.pdf'
    # fig.savefig(path + filename, dpi=200)
    # plt.show()
    # quit()

    fig, axis = plt.subplots(nrows=channels, ncols=colors, sharex='all', sharey='all')
    for i, ax_row in enumerate(axis):
        for j, ax in enumerate(ax_row):
            im = ax.contourf(t_vec, freq_bins, spectrum[j, i], scale, cmap='magma_r', zorder=-3)
            ax.set_yscale('log')
            ax.tick_params(axis='y', pad=-0.1)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=9)
            ax.axvline(x=0, color='grey', linestyle='--', linewidth=1)
            ax.axvline(x=1, color='grey', linestyle='--', linewidth=1)
            if i < channels - 1:
                ax.tick_params(axis='x',
                               which='both',
                               bottom='off',
                               labelbottom='off')
            ax.set_rasterization_zorder(-2)

    # Figure lines for label separation
    line_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    line_ax.axis('off')
    x, y = np.array([[0.05, 0.05, 0.87], [0.05, 0.965, 0.965]])
    line = lines.Line2D(x, y, lw=0.7, color='black')
    line_ax.add_line(line)

    # Axis labeling
    fig.text(0.076, 0.5, r"Frequency [$Hz$]", rotation='vertical', verticalalignment='center',
             horizontalalignment='center')
    fig.text(0.5, 0.015, "Time [$s$]", verticalalignment='center', horizontalalignment='center')
    fig.text(0.96, 0.5, r"Power [$\mu V^2$]", rotation=-90, verticalalignment='center',
             horizontalalignment='center')

    # RGB Text
    color_spc = np.linspace(0.212, 0.728, 3)
    color_y = 0.98
    fig.text(color_spc[0], color_y, "Red", verticalalignment='center', horizontalalignment='center')
    fig.text(color_spc[1], color_y, "Green", verticalalignment='center', horizontalalignment='center')
    fig.text(color_spc[2], color_y, "Blue", verticalalignment='center', horizontalalignment='center')

    # Electrode Text
    eltro_spc = np.linspace(0.092, 0.9, 8)
    eltro_x = 0.03
    fig.text(eltro_x, eltro_spc[7], "P4", verticalalignment='center', horizontalalignment='center', rotation='vertical')
    fig.text(eltro_x, eltro_spc[6], "PO4", verticalalignment='center', horizontalalignment='center',
             rotation='vertical')
    fig.text(eltro_x, eltro_spc[5], "O2", verticalalignment='center', horizontalalignment='center', rotation='vertical')
    fig.text(eltro_x, eltro_spc[4], "Oz", verticalalignment='center', horizontalalignment='center', rotation='vertical')
    fig.text(eltro_x, eltro_spc[3], "POz", verticalalignment='center', horizontalalignment='center',
             rotation='vertical')
    fig.text(eltro_x, eltro_spc[2], "O1", verticalalignment='center', horizontalalignment='center', rotation='vertical')
    fig.text(eltro_x, eltro_spc[1], "PO3", verticalalignment='center', horizontalalignment='center',
             rotation='vertical')
    fig.text(eltro_x, eltro_spc[0], "P3", verticalalignment='center', horizontalalignment='center', rotation='vertical')

    # Colorbar
    cbar_ax = fig.add_axes([0.89, 0.05, 0.013, 0.9])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.solids.set_rasterized(True)

    # Figure config
    fig.subplots_adjust(wspace=0.1, left=0.12, right=0.87, bottom=0.05, top=0.95)
    fig.set_size_inches(7, 9, forward=True)
    path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
    filename = 'HHT_subject_' + subj + '.pdf'
    # fig.savefig(path + filename, dpi=200)
    # plt.show()


def hilbert_spectrum_v2(freq_data, amp_data, freq_bins, imf_skip=None, max_imfs=anco.max_imfs):
    assert amp_data.shape == freq_data.shape, "Input data shape is wrong"

    # Length of spectrum: length of event + given time before and after to observe effect and changes
    spectrum_samples_length = amp_data.shape[-1]
    bin_length = freq_bins.shape[0]
    spectrum = np.zeros((bin_length, spectrum_samples_length))

    # Iterate over each given imf and calculate its instantaneous frequency and amplitude
    for i in range(max_imfs - 1):
        # Skip first IMF which containst mostly line noise
        if imf_skip is not None:
            if i in imf_skip:
                continue
        # Get imf series
        imf_inst_freq = freq_data[i]
        imf_inst_amp = amp_data[i]

        # Convert from amplitude to power
        imf_inst_power = imf_inst_amp ** 2

        # Bin the given inst_freq given from freq_bins,
        # and add each respective power value to its given index in the spectrum
        freq_index = np.digitize(imf_inst_freq, freq_bins) - 1

        # For each power value add to its corresponding time sample and digitized frequency
        for power_ind, power_val in enumerate(imf_inst_power):
            binned_freq = freq_index[power_ind]
            # If value exist in position [binned_freq, power_ind], check if new value is higher and overwrite
            if spectrum[binned_freq, power_ind] < power_val:
                spectrum[binned_freq, power_ind] = power_val
    return spectrum


def hilbert_spectrum_imfs(imfs, freq_bins, max_imfs=anco.max_imfs):
    samples = imfs.shape[1]

    # Iterate over each given imf and calculate its instantaneous frequency and amplitude
    imf_amp = np.zeros((max_imfs, samples))
    imf_freq = np.zeros((max_imfs, samples))
    for i, imf in enumerate(imfs):
        if i > max_imfs - 1:
            break
        imf_amp[i], imf_freq[i] = anfn.hilbert_transform(imfs[i], anco.BOARD_FREQUENCY)
        mean_freq = np.mean(imf_freq[i])
        per10_amp = np.percentile(imf_amp[i], 10)
        print(f"IMF {i}, mean freq {mean_freq}, amp 10percentile {per10_amp}")

        # Median filter to remove EEMD artifacts
        imf_amp[i] = medfilt(imf_amp[i], 3)
        imf_freq[i] = medfilt(imf_freq[i], 3)

    spectrum = hilbert_spectrum_v2(imf_freq, imf_amp, freq_bins, [00], max_imfs=max_imfs)
    return spectrum


def imf_subplot(data, events, max_imfs):
    for i, imf in enumerate(data):
        if i == max_imfs:
            break
        ax = plt.subplot(max_imfs, 1, i + 1)
        ax.plot(imf)
        inst_amp, inst_freq = anfn.hilbert_transform(imf, anco.BOARD_FREQUENCY)
        ax.set_title(f"Mean frequency: {np.mean(inst_freq):{2}.{3}} Hz", fontsize=10, y=0,
                     color=(0.2, 0.2, 0.2))
        ymin, ymax = ax.get_ylim()
        add_events_indication(events, 'red', [ymin, ymax])

    plt.show()


def add_events_indication(events, color, v_height):
    if len(events) < 1:
        return
    for event in events:
        add_indication(event, color, v_height)


def add_indication(event, color, v_height):
    y_min, y_max = v_height
    current_axis = plt.gca()
    current_axis.add_patch(
        Rectangle((event[0], y_min), event[1] - event[0], abs(y_min) + abs(y_max), alpha=0.2, facecolor=color))


def plot_from_emdfilt(self, subject, color):
    y_max = 150
    y_min = -y_max
    data = self.eeg_data[subject, color, :, :-1]
    events = self.events.event_list[subject][color]
    # errors = self.errorlist
    channels = self.eeg_data.shape[-1] - 1

    line_color = ['sienna', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'grey']
    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        plt.plot(data[:, i], linewidth=1.0, color=line_color[i])
        if i < channels - 1:
            plt.tick_params(axis='x',
                            which='both',
                            bottom='off',
                            labelbottom='off')

        plt.tight_layout()
        axis = plt.gca()
        add_events_indication(events[i], 'blue', [y_min, y_max])
        axis.set_ylim([y_min, y_max])
        # axis.set_xlim([10000, 10200])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def channel_subplot(data, events, subj, color):
    data_length = data.shape[-1]
    y_max = 200
    y_min = -y_max
    for channel_ind, channel in enumerate(data):
        imf_data = np.zeros(data_length)
        for i, imf in enumerate(channel):
            inst_amp, inst_freq = anfn.hilbert_transform(imf, anco.BOARD_FREQUENCY)
            mean_freq = np.mean(inst_freq)
            if 0.4 < mean_freq < 40:
                imf_data += imf
                # print(f"IMF {i}, mean freq {mean_freq}")
        plt.subplot(8, 1, channel_ind + 1)
        plt.plot(imf_data)
        axis = plt.gca()
        add_events_indication(events[channel_ind], 'red', [y_min, y_max])
        axis.set_ylim([y_min, y_max])
        axis.set_xlim([0, data_length])
        if channel_ind < anco.channels - 1:
            axis.tick_params(axis='x',
                             which='both',
                             bottom='off',
                             labelbottom='off')
    plt.suptitle(f"Subject {subj+1}, color{color}")
    plt.show()
