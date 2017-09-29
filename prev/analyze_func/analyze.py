import os
import pickle
import numpy as np
import importlib.util
from tkinter import Tk, filedialog
import analyze_func.plot_funcs as plot
import analyze_func.analyze_funcs as anfn
import configuration.analyze_conf as anco
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fftpack import rfft
from scipy.ndimage.filters import gaussian_filter1d as gauss1d
from math import sqrt


# ===========================================================
# =                     CLASSES                             =
# ===========================================================

# =============== Supportive Classes =============

# Class for loading all data from disk into memory
class EEGdataLoader:
    def __init__(self, path):
        # Checks path to imf data exists
        if not os.path.exists(path):
            print("Given path does not exist, exiting")
            quit()
        self.path = path

    # Gets imf data from all subjects and colors from experiment (pulse or SS) from a saved and compressed npz file
    # format: [subject, color, channel, imf, samples]
    def imf_data_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")
        compressed_data_filename = 'IMF_' + experiment + '_data_compressed'
        if os.path.isfile(self.path + '/' + compressed_data_filename + '.npz'):
            return np.load(self.path + '/' + compressed_data_filename + '.npz')['arr_0']
        else:
            print(f"The compressed IMF file does not exist in {self.path}")

    # Gets the raw event data, sample = 0 normal, 1 on event
    # format: [subject, color, samples]
    def event_data_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")
        compressed_data_filename = 'event_' + experiment + '_data_compressed'
        if os.path.isfile(self.path + '/' + compressed_data_filename + '.npz'):
            return np.load(self.path + '/' + compressed_data_filename + '.npz')['arr_0']
        else:
            print(f"The compressed event file does not exist in {self.path}")

    # Imports data of errors as a module with a given variable,
    # for the future, should be changed to xml or other format for better editability
    def error_data_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")
        module_name = experiment + '_error_data'
        suffix = '.py'
        full_path = self.path + '/' + module_name + suffix
        if os.path.isfile(full_path):
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            imported_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported_module)
            return getattr(imported_module, module_name)
        else:
            print(f"No module with name '{module_name}' exists in folder '{self.path}'")

    def SS_event_data_get(self):
        experiment = 'SS'
        module_name = experiment + '_event_data'
        suffix = '.py'
        full_path = self.path + '/' + module_name + suffix
        if os.path.isfile(full_path):
            spec = importlib.util.spec_from_file_location(module_name, full_path)
            imported_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported_module)
            return getattr(imported_module, module_name)
        else:
            print(f"No module with name '{module_name}' exists in folder '{self.path}'")

    def event_list_corrected_get(self):
        filename = anco.experiment + '_event_list_corrected.p'
        full_path = self.path + '/' + filename
        if os.path.isfile(full_path):
            event_list = pickle.load(open(full_path, "rb"))
            return event_list
        else:
            print(f"Event_list does not exists at path {full_path}, generating")
            return -1

    def event_list_corrected_save(self, event_list_corrected):
        filename = anco.experiment + '_event_list_corrected.p'
        full_path = self.path + '/' + filename
        if not os.path.isfile(full_path):
            pickle.dump(event_list_corrected, open(full_path, "wb"))
        else:
            print(f"Event_list already exists at path {full_path}")

    def amp_freq_check(self):
        amp_filename = anco.experiment + '_amp_data.npz'
        freq_filename = anco.experiment + '_freq_data.npz'
        amp_full_path = self.path + '/' + amp_filename
        freq_full_path = self.path + '/' + freq_filename
        if os.path.isfile(amp_full_path) and os.path.isfile(freq_full_path):
            return True
        else:
            print(f"Files for amp and freq data does not exists at path {self.path + '/'}, generating")
            return False

    def amp_freq_get(self):
        amp_filename = anco.experiment + '_amp_data.npz'
        freq_filename = anco.experiment + '_freq_data.npz'
        amp_full_path = self.path + '/' + amp_filename
        freq_full_path = self.path + '/' + freq_filename
        amp_data = np.load(amp_full_path)['arr_0']
        freq_data = np.load(freq_full_path)['arr_0']
        return amp_data, freq_data

    def amp_freq_save(self, amp_data, freq_data):
        amp_filename = anco.experiment + '_amp_data'
        freq_filename = anco.experiment + '_freq_data'
        amp_full_path = self.path + '/' + amp_filename
        freq_full_path = self.path + '/' + freq_filename
        np.savez_compressed(amp_full_path, amp_data)
        np.savez_compressed(freq_full_path, freq_data)


class Events:
    def __init__(self, path):
        self.fileinfo = EEGdataLoader(path)
        if anco.experiment == 'SS':
            self.event_list = self.ss_event_list_create()

        if anco.experiment == 'pulse':
            self.event_list = self.fileinfo.event_list_corrected_get()
            if self.event_list == -1 or anco.recompile_all_data:
                # If event_list with errors removed is not available, update event_list with the event_data
                self.event_error_list_create()
                # Loads error_data and updates error_list
                self.error_data2list()
                # Exclude errors in error_list from event_list
                self.errorlist_exclude()
                # Save list to disk for speedier startup
                self.fileinfo.event_list_corrected_save(self.event_list)

    def ss_event_list_create(self):
        event_data = self.fileinfo.SS_event_data_get()
        num_subjects = anco.subjects
        num_colors = len(anco.color_choices)
        event_list = [[[] for _ in range(num_colors)] for _ in range(num_subjects)]
        for entry in event_data:
            subject = entry[0][0] - 1  # errorlist subects are 1 indexed
            color = entry[0][1]
            active_data = entry[1][0]
            inactive_data = entry[1][1]
            event_list[subject][color].append(active_data)
            event_list[subject][color].append(inactive_data)
        return np.asarray(event_list)

    def event_error_list_create(self):
        event_data = self.fileinfo.event_data_get(anco.experiment)
        num_subjects = anco.subjects
        num_colors = len(anco.color_choices)
        num_channels = anco.channels
        last_event_index = anco.last_event_sample
        # event_list and error_list structure: event_list[subject][color][channel][events]
        self.error_list = [[[[] for _ in range(num_channels)] for _ in range(num_colors)] for _ in range(num_subjects)]
        self.event_list = [[[[] for _ in range(num_channels)] for _ in range(num_colors)] for _ in range(num_subjects)]
        prev_sample = None
        event_start = None
        event_end = None
        for subject in range(num_subjects):
            for color in range(num_colors):
                event_raw = event_data[subject, color, :]
                for channel in range(num_channels):
                    prev_sample = 0
                    for sample_index, sample in enumerate(event_raw):
                        if last_event_index < sample_index:
                            break
                        if sample > prev_sample:
                            event_start = sample_index
                            prev_sample = 1
                        elif sample < prev_sample:
                            event_end = sample_index - 1
                            self.event_list[subject][color][channel].append([event_start, event_end])
                            prev_sample = 0

    def error_data2list(self):
        error_data = self.fileinfo.error_data_get(anco.experiment)
        num_channels = anco.channels
        for entry in error_data:
            subject = entry[0][0] - 1  # errorlist subects are 1 indexed
            color = entry[0][1]
            for channel in range(num_channels):
                # Continue of given subject, color, channel contains no errors
                if len(entry[1][channel]) < 1:
                    continue
                for event in entry[1][channel]:
                    self.error_list[subject][color][channel].append(event)

    def errorlist_exclude(self):
        num_subjects = anco.subjects
        num_colors = len(anco.color_choices)
        num_channels = anco.channels
        for subject in range(num_subjects):
            for color in range(num_colors):
                for channel in range(num_channels):
                    # Continue of given subject, color, channel contains no errors
                    if len(self.error_list[subject][color][channel]) < 1:
                        continue
                    # Check every error for the channel if it has matching range with any event
                    for error in self.error_list[subject][color][channel]:
                        delete_index_list = []
                        # Iterate in reverse to not cause problems with deleting at the same time.
                        for index, event in enumerate(self.event_list[subject][color][channel]):
                            if anfn.check_overlapping_range(error, event):
                                delete_index_list.append(index)

                        if 0 < len(delete_index_list):
                            for index in reversed(delete_index_list):
                                del self.event_list[subject][color][channel][index]


# ======================== Main Class ========================
class EEG:
    def __init__(self, experiment):
        imf_folder = "C:/github/eeg_master_thesis/Python/eeg_data/pulse/eemd_decomp"
        # Tk().withdraw()
        # imf_folder = filedialog.askdirectory()

        # Load path info
        self.eeg_fileinfo = EEGdataLoader(imf_folder)

        if anco.experiment == 'SS':
            self.imf_data = self.eeg_fileinfo.imf_data_get(experiment)
            self.events = Events(imf_folder)

        if anco.experiment == 'pulse':
            if self.eeg_fileinfo.amp_freq_check() and not anco.recompile_all_data:
                self.event_amp, self.event_freq = self.eeg_fileinfo.amp_freq_get()
            else:
                # Load eeg data in decomposed form [subject, color, channel, imfs, samples]
                self.imf_data = self.eeg_fileinfo.imf_data_get(experiment)

                # Generate event lists, [subject][color][channel][[event1],[event2],...,[event-1]]
                self.events = Events(imf_folder)

                #
                self.event_amp, self.event_freq = self.median_epoch_amp_freq()
                self.eeg_fileinfo.amp_freq_save(self.event_amp, self.event_freq)

    def color_power_plot(self, subj, ch, imfs):
        amp_data = self.event_amp[subj, :, ch]
        freq_data = self.event_freq[subj, :, ch]
        plot.power_plot(freq_data, amp_data, imfs)

    def hht_chhannel(self, subj, color, channel):
        amp_data = self.event_amp[subj, color, channel]
        freq_data = self.event_freq[subj, color, channel]
        plot.hilbert_spectrum_v2(freq_data, amp_data)

    def hht_subject_full(self, subj, max_scale):
        amp_data = self.event_amp[subj]
        freq_data = self.event_freq[subj]
        plot.hilbert_spectrum_subj_full_plot(freq_data, amp_data, str(subj + 1), max_scale)

    def plot_imfs(self, subj, color, channel):
        data = self.imf_data[subj, color, channel]
        events = self.events.event_list[subj][color][channel]
        plot.imf_subplot(data, events, 10)

    def plot_channels(self, subj, color):
        data = self.imf_data[subj, color]
        events = self.events.event_list[subj][color]
        plot.channel_subplot(data, events, subj, color)

    def fourier_plot(self, subj, ch):
        colors = self.imf_data.shape[1]
        active_data_index = self.events.event_list[subj, :, 1]
        inactive_data_index = self.events.event_list[subj, :, 0]
        active_data = np.zeros((colors, anco.SS_data_samples))
        inactive_data = np.zeros((colors, anco.SS_data_samples))
        for col in range(colors):
            imfs = self.imf_data[subj, col, ch, :, active_data_index[col, 0]:active_data_index[col, 1]]
            active_data[col] = anfn.imf_to_filt_signal(imfs, 0.5, 45, anco.BOARD_FREQUENCY)
            imfs = self.imf_data[subj, col, ch, :, inactive_data_index[col, 0]:inactive_data_index[col, 1]]
            inactive_data[col] = anfn.imf_to_filt_signal(imfs, 0.5, 45, anco.BOARD_FREQUENCY)

        fs = anco.BOARD_FREQUENCY
        samp = anco.SS_data_samples
        pad_mult = 30
        xf = np.linspace(0, fs / 2, samp * pad_mult)
        max_freq_plot = 45
        ypad = np.zeros(samp * pad_mult)
        yzero = np.zeros(samp * pad_mult)
        color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
        color_list = [[n / 256 for n in color] for color in color_list]
        channel_name_list = ['P4', 'PO4', 'O2', 'Oz', 'POz', 'O1', 'PO3', 'P3']
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex='all', sharey='all')
        for col in range(colors):
            ypad[:samp] = active_data[col]
            yf_ac = (2 * abs(rfft(ypad)) / samp)
            ypad[:samp] = inactive_data[col]
            yf_in = (2 * abs(rfft(ypad)) / samp)
            # tot_power_ac = np.sum(yf_ac) ** 2
            # tot_power_in = np.sum(yf_in) ** 2
            # print(f"Subj {subj},\tch {ch},\tcol {col},\tp_ratio {tot_power_ac/tot_power_in:{3}.{2}},")
            axes[col].fill_between(xf, yf_ac, yzero, color=color_list[col], linewidth=0.65, alpha=1, zorder=-3)
            axes[col].fill_between(xf, yf_in, yzero, color='black', linewidth=0.65, alpha=0.6, zorder=-3)
            axes[col].set_rasterization_zorder(-2)
            # axes[col].plot(xf, yf_ac, color=color_list[col], alpha=1)
            # axes[col].plot(xf, yf_in, color='black', alpha=1)
        for ax in axes:
            ax.set_xlim(xmin=0, xmax=max_freq_plot)
            ax.set_ylim(ymin=0)
        fig.text(0.02, 0.5, r"Amplitude [$\mu V$]", rotation='vertical', verticalalignment='center',
                 horizontalalignment='center')
        fig.text(0.5, 0.028, r"Frequency [$Hz$]", verticalalignment='center', horizontalalignment='center')
        fig.subplots_adjust(wspace=0.1, left=0.06, right=0.98, bottom=0.135, top=0.92)
        fig.set_size_inches(7, 4.2, forward=True)
        # fig.set_size_inches(16, 8, forward=True)
        path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
        filename = 'FFT_plot_' + str(subj + 1) + '_' + channel_name_list[ch] + '.pdf'
        fig.savefig(path + filename, dpi=400)
        print(f"Subject {subj+1}, channel {channel_name_list[ch]}\t{ch+1}")

    def erp_power_ratio(self, imf):
        active_time = [anco.prefetch + 40, anco.prefetch + anco.event_length + 40]
        inactive_time = [anco.prefetch + anco.event_length + 40, anco.prefetch + anco.event_length + anco.postfetch]
        subjects = self.event_amp.shape[0]
        colors = self.event_amp.shape[1]
        channels = self.event_amp.shape[2]
        energy_avg_active_list = np.zeros((subjects, colors, channels))
        energy_avg_inactive_list = np.zeros((subjects, colors, channels))
        energy_std_active_list = np.zeros((subjects, colors, channels))
        energy_std_inactive_list = np.zeros((subjects, colors, channels))

        for subj in range(subjects):
            for color in range(colors):
                for ch in range(channels):
                    inactive = self.event_amp[subj, color, ch, imf, inactive_time[0]:inactive_time[1]]
                    inactive_avg_energy = np.average(inactive) ** 2
                    inactive_std_energy = np.std(inactive)
                    active = self.event_amp[subj, color, ch, imf, active_time[0]:active_time[1]]
                    active_avg_enery = np.average(active) ** 2
                    active_std_energy = np.std(active)
                    if active_avg_enery != 0 and inactive_avg_energy != 0:
                        energy_avg_active_list[subj, color, ch] = active_avg_enery
                        energy_avg_inactive_list[subj, color, ch] = inactive_avg_energy
                        energy_std_active_list[subj, color, ch] = active_std_energy
                        energy_std_inactive_list[subj, color, ch] = inactive_std_energy

        ymin = 0.6
        width = 0.35
        ind = np.arange(channels)
        for subj in range(subjects):
            fig, axes = plt.subplots(nrows=colors, ncols=1, sharex='all', sharey='all')

            ymax1 = np.max(energy_avg_active_list[subj])
            ymax2 = np.max(energy_avg_inactive_list[subj])
            if ymax1 > ymax2:
                ymax = ymax1 * 1.2
            else:
                ymax = ymax2 * 1.2

            for color in range(colors):
                axes[color].bar(ind, energy_avg_active_list[subj, color], width, label="Active",
                                yerr=energy_std_active_list[subj, color] * 2, color=(238 / 255, 87 / 255, 96 / 255),
                                ecolor=(34 / 255, 17 / 255, 80 / 255))
                axes[color].bar(ind + width, energy_avg_inactive_list[subj, color], width, label="Inactive",
                                yerr=energy_std_inactive_list[subj, color] * 2, color=(254 / 255, 189 / 255, 130 / 255),
                                ecolor=(34 / 255, 17 / 255, 80 / 255))
                axes[color].set_ylim(ymin=ymin, ymax=ymax)
                axes[color].grid(axis='y')

            axes[-1].set_xlabel("Channel", size=9)
            axes[-1].set_xticks(ind + width / 2)
            axes[-1].set_xticklabels(['P4', 'PO4', 'O2', 'Oz', 'POz', 'O1', 'PO3', 'P3'])
            for tick in axes[-1].xaxis.get_major_ticks():
                tick.label.set_fontsize(9)
            fig.text(0.04, 0.95, f"Subject {subj+1} - IMF {str(imf)[1:-1]}", size=9)
            fig.text(0.04, 0.5, r"Avg power [$\mu V^2$]", rotation='vertical', verticalalignment='center',
                     horizontalalignment='center', size=9)
            color_spc = np.linspace(0.36, 0.945, 3)
            color_x = 0.55
            fig.text(color_x, color_spc[2], "Red", verticalalignment='center', horizontalalignment='center', size=9)
            fig.text(color_x, color_spc[1], "Green", verticalalignment='center', horizontalalignment='center', size=9)
            fig.text(color_x, color_spc[0], "Blue", verticalalignment='center', horizontalalignment='center', size=9)

            legend = axes[0].legend(loc="upper right", bbox_to_anchor=(1.025, 1.33), framealpha=1, edgecolor='black',
                                    fancybox=False, fontsize=9)
            legend.get_frame().set_linewidth(0.8)
            fig.subplots_adjust(wspace=0.12, left=0.15, right=0.99, bottom=0.1, top=0.93)
            fig.set_size_inches(3, 4, forward=True)
            path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
            filename = 'P_ratio_subject_' + str(subj + 1) + '_' + str(imf) + '.pdf'
            fig.savefig(path + filename, dpi=400)
            # plt.show()

    def color_correlation(self, ch_list, imf, max_modes):
        noise_std = 1
        subject_index = [ind for ind, imf in enumerate(ch_list) if imf != -1]
        ch_list = [i for i in ch_list if i != -1]
        data = self.event_amp[subject_index, :, ch_list, imf]
        imf_data = np.zeros((data.shape[0], data.shape[1], max_modes, data.shape[-1]))
        subjects = data.shape[0]
        colors = data.shape[1]
        samples = data.shape[-1]
        datasets = subjects * colors
        num_correlations = subjects ** 2 * colors ** 2 * max_modes
        corr_data = np.zeros((subjects, subjects, colors, colors, max_modes))
        corr_avg = np.zeros((max_modes, colors, colors))
        corr_std = np.zeros((max_modes, colors, colors))
        # print(corr_data.shape)
        for subj_ind, subj in enumerate(data):
            for color_ind, color in enumerate(subj):
                imfs = anfn.eemd_mt(color, noise_std, max_modes, 1, 1)
                imf_data[subj_ind, color_ind] = imfs[:max_modes, :]

        for subj1 in range(subjects):
            for col1 in range(colors):
                for imf in range(max_modes):
                    for subj2 in range(subjects):
                        for col2 in range(colors):
                            # If x and y is the same dataset, set its correlation data as nan
                            if subj1 == subj2 and col1 == col2:
                                corr_data[subj1, subj2, col1, col2, imf] = np.nan
                            else:
                                x = imf_data[subj1, col1, imf]
                                y = imf_data[subj2, col2, imf]
                                correlation = abs(np.corrcoef(x, y)[1, 0])
                                corr_data[subj1, subj2, col1, col2, imf] = correlation
        for col1 in range(colors):
            for col2 in range(colors):
                for imf in range(max_modes):
                    corr_avg[imf, col1, col2] = np.nanmean(corr_data[:, :, col1, col2, imf])
                    corr_std[imf, col1, col2] = np.nanstd(corr_data[:, :, col1, col2, imf])
        width = 0.35
        ind = np.arange(6)
        index_num = 0
        for imf in range(max_modes):
            fig, axes = plt.subplots(nrows=1, ncols=1)
            for col1 in range(0, colors):
                for col2 in range(col1, colors):
                    axes.bar(ind[index_num], corr_avg[imf, col1, col2], width, label="Active",
                             yerr=corr_std[imf, col1, col2] * 2, color=(238 / 255, 87 / 255, 96 / 255),
                             ecolor=(34 / 255, 17 / 255, 80 / 255))
                    index_num += 1
            axes.set_xticks(ind)
            axes.set_xticklabels(['Red\nRed', 'Red\nGreen', 'Red\nBlue', 'Green\nGreen', 'Green\nBlue', 'Blue\nBlue'])
            fig.suptitle(f"IMF {imf}")
            plt.show()
            index_num = 0

    def color_comparison_single(self, electrode_list, imf):
        subjects = self.event_amp.shape[0]
        colors = self.event_amp.shape[1]
        samples = anco.prefetch + anco.event_length + anco.postfetch
        color_total = np.empty((subjects, colors, samples))
        color_total[:] = np.nan
        for color in range(colors):
            for subj in range(subjects):
                if electrode_list[subj] != -1:
                    if len(imf) > 1:
                        color_total_tmp = np.zeros(samples)
                        for sample in range(samples):
                            color_total_tmp[sample] = np.sum(
                                self.event_amp[subj, color, electrode_list[subj], imf, sample] ** 2)
                        color_max = np.max(color_total_tmp)
                        color_total[subj, color] = color_total_tmp / color_max
                    else:
                        color_data = self.event_amp[subj, color, electrode_list[subj], imf] ** 2
                        color_max = np.max(color_data)
                        color_total[subj, color] = color_data / color_max


        # Time vector
        start_time = -anco.prefetch / anco.BOARD_FREQUENCY
        stop_time = (anco.event_length + anco.postfetch) / anco.BOARD_FREQUENCY
        # samples = 100 * 2 + anco.event_length
        t_vec = np.linspace(start_time, stop_time, samples)

        color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
        color_list = [[n / 256 for n in color] for color in color_list]
        for subject, val in enumerate(electrode_list):
            if val == -1:
                continue
            fig, ax = plt.subplots(1, 1)
            # for color in range(colors):
            #     ax.fill_between(t_vec, color_mean[color, 63:439] - color_std[color, 63:439],
            #                     color_mean[color, 63:439] + color_std[color, 63:439], color=color_list[color], alpha=0.25)
            for color in range(colors):
                ax.plot(t_vec, color_total[subject, color], color=color_list[color], linewidth=0.8, alpha=1)
            ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
            ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
            ax.set_xlim([start_time, stop_time])
            ax.set_ylim(ymin=0)
            ax.set_xlabel(r"Time [s]")
            ax.set_ylabel(r"Normalized power")
            fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.99)
            fig.set_size_inches(7, 3, forward=True)
            print(f"subject {subject+1}")
            # plt.show()
            path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
            filename = 'Power_plot_' + str(subject+1) + '.pdf'
            fig.savefig(path + filename, dpi=600)

    def color_comparison_all(self, electrode_list, imf):
        subjects = self.event_amp.shape[0]
        colors = self.event_amp.shape[1]
        samples = anco.prefetch + anco.event_length + anco.postfetch
        color_total = np.empty((colors, subjects, samples))
        color_total[:] = np.nan
        color_mean = np.empty((colors, samples))
        color_mean[:] = np.nan
        color_std = np.empty((colors, samples))
        color_std[:] = np.nan
        for color in range(colors):
            for subj in range(subjects):
                if electrode_list[subj] != -1:
                    if len(imf) > 1:
                        color_total_tmp = np.zeros(samples)
                        for sample in range(samples):
                            color_total_tmp[sample] = np.sum(
                                self.event_amp[subj, color, electrode_list[subj], imf, sample] ** 2)
                        color_max = np.max(color_total_tmp)
                        color_total[color, subj] = color_total_tmp / color_max
                    else:
                        color_data = self.event_amp[subj, color, electrode_list[subj], imf] ** 2
                        color_max = np.max(color_data)
                        color_total[color, subj] = color_data / color_max

        for color in range(colors):
            for sample in range(color_total.shape[-1]):
                color_mean[color, sample] = np.nanmean(color_total[color, :, sample])
                color_std[color, sample] = np.nanstd(color_total[color, :, sample]) / 2

        max_modes = 6
        # imfs = anfn.eemd_mt(color_mean[0], 10, max_modes)

        # Time vector
        start_time = -63 / anco.BOARD_FREQUENCY
        stop_time = (anco.event_length + 63) / anco.BOARD_FREQUENCY
        samples = 63 * 2 + anco.event_length
        t_vec = np.linspace(start_time, stop_time, samples)

        color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
        color_list = [[n / 256 for n in color] for color in color_list]

        fig, ax = plt.subplots(1, 1)
        for color in range(colors):
            ax.fill_between(t_vec, color_mean[color, 63:439] - color_std[color, 63:439],
                            color_mean[color, 63:439] + color_std[color, 63:439], color=color_list[color], alpha=0.25)
        for color in range(colors):
            ax.plot(t_vec, color_mean[color, 63:439], color=color_list[color], linewidth=0.8, alpha=1)
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlim([-0.25, 1.25])
        ax.set_xlabel(r"Time [$s$]")
        ax.set_ylabel(r"Normalized average power")
        fig.suptitle("Average normalized power over each subject's best responding channel\nIMFs = [1,2]")
        fig.subplots_adjust(left=0.08, right=0.99)
        fig.set_size_inches(7, 3, forward=True)
        # plt.show()
        path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
        filename = 'Power_plot_' + str(imf) + '.pdf'
        fig.savefig(path + filename, dpi=600)

    def median_epoch_amp_freq(self):
        epoch_length = anco.prefetch + anco.event_length + anco.postfetch
        epoch_amp = np.zeros(np.append(self.imf_data.shape[0:4], epoch_length))
        epoch_freq = np.zeros(np.append(self.imf_data.shape[0:4], epoch_length))
        for subj_ind, subj in enumerate(self.imf_data):
            print(f"Meaning subject {subj_ind+1}")
            for color_ind, color in enumerate(subj):
                for ch_ind, ch in enumerate(color):
                    event_count = len(self.events.event_list[subj_ind][color_ind][ch_ind])
                    for i, imf in enumerate(ch):
                        event_amp_tmp = np.zeros((event_count, epoch_length))
                        event_freq_tmp = np.zeros((event_count, epoch_length))
                        for event_ind, event in enumerate(self.events.event_list[subj_ind][color_ind][ch_ind]):
                            data = imf[event[0] - anco.prefetch:event[1] + anco.postfetch + 1]
                            if len(data) != epoch_length:
                                print(
                                    f"Subj {subj_ind+1}, col {color_ind}, ch {ch_ind}, imf {i}, event {event_ind} out of data range, skipping")
                                event_amp_tmp[event_ind] = np.nan
                                event_freq_tmp[event_ind] = np.nan
                            else:
                                inst_amp, inst_freq = anfn.hilbert_transform(data, anco.BOARD_FREQUENCY)
                                inst_amp = medfilt(inst_amp, 3)
                                inst_freq = medfilt(inst_freq, 3)
                                event_amp_tmp[event_ind] += inst_amp
                                event_freq_tmp[event_ind] += inst_freq
                        if 0 < event_count:
                            for sample in range(epoch_length):
                                epoch_amp[subj_ind, color_ind, ch_ind, i, sample] = np.nanmedian(
                                    event_amp_tmp[:, sample])
                                epoch_freq[subj_ind, color_ind, ch_ind, i, sample] = np.nanmedian(
                                    event_freq_tmp[:, sample])
        return epoch_amp, epoch_freq

    def pulse_color_imf_mean_freq(self):
        subjects = 10
        colors = 3
        imf_list = [1, 2, 3]
        num_imfs = len(imf_list)
        offset = 40
        active_area = [anco.prefetch+offset, anco.prefetch + anco.event_length+offset]
        inactive_area = [anco.prefetch + anco.event_length+offset, anco.prefetch + anco.prefetch + anco.event_length+offset]
        electrode_list = [2, 5, 2, 2, 3, 3, 3, 3, 4, 3]
        active_freq_mean = np.zeros((subjects, colors, num_imfs))
        active_freq_std = np.zeros((subjects, colors, num_imfs))
        inactive_freq_mean = np.zeros((subjects, colors, num_imfs))
        inactive_freq_std = np.zeros((subjects, colors, num_imfs))
        for subject in range(subjects):
            for color in range(colors):
                electrode = electrode_list[subject]
                for i, imf in enumerate(imf_list):
                    active_freq_mean[subject, color, i] = np.mean(self.event_freq[subject, color, electrode, imf, active_area[0]:active_area[1]])
                    active_freq_std[subject, color, i] = np.std(self.event_freq[subject, color, electrode, imf, active_area[0]:active_area[1]])
                    inactive_freq_mean[subject, color, i] = np.mean(self.event_freq[subject, color, electrode, imf, inactive_area[0]:inactive_area[1]])
                    inactive_freq_std[subject, color, i] = np.std(self.event_freq[subject, color, electrode, imf, inactive_area[0]:inactive_area[1]])

        active_freq_mean_tot = np.zeros((colors, num_imfs))
        active_freq_std_tot = np.zeros((colors, num_imfs))
        inactive_freq_mean_tot = np.zeros((colors, num_imfs))
        inactive_freq_std_tot = np.zeros((colors, num_imfs))

        for imf in range(num_imfs):
            for color in range(colors):
                active_freq_mean_tot[color, imf] = np.mean(active_freq_mean[:, color, imf])
                print(f"imf {imf}, color {color}, avg {np.mean(active_freq_mean[:, color, imf])}")
                active_freq_std_tot[color, imf] = (np.mean(active_freq_std[:, color, imf]))
                inactive_freq_mean_tot[color, imf] = np.mean(inactive_freq_mean[:, color, imf])
                inactive_freq_std_tot[color, imf] = (np.mean(inactive_freq_std[:, color, imf]))


        color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
        color_list = [[n / 256 for n in color] for color in color_list]

        width = 0.7 / 3
        ind = np.arange(num_imfs)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for imf in range(num_imfs):
            for col in range(colors):
                axes.bar(ind[imf] + width * col + 0.003, active_freq_mean_tot[col, imf], width, color=color_list[col],
                         yerr=active_freq_std_tot[col, imf] * 2)
        axes.set_ylim(ymin=7)
        # axes.set_x
        # label("IMF ", size=9)
        axes.set_ylabel("Frequency [Hz]")
        axes.set_xticks(ind + width)
        axes.set_xticklabels(['IMF-1', 'IMF-2', 'IMF-3'])
        axes.yaxis.grid()
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.05, top=0.99)
        fig.set_size_inches(2.9, 6, forward=True)
        path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
        filename = 'IMF_RGB_mean_freq' + '.pdf'
        fig.savefig(path + filename, dpi=200)
        plt.show()


    def SS_fft_response_consistency(self):
        subjects = anco.subjects
        colors = len(anco.color_choices)
        imfs = 3

        red_imf1 = [27.6, 27, 26.9, 27.7, 25, 24.9]
        red_imf2 = [18.4, 17.5, 18.2, 18.1, 18.0, 18.4, 16.7, 16.6]
        red_imf3 = [9.1, 8.7, 9.1, 8.9, 8.8, 9.4, 8.3, 8.3]

        green_imf1 = [30.1, 29.7, 29.9, 29.6, 30.9, 29.7, 30.2]
        green_imf2 = [19.8, 19.4, 19.9, 19.6, 19.3, 19.2, 19.1, 18.6]
        green_imf3 = [9.7, 9.7, 9.9, 9.7, 9.7, 9.6, 9.6, 9.4]

        blue_imf1 = [27.2, 27.2, 27.2, 27.3, 27.2, 27.2, 27.3]
        blue_imf2 = [18.2, 18.1, 19.1, 18.1, 18.1, 18.1, 18.1, 18.2, 18.1, 18.1]
        blue_imf3 = [9.1, 9.0, 9.5, 9.0, 9.1, 9.1, 9.1, 9.1, 9.1, 9.4]

        ss_freq_mean = np.empty((colors, imfs))
        ss_freq_std = np.empty((colors, imfs))

        ss_freq_mean[0, 0] = np.mean(red_imf1)
        ss_freq_mean[0, 1] = np.mean(red_imf2)
        ss_freq_mean[0, 2] = np.mean(red_imf3)

        ss_freq_mean[1, 0] = np.mean(green_imf1)
        ss_freq_mean[1, 1] = np.mean(green_imf2)
        ss_freq_mean[1, 2] = np.mean(green_imf3)

        ss_freq_mean[2, 0] = np.mean(blue_imf1)
        ss_freq_mean[2, 1] = np.mean(blue_imf2)
        ss_freq_mean[2, 2] = np.mean(blue_imf3)

        ss_freq_std[0, 0] = np.std(red_imf1)
        ss_freq_std[0, 1] = np.std(red_imf2)
        ss_freq_std[0, 2] = np.std(red_imf3)

        ss_freq_std[1, 0] = np.std(green_imf1)
        ss_freq_std[1, 1] = np.std(green_imf2)
        ss_freq_std[1, 2] = np.std(green_imf3)

        ss_freq_std[2, 0] = np.std(blue_imf1)
        ss_freq_std[2, 1] = np.std(blue_imf2)
        ss_freq_std[2, 2] = np.std(blue_imf3)

        color_list = [[243, 89, 102], [46, 204, 113], [52, 152, 219]]
        color_list = [[n / 256 for n in color] for color in color_list]

        width = 0.7 / 3
        ind = np.arange(imfs)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for imf in range(imfs):
            for col in range(colors):
                axes.bar(ind[imf] + width * col + 0.003, ss_freq_mean[col, imf], width, color=color_list[col],
                         yerr=ss_freq_std[col, imf] * 2)
        axes.set_ylim(ymin=7)
        axes.set_xlabel("Frequency reponses")
        axes.set_ylabel("Frequency [Hz]")
        axes.set_xticks(ind + width)
        axes.set_xticklabels(['r1', 'r2', 'r3'])
        axes.yaxis.grid()
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.08, top=0.99)
        fig.set_size_inches(2.9, 6, forward=True)
        path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
        filename = 'Fq_response_distribution' + '.pdf'
        fig.savefig(path + filename, dpi=200)
        plt.show()

    def ss_imf_stability_plot(self):
        subjects = 10
        colors = 3
        channel = 3
        max_imfs = 7
        imf_mean_total_freq = np.zeros((subjects, colors, max_imfs))
        imf_freq_mean = np.zeros(max_imfs)
        imf_freq_std = np.zeros(max_imfs)

        for subject in range(subjects):
            for color in range(colors):
                # print(f"Subject {subject+1}, color {color}")
                events = self.events.event_list[subject, color, 0]
                imfs = self.imf_data[subject, color, channel, :, events[0]:events[1]]
                for i, imf in enumerate(imfs):
                    if i > max_imfs - 1:
                        break
                    amp, freq = anfn.hilbert_transform(imf, anco.BOARD_FREQUENCY)
                    imf_mean_total_freq[subject, color, i] = np.mean(freq)
                    # print(f"IMF {i}, mean freq {np.mean(freq)}")

        for imf in range(max_imfs):
            imf_freq_mean[imf] = np.mean(imf_mean_total_freq[:, :, imf])
            imf_freq_std[imf] = np.std(imf_mean_total_freq[:, :, imf])

        width = 0.7
        ind = np.arange(max_imfs)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for imf in range(max_imfs):
            print(f"IMF {imf}, mean freq {imf_freq_mean[imf]:{3}.{3}}, 2sigma {imf_freq_std[imf] * 2:{3}.{3}}")
            axes.bar(ind[imf], imf_freq_mean[imf], width, yerr=imf_freq_std[imf] * 2)
        axes.set_yscale('log')
        fig.text(0.55, 0.02, "IMF", horizontalalignment='center', verticalalignment='center', fontsize='10')
        fig.text(0.05, 0.45, "Frequency [Hz]", rotation='vertical', horizontalalignment='center',
                 verticalalignment='center', fontsize='10')
        # axes.set_ylabel("Frequency [Hz]")
        axes.set_xticks(ind)
        axes.yaxis.grid(b=True, which='major', linestyle='-', linewidth=0.8)
        axes.yaxis.grid(b=True, which='minor', linestyle='--', linewidth=0.6)
        fig.subplots_adjust(left=0.2, right=0.98, bottom=0.10, top=0.99)
        fig.set_size_inches(2, 4, forward=True)
        path = r'C:/Users/trondhem/Google Drive/Skole/Kyb/TrondLars/Master/Plots/'
        filename = 'IMF_freq_consistency' + '.pdf'
        fig.savefig(path + filename, dpi=200)
        plt.show()
