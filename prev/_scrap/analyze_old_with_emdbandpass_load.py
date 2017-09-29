import os
import time
import numpy as np
from scipy.signal import butter, lfilter, hilbert
import importlib.util
from multiprocessing import Process, Queue, current_process
from tkinter import Tk, filedialog
from analyze_func.emd import empirical_mode_decomposition as emd, is_monocomponent as monocomp
# import analyze_func.plot_funcs as plot

import configuration.record_conf as conf
import configuration.analyze_conf as anco


# ===========================================================
# =                     FUNCTIONS                           =
# ===========================================================

# ===================== EMD ======================
def hilbert_transform(data, f_samp):
    complex_val = hilbert(data)
    inst_amp = np.abs(complex_val)
    inst_phase = np.unwrap(np.angle(complex_val))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * f_samp
    return inst_freq, inst_amp


def eemd_mt(data, noise_std, max_modes, num_processes=8, ensembles_per_process=30):
    data_length = len(data)
    imfs = np.zeros((max_modes + 1, data_length))
    output = Queue(maxsize=num_processes)
    time_start = time.perf_counter()
    jobs = []
    for i in range(num_processes):
        p = Process(target=ensemble_process,
                    args=(
                        data,
                        data_length,
                        noise_std,
                        max_modes,
                        ensembles_per_process,
                        output))
        jobs.append(p)

    for job in jobs:
        job.start()

    mt_result = [output.get() for _ in jobs]
    for i, imf_ens in enumerate(mt_result):
        imfs += imf_ens

    imfs = np.multiply(imfs, 1.0 / (ensembles_per_process * num_processes))
    print(f"eemd on signal took {time.perf_counter() - time_start:{4}.{2}} seconds")
    return imfs


def ensemble_process(data, data_length, noise_std, max_modes, ensembles_per_process, output):
    name = current_process().name
    # print(f"{name} starting")
    imfs = np.zeros((max_modes + 1, data_length))
    for i in range(ensembles_per_process):
        noise = np.random.normal(0, noise_std, size=data_length)
        noise_assisted_data = np.add(data, noise)
        ensemble, _ = emd(noise_assisted_data)
        for j, imf in enumerate(ensemble):
            if j > max_modes:
                break
            try:
                imfs[j, :] += imf
            except:
                pass
    # print(f"{name} exiting")
    output.put(imfs)


def eemd_st(x, noise_std=35, max_modes=12, ensembles=200):
    data_length = len(x)
    imfs = np.zeros((max_modes + 1, data_length))

    for i in range(ensembles):
        print(f"Ensemble {i}")
        time_start = time.perf_counter()
        noise = np.random.normal(0, noise_std, size=data_length)
        noise_assisted_data = np.add(x, noise)
        ensemble, _ = emd(noise_assisted_data)
        for j, imf in enumerate(ensemble):
            imfs[j, :] += ensemble[j, :]
        print(time.perf_counter() - time_start)

    imfs = np.multiply(imfs, 1.0 / ensembles)
    return imfs


def emd_band_filter(data, freq_low, freq_high, eemd=False):
    data_extract = np.zeros(data.shape)
    if monocomp(data):
        return data_extract
    if eemd:
        c_i = eemd_mt(data)
    else:
        c_i, _ = emd(data)
    for i, imf in enumerate(c_i):
        inst_freq, _ = hilbert_transform(imf, conf.BOARD_FREQUENCY)
        imf_mean_freq = np.mean(inst_freq)
        print(f"imf: {i+1}, mean freq: {imf_mean_freq}")
        if imf_mean_freq < freq_low:
            break
        elif imf_mean_freq < freq_high:
            print(f"Adding imf {i+1}")
            data_extract += imf
    return data_extract


# ======================= MISC ========================
def file_info_get(string, splitchar, index):
    string_number = string.split(splitchar)
    return int(string_number[index])


def check_overlapping_range(x, y):
    assert len(x) == 2 and len(y) == 2, print(f"Input x and y must be array of shape (2,1)")
    x_start = x[0]
    x_end = x[1]
    y_start = y[0]
    y_end = y[1]
    assert x_start <= x_end and y_start <= y_end, print(f"Range in bracket is noncorrect [a, b] | a<=b")

    if x_start > y_end:
        return False
    elif x_end < y_start:
        return False
    elif x_start <= y_end and x_end >= y_start:
        return True
    else:
        print(f"Creators logic dun fucked up")





def cma_calc(x_np1, n, cma_n=0):  # Calculate next iteration of Cumulative Moving Average
    if cma_n is 0:
        cma_n = np.zeros(x_np1.shape)
    return (x_np1 + n * cma_n) / (n + 1)


def get_subj_color_from_file(file):
    subj, mode, color, *rest = file.split('_')
    subj = int(subj) - 1
    color = anco.color_choices.index('_' + color + '_')
    return subj, color


# ===================== TEMPLATE ======================

# ===========================================================
# =                     CLASSES                             =
# ===========================================================

# =============== Supportive Classes =============

# Class for loading all data from disk into memory
class EEGdataLoader:
    def __init__(self, path_emdfilt, path_imf):

        # Checks path to emd filtered data exists
        if not os.path.exists(path_emdfilt):
            print("Given path does not exist, exiting")
            quit()
        self.path_emdfilt = path_emdfilt
        self.filelist = []
        for (dirpath, dirnames, filenames) in os.walk(path_emdfilt):
            self.filelist.extend(filenames)
            break

        # Checks path to imf data exists
        if not os.path.exists(path_imf):
            print("Given path does not exist, exiting")
            quit()
        self.path_imf = path_imf

    # Gets emd filtered data from disk and returns it
    def emd_filtered_data_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")

        # Check if data is already extracted, loaded and saved into a compressed format for faster functioncall
        compressed_data_filename = experiment + 'data_compressed'
        if os.path.isfile(self.path_emdfilt + '/' + compressed_data_filename + '.npz'):
            return np.load(self.path_emdfilt + '/' + compressed_data_filename + '.npz')['arr_0']
        experiment_count = 0
        experiment_list = []
        for filename in self.filelist:
            if filename.find('error_list') != -1:
                continue
            elif filename.find('_' + experiment + '_') != -1:
                experiment_list.append(filename)
                experiment_count += 1

        if experiment == 'SS':
            samples = anco.SS_experimentduration * conf.BOARD_FREQUENCY
        elif experiment == 'pulse':
            samples = anco.pulse_experimentduration * conf.BOARD_FREQUENCY - anco.precut
        # Data shape: [subjects, R/G/B, num_samples, channels]
        subjects = experiment_count // conf.color_channels
        data = np.zeros((subjects, conf.color_channels, samples, conf.eeg_channels + 1))
        colors = anco.color_choices
        for file in experiment_list:
            print(file)
            for color_index, color in enumerate(colors):
                # Get subjectnumber
                subject = -1 + file_info_get(file, '_', 0)
                # Setting colums to retrieve, [eeg channels + corresponding trigger column]
                usecols = np.arange(0, conf.eeg_channels)
                usecols = np.append(usecols, [color_index + conf.eeg_channels])

                # If choise of color exists in filename, add eeg and trigger info to data
                if file.find(color) != -1:
                    tmp_data = np.loadtxt(self.path_emdfilt + '/' + file,
                                          delimiter=',',
                                          usecols=usecols)
                    data[subject, color_index, :tmp_data.shape[0], :tmp_data.shape[1]] = tmp_data
        np.savez_compressed(self.path_emdfilt + '/' + compressed_data_filename, data)
        return data

    # Imports list of errors as a module with a given variable,
    # should be changed to xml or similar for better editability
    def errorlist_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")
        for filename in self.filelist:
            if filename.find(experiment + '_error_list') != -1:
                full_path = self.path_emdfilt + '/' + filename
                module_name = experiment + '_error_list'
                spec = importlib.util.spec_from_file_location(module_name, full_path)
                imported_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(imported_module)
                return getattr(imported_module, module_name)

    # Gets imf data from all subjects and colors from experiment (pulse or SS) from a saved and compressed npz file
    def imf_data_get(self, experiment):
        assert experiment == 'SS' or experiment == 'pulse', print("Noncorrect experiment value chosen")
        compressed_data_filename = 'IMF_' + experiment + '_data_compressed'
        if os.path.isfile(self.path_imf + '/' + compressed_data_filename + '.npz'):
            return np.load(self.path_imf + '/' + compressed_data_filename + '.npz')['arr_0']
        else:
            print(f"The compressed IMF file does not exist in {self.path_emdfilt}")


class Filter:
    def __init__(self, order, type, f_cutoff, fsamp):
        nyquist = 0.5 * fsamp
        if len(f_cutoff) > 1:
            f_low, f_high = f_cutoff
            f_low_norm = f_low / nyquist
            f_high_norm = f_high / nyquist
            self.b, self.a = butter(order, [f_low_norm, f_high_norm], btype=type)
        else:
            f_cutoff_norm = f_cutoff / nyquist
            self.b, self.a = butter(order, f_cutoff_norm, btype=type)

    def filter_data(self, data):
        return lfilter(self.b, self.a, data)


class Events:
    def __init__(self, data):
        trigger_index = 8
        subjects = data.shape[0]
        colors = data.shape[1]
        channels = data.shape[3] - 1  # Channels + one event/trigger channel
        # event_list structure: event_list[subject][color][channel][events]
        self.event_list = [[[[] for _ in range(channels)] for _ in range(colors)] for _ in range(subjects)]
        prev_sample = None
        event_start = None
        event_end = None
        for subject in range(subjects):
            for color in range(colors):
                for channel in range(channels):
                    prev_sample = 0
                    for sample_index, sample in enumerate(data[subject, color, :, trigger_index]):
                        if sample > prev_sample:
                            event_start = sample_index
                            prev_sample = 1
                        elif sample < prev_sample:
                            event_end = sample_index - 1
                            self.event_list[subject][color][channel].append([event_start, event_end])
                            prev_sample = 0

    def errorlist_exclude(self, errorlist):
        # Number of channels in the errorlist data
        channels = len(errorlist[0][1])
        for i in errorlist:
            subject = i[0][0] - 1  # errorlist subects are 1 indexed
            color = i[0][1]
            for channel in range(channels):
                errors = i[1][channel]
                num_errors = len(errors)
                if num_errors < 1:
                    continue
                else:
                    # Iterate over each error in the error list for its channel and check if it
                    # overlaps with any events.
                    for error in errors:
                        # Event delete list to avoid deleting list items while iterating over it
                        delete_index_list = []
                        for event_index, event in enumerate(self.event_list[subject][color][channel]):
                            if check_overlapping_range(error, event):
                                delete_index_list.append(event_index)
                        # If the delete list contains entries, these events is deleted from events.event_list
                        # for loop is reversed due to troubles with deleting indexed values while looping over them
                        if len(delete_index_list) > 0:
                            for index in reversed(delete_index_list):
                                # print(index)
                                del self.event_list[subject][color][channel][index]


# ======================== Main Class ========================
class EEG:
    def __init__(self, experiment):
        emdfilt_folder = "C:/github/eeg_master_thesis/Python/eeg_data/emdfilt"
        imf_folder = "C:/github/eeg_master_thesis/Python/eeg_data/pulse/eemd_decomp"
        # Tk().withdraw()
        # foldername = filedialog.askdirectory()


        # LOAD EMD FILTERED DATA
        self.eeg_fileinfo = EEGdataLoader(emdfilt_folder, imf_folder)
        self.eeg_data = self.eeg_fileinfo.emd_filtered_data_get(experiment)
        self.events = Events(self.eeg_data)

        # LOAD IMF DATA
        self.imf_data = self.eeg_fileinfo.imf_data_get(experiment)

        # Load errorlist and exclude from events.eventlist
        self.errorlist = self.eeg_fileinfo.errorlist_get(experiment)
        self.events.errorlist_exclude(self.errorlist)

    def plot(self):
        pass


