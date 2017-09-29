import time
from multiprocessing import Process, Queue, current_process
import numpy as np
from scipy.signal import hilbert
import configuration.analyze_conf as anco
from analyze_func.emd import empirical_mode_decomposition as emd, is_monocomponent as monocomp


# ===========================================================
# =                     FUNCTIONS                           =
# ===========================================================

# ===================== EMD ======================
def hilbert_transform(data, f_samp):
    inst_freq = np.zeros(data.shape)
    complex_val = hilbert(data)
    inst_amp = np.abs(complex_val)
    inst_phase = np.unwrap(np.angle(complex_val))
    inst_freq[:-1] = np.diff(inst_phase) / (2.0 * np.pi) * f_samp
    # Linearly interpolate last two values to complete same shape
    inst_freq[-1] = 1.5 * inst_freq[-2] - inst_freq[-3]
    return inst_amp, inst_freq


def imf_to_filt_signal(imfs, min_freq, max_freq, fsamp):
    signal = np.zeros(imfs.shape[-1])
    for imf in imfs:
        inst_amp, inst_freq = hilbert_transform(imf, fsamp)
        mean_freq = np.mean(inst_freq)
        if mean_freq > max_freq:
            continue
        elif min_freq <= mean_freq <= max_freq:
            signal += imf
        elif mean_freq < min_freq:
            break
        else:
            print("imf_to_filt_signal did something it should not")
    return signal


def eemd_mt(data, noise_std, max_modes, num_processes=6, ensembles_per_process=200):
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
        c_i = eemd_mt(data, anco.noise_std, anco.max_modes)
    else:
        c_i, _ = emd(data)
    for i, imf in enumerate(c_i):
        inst_freq, _ = hilbert_transform(imf, anco.BOARD_FREQUENCY)
        imf_mean_freq = np.mean(inst_freq)
        print(f"imf: {i+1}, mean freq: {imf_mean_freq}")
        if imf_mean_freq < freq_low:
            break
        elif imf_mean_freq < freq_high:
            print(f"Adding imf {i+1}")
            data_extract += imf
    return data_extract


# ======================= HHT support ========================
def spectrum_boolean_checker(spectrum):
    spectrum_boolean = np.zeros(spectrum.shape)
    for i, row in enumerate(spectrum):
        for j, value in enumerate(row):
            if value != 0:
                spectrum_boolean[i, j] = 1
    return spectrum_boolean


def spectrum_avg_normalizer(spec_total, spec_val_count):
    for i, row in enumerate(spec_val_count):
        for j, value in enumerate(row):
            if 1 < value:
                spec_total[i, j] /= spec_val_count[i, j]
    return spec_total


def hilbert_power_scale_get(spectrum, num_scale):
    percentile_low = np.percentile(spectrum, 20)
    percentile_high = np.percentile(spectrum, 95)
    mean = np.percentile(spectrum, 50)
    scale = np.linspace(percentile_low, percentile_high, num_scale)
    print(f"Scale from {percentile_low} to {percentile_high}")
    return scale


def hilbert_power_scale_limit_full(spectrum, max_val):
    for i_col, color in enumerate(spectrum):
        for i_ch, channel in enumerate(color):
            for i_row, row in enumerate(channel):
                for i, val in enumerate(row):
                    if max_val < val:
                        spectrum[i_col, i_ch, i_row, i] = max_val
    return spectrum


def hilbert_power_scale_limit(spectrum, max_val):
    for i_ch, channel in enumerate(spectrum):
        for i_row, row in enumerate(channel):
            for i, val in enumerate(row):
                if max_val < val:
                    spectrum[i_ch, i_row, i] = max_val
    return spectrum


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


def mean_label(axis, bars):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2., 1.05 * height,
                  '%d' % int(height),
                  ha='center', va='bottom')
