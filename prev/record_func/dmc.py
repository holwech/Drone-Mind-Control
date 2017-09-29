import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


# Custom IIR filter
def iirfilter_iter(a, b, x, y_prev):
    a_len = len(a) - 1
    b_len = len(b)
    x_len = len(x)
    yprev_len = len(y_prev)

    if a_len < 1 or b_len < 1:
        print("a or b length error, quitting")
        quit()

    if x_len == 0:
        x_comp = np.zeros(b_len)
    elif 0 < x_len < b_len:
        x_comp = np.zeros(b_len)
        x_comp[-x_len:] = x
    elif x_len == b_len:
        x_comp = x
    elif x_len > b_len:
        x_comp = x[-b_len:]
    else:
        print(f"x size error, len_x = {x_len}, quitting")
        quit()

    if yprev_len == 0:
        y_comp = np.zeros(a_len)
    elif 0 < yprev_len < a_len:
        y_comp = np.zeros(a_len)
        y_comp[-yprev_len:] = y_prev
    elif yprev_len == a_len:
        y_comp = y_prev
    elif yprev_len > a_len:
        y_comp = y_prev[-a_len:]
    else:
        print("y size error, quitting")
        quit()


    x_comp = np.multiply(x_comp, np.flipud(b))
    y_comp = np.multiply(y_comp, np.flipud(a[1:]))
    y = np.sum(x_comp) - np.sum(y_comp)
    # y = y * 1 / a[0]
    return y


# FIR filter ----------------------------------------------------
def firfilter_band(n, f_sample, f1, f2, data):
    nyq = 0.5 * f_sample
    f1, f2 = 1, 100
    h = scipy.signal.firwin(n, [f1, f2], window='blackmanharris', pass_zero=False, nyq=nyq)
    data_filt = scipy.signal.lfilter(h, 1.0, data, 0)
    return data_filt


# IIR filter ----------------------------------------------------

# Offline d_rec filters ---------------

# Butterworth bandpass
def butterfilter_band(n, f_sample, f1, f2, data):
    # type: (object, object, object, object, object) -> object
    nyq = 0.5 * f_sample
    f1_norm = f1 / nyq
    f2_norm = f2 / nyq
    b, a = scipy.signal.butter(n, [f1_norm, f2_norm], 'band', analog=False)
    data_filt = scipy.signal.lfilter(b, a, data, 0)
    return data_filt


# 50 Hz notch filter
def iirfilter_notch(data, f_sample, freq, band, n):
    ripple = 10
    nyq = 0.5 * f_sample
    low = freq - band / 2.0
    high = freq + band / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = scipy.signal.iirfilter(n, [low_norm, high_norm], rp=ripple, btype='bandstop', analog=False, ftype='butter')
    data_filt = scipy.signal.filtfilt(b, a, data)
    return data_filt

def iirfilter_low(data, f_sample, w0, n):
    nyq = 0.5 * f_sample
    w0_norm = w0 / nyq
    b, a = scipy.signal.iirfilter(n, w0_norm, btype='lowpass', analog=False, ftype='butter')
    data_filt = scipy.signal.filtfilt(b, a, data)
    return data_filt


# Online d_rec filters ---------------

# Butterworth bandpass
def butterfilter_band_poly(n, f_sample, f1, f2):
    nyq = 0.5 * f_sample
    f1_norm = f1 / nyq
    f2_norm = f2 / nyq
    b, a = scipy.signal.butter(n, [f1_norm, f2_norm], 'band', analog=False)
    return [b, a]


# 50 Hz notch filter
def iirfilter_notch_poly(n, f_sample, freq, band):
    ripple = 10
    nyq = 0.5 * f_sample
    low = freq - band / 2.0
    high = freq + band / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = scipy.signal.iirfilter(n, [low_norm, high_norm], rp=ripple, btype='bandstop', analog=False, ftype='butter')
    return [b, a]


# Filter d_rec with prev filter state
def iirfiltering_prevstate(filter_poly, data, zi):
    b = filter_poly[0]
    a = filter_poly[1]
    print(data)
    y, zf = scipy.signal.lfilter(b, a, data, axis=-1, zi=zi)
    return y, zf


# MISC ------------------------------------------------------------
def PlotData(data, time):
    plt.plot(time, data)
    plt.ylabel('Data')
    plt.xlabel('Sample')


def data_segment(start, stop, data):
    length = stop - start
    data_ret = np.ones((int(length), 1))
    for i in range(start, stop):
        data_ret[i - start] = data[i]
    time_epoch = np.linspace(start / 250, stop / 250, stop - start)
    return data_ret, time_epoch


def CMA_calc(x_np1, n, CMAn):  # Calculate next iteration of Cumulative Moving Average
    return ((x_np1 + n * CMAn) / (n + 1))


def array_compress(data, fetch_interval):
    row_count = 0
    column_count = 0
    first_append = True

    data_cut = np.array([])

    for row in range(0, len(data[:, 0])):
        if row_count == fetch_interval:
            row_count = 0
            row_tmp = np.array([])
            for column in range(0, len(data[:, 0])):
                if column_count == fetch_interval:
                    column_count = 0
                    row_tmp = np.append(row_tmp, data[row, column])
                else:
                    column_count += 1
            column_count = 0
            if first_append:
                data_cut = row_tmp
                first_append = False
            else:
                data_cut = np.vstack((data_cut, row_tmp))
        else:
            row_count += 1

    return data_cut


def vector_compress(data, fetch_interval):
    element_count = 0
    data_cut = np.array([])
    for element in range(0, len(data)):
        if element_count == fetch_interval:
            element_count = 0
            data_cut = np.append(data_cut, data[element])
        else:
            element_count += 1

    return data_cut


def array_compress_axis(data, fetch_interval, offAxis=False):
    column_count = 0
    first_append = True

    if offAxis == True:
        data = data.T

    data_cut = np.array([])

    for row in range(0, len(data[:, 0])):
        row_tmp = np.array([])
        for column in range(0, len(data[0, :])):
            if column_count == fetch_interval:
                column_count = 0
                row_tmp = np.append(row_tmp, data[row, column])
            else:
                column_count += 1
        column_count = 0
        if first_append:
            data_cut = row_tmp
            first_append = False
        else:
            data_cut = np.vstack((data_cut, row_tmp))

        if offAxis == True:
            data_cut = data_cut.T
    return data_cut
