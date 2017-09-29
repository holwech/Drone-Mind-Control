import time
import numpy as np
import scipy.signal
from record_func.ringbuffer import RingBuffer2D
from record_func.dmc import iirfilter_iter
import configuration.variables as vari
import configuration.record_conf as conf

class PlotData:
    def __init__(self):
        self.ringbuf_inp = RingBuffer2D(conf.eeg_channels, conf.filtorder_rt * 2 + 1)
        self.ringbuf_out = RingBuffer2D(conf.eeg_channels, conf.window_duration * conf.BOARD_FREQUENCY)
        nyquist = 0.5 * conf.BOARD_FREQUENCY
        f_low_norm = conf.f_low_rt / nyquist
        f_high_norm = conf.f_high_rt / nyquist
        self.b, self.a = scipy.signal.butter(conf.filtorder_rt, [f_low_norm, f_high_norm], btype='band')
        self.channels = conf.eeg_channels
        self.sample_number = 0

    def filter_inp(self, inp):
        self.ringbuf_inp.append(inp)
        tmp_out = np.empty(self.channels)
        for i in range(self.channels):
            tmp_out[i] = iirfilter_iter(self.a, self.b, self.ringbuf_inp.get(i), self.ringbuf_out.get(i))
        self.ringbuf_out.append(tmp_out)
        self.sample_number += 1