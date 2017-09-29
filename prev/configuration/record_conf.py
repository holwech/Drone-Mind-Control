#################################
# EEG RECORD SETTINGS
#################################
subjectname = "TEST"
folder = "E:\\GitHub\\eeg_master_thesis\\Python\\eeg_data\\"
folder = "C:\\github\\eeg_master_thesis\\Python\\eeg_data\\"
filesuffix = ".csv"


#################################
# ARDUINO SETTINGS
#################################
duration = 60  # Duration of experiment in seconds
pulse_duration = 1  # Length of  lightpulses in seconds
minimum_interval = 6  # Minimum random interval between pulses in seconds
maximum_interval = 7  # Maximum random interval between pulses in seconds

steadystate_pre_duration = 2
steadystate_duration = 60  # Length of steady-state light in seconds

color_choices = ['R', 'G', 'B']

COM_arduino = 'COM7'
BAUD_arduino = 9600

#################################
# OBCI SETTINGS
#################################
gain = 24
vref = 4.5
scale_factor = vref * 1000000 / (float((pow(2, 23) - 1)) * gain)

BOARD_FREQUENCY = 250

COM_obci = 'COM3'
BAUD_obci = 115200

eeg_channels = 8
color_channels = 3

#################################
# RT VIEW SETTINGS
#################################
window_duration = 4
window_data_lagbuffer_duration = 0.5
window_update_freq = 30
f_low_rt = 1
f_high_rt = 10
filtorder_rt = 2

win_update_rate = int(BOARD_FREQUENCY / window_update_freq)
num_samples_lag_threshold = BOARD_FREQUENCY * window_data_lagbuffer_duration
