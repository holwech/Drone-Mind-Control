#############################
# EEMD decomp SETTINGS      #
#############################

subjects = 10
channels = 8
eemd_decomp_experiment = 'pulse'
eemddecomp_save_dir = 'eemd_decomp'
max_modes = 12
noise_std = 25
# Remove start of data due to glitch in numbers at sample ~140 in all recordings
precut = 500

#################################
# ANALYZE SETTINGS              #
#################################
BOARD_FREQUENCY = 250
experiment = 'pulse'

# If true, will always generate all data from scratch instead of loading in previous saved
recompile_all_data = False


SS_data_samples = 4000

SS_experimentduration = 120
pulse_experimentduration = 200
color_choices = ['_R_', '_G_', '_B_']

# Filtering
f_low_anly = 0.5
f_high_anly = 50
f_notch = 50
f_lowpass = 45

order_band_anly = 2
order_notch = 3
notch_band_size = 2

# Hilbert spectrum
prefetch = 125
postfetch = 250
hht_freq_min = 0.8
hht_freq_max = 60
num_freq_bins = 200
max_imfs = 9
event_length = 250
last_event_sample = 48000
