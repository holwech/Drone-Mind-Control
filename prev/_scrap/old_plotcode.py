# def imf_amp_epoch(self):
#     prefetch = 300
#     signal_length = 250
#     postfetch = 500
#     subject_list = [9]
#     color_list = [0]
#     channel_list = [4]
#     inst_amp_epoch = np.zeros((self.imf_data.shape[-2], prefetch + signal_length + postfetch))
#     inst_freq_epoch = np.zeros((self.imf_data.shape[-2], prefetch + signal_length + postfetch))
#     avg_count = 0
#     for subject in subject_list:
#         for color in color_list:
#             for channel in channel_list:
#                 for i, imf in enumerate(self.imf_data[subject, color, channel]):
#
#                     inst_amp, inst_freq = anly.hilbert_transform(imf, anco.BOARD_FREQUENCY)
#                     if i == 0:
#                         print(f"subj {subject}, color {color}, channel {channel}, mean freq: {np.mean(inst_freq)}")
#                     for event in self.events.event_list[subject][color][channel]:
#                         try:
#                             inst_amp_epoch[i, :] += inst_amp[event[0] - prefetch:event[1] + postfetch + 1]
#                             inst_freq_epoch[i, :] += inst_freq[event[0] - prefetch:event[1] + postfetch + 1]
#                         except:
#                             pass
#                 avg_count += len(self.events.event_list[subject][color][channel])
#
#     inst_amp_epoch = inst_amp_epoch / avg_count
#     inst_amp_epoch = inst_amp_epoch ** 2
#     inst_freq_epoch = inst_freq_epoch / avg_count
#     low_freq_index = 0
#     for i, imf in enumerate(inst_freq_epoch):
#         if np.mean(imf) < 0.6:
#             low_freq_index = i
#             break
#
#     fig = plt.figure()
#     fig.set_facecolor((0.95, 0.95, 0.95))
#     for i in range(len(inst_amp_epoch)):
#         if i == low_freq_index:
#             break
#
#         ax = plt.subplot(low_freq_index, 1, i + 1)
#         ax.plot(inst_amp_epoch[i])
#
#         ax.grid()
#         ymin, ymax = ax.get_ylim()
#         ax.set_xlim([0, len(inst_amp_epoch[i])])
#         add_indication([prefetch, prefetch + signal_length], 'red', [ymin, ymax])
#         if i < low_freq_index - 1:
#             ax.tick_params(axis='x',
#                            which='both',
#                            bottom='off',
#                            labelbottom='off')
#         ax.set_title(f"Mean frequency: {np.mean(inst_freq_epoch[i]):{2}.{3}} Hz", fontsize=10, y=0,
#                      color=(0.2, 0.2, 0.2))
#         fig.tight_layout()
#     fig.subplots_adjust(wspace=0, hspace=0.18)
#     plt.show()
#
#
# def wavelet_experiment(self):
#     prefetch = 100
#     postfetch = 250
#     scales = np.arange(1, 129)
#     wavelet = 'cmor'
#     coef_avg = 0
#     power_avg = 0
#     freqs = 0
#     average_count = 0
#     channel_index = [3, 4]
#     channel_index = [2, 5]
#     color_index = [0]
#     for subject in range(10):
#         for color in color_index:
#             for channel in channel_index:
#                 for event_number, event in enumerate(self.events.event_list[subject][color][channel]):
#                     coef, freqs = pywt.cwt(
#                         self.eeg_data[subject, color, event[0] - prefetch:event[1] + postfetch, channel],
#                         scales, wavelet, sampling_period=1 / anco.BOARD_FREQUENCY)
#                     power = abs(coef) ** 2
#                     power_avg = anly.cma_calc(power, average_count, coef_avg)
#                     average_count += 1
#                     print(average_count)
#     t = np.arange(power_avg.shape[-1])
#     fig2, ax = plt.subplots()
#     mngr = plt.get_current_fig_manager()
#     mngr.window.setGeometry(1950, 50, 1600, 800)
#     T, S = np.meshgrid(t, freqs)
#     ax.contourf(T, S, power_avg, 1000)
#     ax.set_yscale('log')
#     fig2.show()
#     plt.show()

# def hht_signal_to_spectrum(x, freq_bins):
#     # Create power spectrum of size [number of freq bins, number of samples]
#     samples = x.shape[1]
#     bin_length = freq_bins.shape[0]
#     spectrum = np.zeros((bin_length, samples))
#     # Iterate over each given imf and calculate its instantaneous frequency and amplitude
#     if i > max_imfs - 1:
#         break
#     inst_freq, inst_amp = anfn.hilbert_transform(imf, anco.BOARD_FREQUENCY)
#     # Convert to power
#     inst_power = inst_amp ** 2
#
#     # Median filter to remove EEMD artifacts
#     inst_power = medfilt(inst_power, 7)
#     inst_freq = medfilt(inst_freq, 13)
#     # Bin the given inst_freq given from freq_bins,
#     # and add each respective power value to its given index in the spectrum
#     freq_index = np.digitize(inst_freq, freq_bins) - 1
#     for amp_ind, amp_val in enumerate(inst_power):
#         binned_freq = freq_index[amp_ind]
#         spectrum[binned_freq, amp_ind] = amp_val
#     return spectrum

# def hilbert_spectrum_subj_color_plot(freq_data, amp_data):
#     log_min_freq = np.log10(anco.hht_freq_min)
#     log_max_freq = np.log10(anco.hht_freq_max)
#     freq_bins = np.logspace(log_min_freq, log_max_freq, anco.num_freq_bins, base=10)
#     channels = freq_data.shape[0]
#     spectrum = np.zeros((channels, anco.num_freq_bins, freq_data.shape[-1]))
#     for ch in range(channels):
#         spectrum[ch] = hilbert_spectrum_v2(freq_data[ch], amp_data[ch], freq_bins, [0, 6, 7, 8, 9])
#         # spectrum[ch] = gaussian_filter(spectrum[ch], sigma=0.3, mode='constant')
#
#     # Time vector
#     start_time = -anco.prefetch / anco.BOARD_FREQUENCY
#     stop_time = (anco.event_length + anco.postfetch) / anco.BOARD_FREQUENCY
#     t_vec = np.linspace(start_time, stop_time, freq_data.shape[-1])
#
#     # Plot scale and array
#     scale_max = np.max(spectrum)
#     scale_max = 150
#     scale = np.logspace(np.log10(10), np.log10(scale_max), 12)
#     spectrum = hilbert_power_scale_limit(spectrum, scale_max)
#
#     fig, axis = plt.subplots(nrows=channels, ncols=1, sharex='all', sharey='all')
#     for i, ax in enumerate(axis):
#         im = ax.contourf(t_vec, freq_bins, spectrum[i], scale, cmap='magma_r', zorder=-1)
#         # ax.set_rasterized(True)
#         # plt.clim(0, scale_maxmax)
#         ax.set_yscale('log')
#         ax.axvline(x=0, color='grey', linestyle='--')
#         ax.axvline(x=1, color='grey', linestyle='--')
#         if i < channels - 1:
#             ax.tick_params(axis='x',
#                            which='both',
#                            bottom='off',
#                            labelbottom='off')
#         else:
#             ax.set_xlabel("Time [s]")
#         ax.set_rasterization_zorder(0)
#
#     fig.subplots_adjust(wspace=0.1, hspace=0.1, right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     fig.set_size_inches(4, 9, forward=True)
#     # fig.savefig('_ign_pics/raster400.pdf', dpi=600)
#
#     plt.show()


# def hilbert_spectrum_ch_plot(freq_data, amp_data):
#     log_min_freq = np.log10(anco.hht_freq_min)
#     log_max_freq = np.log10(anco.hht_freq_max)
#     freq_bins = np.logspace(log_min_freq, log_max_freq, anco.num_freq_bins, base=10)
#
#     spectrum = hilbert_spectrum_v2(freq_data, amp_data, freq_bins)
#
#     # Time vector
#     start_time = -anco.prefetch / anco.BOARD_FREQUENCY
#     stop_time = (anco.event_length + anco.postfetch) / anco.BOARD_FREQUENCY
#     t_vec = np.linspace(start_time, stop_time, spectrum.shape[-1])
#
#     # Power scale
#     scale_max = np.max(spectrum) * 0.5
#     print(f"Spectrum max {np.max(spectrum)}, spectrum 95% {np.percentile(spectrum, 95)}")
#     print(f"scale_top value {scale_max}")
#     scale = np.linspace(0, scale_max, 8)
#     plt.contourf(t_vec, freq_bins, spectrum, scale)
#     plt.yscale('log')
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.axvline(x=0, color='red', linestyle='--')
#     plt.axvline(x=1, color='red', linestyle='--')
#     plt.show()

# def hilbert_spectrum(data, events):
#     log_min_freq = np.log10(anco.hht_freq_min)
#     log_max_freq = np.log10(anco.hht_freq_max)
#     freq_bins = np.logspace(log_min_freq, log_max_freq, anco.num_freq_bins, base=10)
#     # Length of spectrum: length of event + given time before and after to observe effect and changes
#     event_length = events[0][1] - events[0][0]
#     spectrum_samples_length = event_length + anco.prefetch + anco.postfetch
#     bin_length = freq_bins.shape[0]
#     spectrum_total = np.zeros((bin_length, spectrum_samples_length))
#     spectrum_count = np.zeros((bin_length, spectrum_samples_length))
#     for event in events:
#         data_event = data[:, event[0] - anco.prefetch:event[1] + anco.postfetch]
#         spectrum = hilbert_transf_imfs(data_event, freq_bins)
#         spectrum_count += anfn.spectrum_boolean_checker(spectrum)
#         spectrum_total += spectrum
#
#     spectrum_total = anfn.spectrum_avg_normalizer(spectrum_total, spectrum_count)
#
#     # Time vector
#     start_time = -anco.prefetch / anco.BOARD_FREQUENCY
#     stop_time = (event_length + anco.postfetch) / anco.BOARD_FREQUENCY
#     t_vec = np.linspace(start_time, stop_time, spectrum_samples_length)
#
#     # Power scale
#     scale = anfn.hilbert_power_scale_get(spectrum_total, 8)
#     plt.contourf(t_vec, freq_bins, spectrum_total, scale)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Frequency [Hz]")
#     plt.axvline(x=0, color='red', linestyle='--')
#     plt.axvline(x=1, color='red', linestyle='--')
#     plt.show()