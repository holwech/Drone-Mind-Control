import analyze_func.analyze as anly
import configuration.analyze_conf as anco
import matplotlib.pyplot as plt


def main():
    EEG = anly.EEG(anco.experiment)
    electrode_list = [2, 5, 2, -1, 5, -1, 3, -1, -1, 3]
    imf = [1,2]
    EEG.color_comparison_single(electrode_list, imf)
    # EEG.SS_fft_response_consistency()

    # for subject in range(10):
    #     EEG.fourier_plot(subject, electrode_list[subject])
    # EEG.fourier_plot(0, 2)
    # EEG.plot_channels(9,0)
    # electrode_list = [2, 5, 2, 3, 5, 3, 3, 3, 3, 3]
    # EEG.color_correlation(electrode_list, imf, max_modes)
    # EEG.color_comparison(electrode_list, imf)
    # EEG.color_energy_plot(subject-1, channel, imfs)
    # EEG.erp_power_ratio(imfs)
    # max_scale = [100, 200, 100, 81, 400, 81, 150, 81, 81, 100]
    # for subject in range(10):
    #     EEG.hht_subject_full(subject, max_scale[subject])


if __name__ == '__main__':
    main()
