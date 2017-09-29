import os
import numpy as np
from tkinter import Tk, filedialog
from analyze_func.analyze_funcs import eemd_mt
import analyze_func.analyze as anly
import analyze_func.analyze_funcs as anfn
import configuration.analyze_conf as anco
import record_func.dmc as dmc


#############################
# Filtering script          #
#############################
def main():
    # Choose folder of where the files lie to be filtered
    Tk().withdraw()
    folderpath = filedialog.askdirectory()
    save_destination = folderpath + '/' + anco.eemddecomp_save_dir
    filename_save_imf = 'IMF_' + anco.eemd_decomp_experiment + '_data_compressed'
    filename_save_event = 'event_' + anco.eemd_decomp_experiment + '_data_compressed'

    # Extract list of filenames from folder and create savefolder if it does not exist
    filelist = []
    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        filelist.extend(filenames)

        if not os.path.exists(save_destination):
            print("Destinationfolder does not exist, creating")
            os.makedirs(save_destination)
        break

    experiment_list = []
    for filename in filelist:
        if filename.find('error_list') != -1:
            continue
        elif filename.find('_' + anco.eemd_decomp_experiment + '_') != -1:
            experiment_list.append(filename)

    num_subjects = anco.subjects
    colors = anco.color_choices
    num_colors = len(colors)
    num_channels = anco.channels
    if anco.eemd_decomp_experiment == 'pulse':
        num_samples = anco.pulse_experimentduration * anco.BOARD_FREQUENCY
    elif anco.eemd_decomp_experiment == 'SS':
        num_samples = anco.SS_experimentduration * anco.BOARD_FREQUENCY
    else:
        print("MISSING FOR STEADYSTATE")
        quit()

    # Container for EMD for experiment, format: [subject, color, channel, imf, samples]
    data_decomp = np.zeros((num_subjects, num_colors, num_channels, anco.max_modes + 1, num_samples - anco.precut))

    # Container for events, format: [subject, color, samples]
    data_event = np.zeros((num_subjects, num_colors, num_samples - anco.precut))

    # For each file in folder
    for filenumber, file in enumerate(filelist):
        print(f"Filtering file {filenumber+1} of {len(filelist)}, {file}")
        # Get filename of file and its filtered counterpart
        subj, color = anfn.get_subj_color_from_file(file)
        # Load data from file
        data = np.loadtxt(folderpath + '/' + file, delimiter=',')[anco.precut:, :].T

        # For each channel in file,
        for ch in range(anco.channels):
            print(f"Filtering channel {ch+1}")
            data_filt = data[ch, :]
            data_filt = dmc.iirfilter_low(data_filt, anco.BOARD_FREQUENCY, anco.f_lowpass, anco.order_notch)
            data_filt = dmc.iirfilter_notch(data_filt, anco.BOARD_FREQUENCY, anco.f_notch, anco.notch_band_size,
                                            anco.order_notch)
            data_filt = eemd_mt(data_filt, anco.noise_std, anco.max_modes)
            data_decomp[subj, color, ch, :, :data_filt.shape[-1]] = data_filt

        # Copy event data over to its own file
        data_event[subj, color, :data.shape[-1]] = data[anco.channels + color, :]

    # Save to disk
    np.savez_compressed(save_destination + '/' + filename_save_imf, data_decomp)
    np.savez_compressed(save_destination + '/' + filename_save_event, data_event)

    print(f"\nAll files in selected folder are filtered and saved in folder:\n'{save_destination}'")


if __name__ == '__main__':
    main()
