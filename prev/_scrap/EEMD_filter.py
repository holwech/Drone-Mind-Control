import os
import numpy as np
from tkinter import Tk, filedialog
from analyze_func.analyze import emd_band_filter
import configuration.analyze_conf as anco

#############################
# Filtering script          #
#############################
def main():
    # Choose folder of where the files lie to be filtered
    Tk().withdraw()
    folderpath = filedialog.askdirectory()
    save_destination = folderpath + '/' + anco.eemdfilt_save_dir

    # Extract list of filenames from folder and create savefolder if it does not exist
    filelist = []
    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        filelist.extend(filenames)

        if not os.path.exists(save_destination):
            print("Destinationfolder does not exist, creating")
            os.makedirs(save_destination)
        break
    # For each file in folder, save a filtered version in a subdirectory
    for filenumber, file in enumerate(filelist):
        # Get filename of file and its filtered counterpart
        subj, mode, color, *rest = file.split('_')
        filename_save = subj + '_' + mode + '_' + color + '_' + 'emdfilt' + '.csv'
        full_path = save_destination + '/' + filename_save

        # If filtered file already exist, continue to next file
        if os.path.isfile(full_path):
            print(f"File '{filename_save}' exists, skipping...")
            continue
        print(f"Filtering file number {filenumber+1} of {len(filelist)}, {filename_save}")
        data = np.loadtxt(folderpath + '/' + file, delimiter=',')[anco.precut:, :]
        data_eegfilt = np.zeros(data.shape)

        # For each channel in file, filter with emd
        for ch in range(anco.channels):
            print(f"Filtering channel {ch+1}")
            data_eegfilt[:, ch] = emd_band_filter(data[:, ch], anco.freq_low, anco.freq_high, eemd=True)

        # Copy aux data over to the filtered file
        data_eegfilt[:, anco.channels:] = data[:, anco.channels:]
        # Save to disk
        np.savetxt(full_path, data_eegfilt, fmt='%4.4f', delimiter=', ')

    print(f"\nAll files in selected folder are filtered and saved in folder:\n'{save_destination}'")


if __name__ == '__main__':
    main()
