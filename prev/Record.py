import time
import atexit
import numpy as np
import record_func.arduino as ard
import record_func.timer as tim
import record_func.obci as obci
import record_func.logic as logi
import record_func.liveplot as lp
import record_func.livefiltering as lf
import configuration.variables as vari
import configuration.record_conf as conf


def main():
    # Init exit thread
    tim.start_thread(tim.kbhit_program_abort)

    # Initialize variables
    data = logi.dataarray_init()
    pulse_sample_number = 0
    data_length = conf.BOARD_FREQUENCY * conf.duration

    # Initialize comms
    obci.device_start_stream()
    ard.serial_wake()

    # Light mode
    logi.mode_initialize()

    # Start timing loop and global t0 initial time
    vari.t0 = time.time()
    tim.start_thread(tim.light_event_loop, vari.mode)

    # Live plot setup
    livedata = lf.PlotData()
    liveplot = lp.LivePlot(livedata.ringbuf_out.get)
    liveplot.update_setup(10)

    while vari.RUNNING:

        # Retrieve record_data from obci headset
        data_sample = obci.device_data_retrieve()

        # If record_data exists, add to sample number
        if np.any(data_sample):
            data[vari.sample_number, :8] = data_sample
            livedata.filter_inp(data_sample)

            # If global color_enable is set, add light record_data to array
            if vari.color_enable == 1:
                eeg_trigger = logi.data_color_index_checker()
                data[vari.sample_number, conf.eeg_channels + eeg_trigger] = 1
                pulse_sample_number += 1
                if pulse_sample_number == vari.light_duration * conf.BOARD_FREQUENCY:
                    pulse_sample_number = 0
                    vari.color_enable = 0

            # Sample retrieved, incrementing sample number counter
            vari.sample_number += 1

        # Live plot update
        liveplot.window_refresh(vari.sample_number)

        if vari.sample_number % 250 == 0 and vari.sample_number != 0:
            print(vari.sample_number / 250)

        # Main program while loop exit check
        logi.main_loop_exit_check(vari.sample_number, data_length)

    # Duration of main loop exit print
    logi.exit_time_print()

    # Saving record_data to file
    logi.datafile_save_array(data)


def safe_exit():
    ard.arduino_set_confmode()
    ard.serial_close()
    obci.serial_close()


if __name__ == '__main__':
    atexit.register(safe_exit)
    main()
