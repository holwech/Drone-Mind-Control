import numpy as np
import time
import record_func.obci as obci
import record_func.arduino as ard
import configuration.record_conf as conf
import configuration.variables as vari
from datetime import datetime


def dataarray_init():
    experiment_samples = int(conf.duration * conf.BOARD_FREQUENCY)
    channels = conf.eeg_channels + conf.color_channels
    data = np.zeros((experiment_samples, channels), dtype='f')
    return data


def mode_initialize():
    vari.mode = int(input(' [1] Pulse \n [2] Steady-State \n [3] Random \n Choose: '))
    if vari.mode == 1 or vari.mode == 2:
        vari.color = input(' [R] Red \n [G] Green \n [B] Blue \n Choose: ')
    if not 0 < vari.mode < 4:
        print("No mode chosen, quitting...")
        quit()
    ard.config_send(vari.mode)
    lightduration_init()


def lightduration_init():
    if vari.mode == 1 or vari.mode == 3:
        vari.light_duration = conf.pulse_duration
    elif vari.mode == 2:
        vari.light_duration = conf.steadystate_duration
    else:
        print("Mode error in lightduration, qutting")
        quit()


def datafile_open_empty(filename_save):
    open(filename_save, 'w').close()
    f = open(filename_save, "a")
    return f


def mode_to_string():
    if vari.mode == 1:
        mode_str = 'pulse'
    elif vari.mode == 2:
        mode_str = 'SS'
    elif vari.mode == 3:
        mode_str = 'rand'
    else:
        mode_str = ''

    if vari.mode == 1 or vari.mode == 2:
        pulse_color_str = vari.color
    elif vari.mode == 3:
        pulse_color_str = 'rgb'
    else:
        pulse_color_str = ''

    return mode_str + '_' + pulse_color_str


def datafile_save_array(data):
    datestring = datetime.now().strftime("%Y_%m_%d_%H_%M")
    mode_str = mode_to_string()
    filename = conf.folder + conf.subjectname + '_' + mode_str + '_' + datestring + conf.filesuffix
    np.savetxt(filename, data, delimiter=',\t', fmt='%6.5f')


def datafile_retrieve_save(file):
    data = obci.device_data_retrieve()
    if np.any(data):
        file.write("{}\n".format(data[0]))
        return data


def data_color_index_checker():
    if vari.color == 'R':
        return 0
    elif vari.color == 'G':
        return 1
    elif vari.color == 'B':
        return 2
    else:
        print("color has not correct value, quitting, Value:" + str(vari.color))
        quit()


def exit_time_print():
    program_duration = "%.2f" % (time.time() - vari.t0)
    print(f"Exit time at {program_duration} seconds")


def main_loop_exit_check(sample_number, data_length):
    if sample_number >= data_length:
        print("End of specified duration reached, exiting main loop")
        vari.RUNNING = False
