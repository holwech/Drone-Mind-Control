from random import uniform
from random import choice
from threading import Thread
import time
import msvcrt
import record_func.arduino as ard
import configuration.variables as vari
import configuration.record_conf as conf


# Start daemon thread
def start_thread(function, args=None):
    if args == None:
        thread = Thread(target=function)
    else:
        thread = Thread(target=function, args=(args,))
    thread.daemon = True
    thread.start()


# Timing loop which delays the light event for a given interval
def light_event_loop(mode):
    while vari.RUNNING:
        if mode == 1 or mode == 3:
            random_val = uniform(0, conf.maximum_interval)
            sleep_length = random_val + conf.minimum_interval
            time.sleep(sleep_length)
            sample_time = time.time() - vari.t0
            light_event(mode, sample_time)

        elif mode == 2:
            time.sleep(conf.steadystate_pre_duration)
            sample_time = time.time() - vari.t0
            light_event(mode, sample_time)
            break

        else:
            print("Noncorrect mode chosen, quitting")
            quit()


# Event for each interval, prints the color lighted in the console and sends the command to the arduino
def light_event(mode, sample_time):
    if mode == 3:
        color = choice(conf.color_choices)
        vari.color = color
    else:
        color = vari.color
    ard.printLED(color, sample_time)
    ard.arduino_character_send(color)
    vari.color_enable = 1


# Checks if the keyboard key 'q' has been hit for program termination
def kbhit_program_abort():
    while vari.RUNNING:
        time.sleep(0.3)
        if msvcrt.kbhit():
            button = msvcrt.getch()
            if button == b'q':
                print("q pressed, exiting...")
                vari.RUNNING = False

# def liveplot_update_smoother(self, inp):
#     if vari.sample_number <= conf.num_samples_lag_threshold:
#         time.sleep(0.1)
#     else:
#         self.filter_inp(inp)
