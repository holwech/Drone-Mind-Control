import pandas as pd
from lib.stream.stream import Stream
import os.path
import os
import sys
import bcrypt
from enum import Enum
import numpy as np
import time
import threading


from pylsl import StreamInlet, resolve_stream
import time
from timeit import default_timer as timer

dir = os.path.dirname(__file__)

SAMPLE_RATE = 250

# Number of each experiment type
NEUTRAL = 0
RIGHT = 70
LEFT = 0

exp_type_to_str = ["neutral", "right", "left"]

# Prints whether to close left or right fist
# Defined as own function for threading purposes
def print_action_start(exp_type):
    if exp_type == 1:
        print("""
                         |\\
                         | \\
     ====================|  \\
     |       RIGHT       |   |
     ====================|  /
                         | /
                         |/ \n""")
    elif exp_type == 2:
        print("""
   /|
  / |
 /  |====================
|   |         LEFT      |
 \  |====================
  \ |
   \| \n""")


# Prints when to stop closing fist
# Defined as own function for threading purposes
def print_action_stop(exp_type):
    if exp_type != 0:
        print("""
 ____________ _____
/ __|_ __/ _ \|  _ \\
\__ \ | | (_) | |_) |
|___/ | |\___/| .__/
              | |
              |_| \n""")

def countdown(seconds):
    while seconds > 0:
        print(seconds, end='\r')
        time.sleep(1)
        seconds -= 1
    os.system('clear')


# Samples for a given duration of time in milliseconds
def pull_time_series(stream, duration, exp_type):
    elapse_time = timer()
    #print("Printing time_series for ", duration, " milliseconds.")

    num_samples = int(round(250 * (duration / 1000)))
    time_series = []
    count = 0
    first = True

    while count <= num_samples:
        if ((timer() - elapse_time) >= 3.0) & first:
            print_action_start(exp_type)
            first = False
        count += 1
        samples, timestamp = stream.pull_sample()
        time_series.append([timestamp] + samples)

        # Wait for a given period based on the sampling frequency
        start_time = timer()
        curr_time = timer()
        while (curr_time - start_time) < (1 / SAMPLE_RATE):
            curr_time = timer()

    #print_action_stop(exp_type)
    #print("Timeseries complete")
    #print("Time elapsed is ", timer() - elapse_time)
    #print("Number of samples are ", len(time_series))
    return time_series


# NEUTRAL = 0
# RIGHT = 1
# LEFT = 2
def run_experiment(stream, path):
    # Set up a number of each experiment and shuffle the order of them
    exp_size = RIGHT + LEFT + NEUTRAL
    exp_types = np.zeros(exp_size, dtype=int)
    exp_types[NEUTRAL:(NEUTRAL + RIGHT)] += 1
    exp_types[(NEUTRAL + RIGHT):(NEUTRAL + RIGHT + LEFT)] += 2
    np.random.shuffle(exp_types)
    print(exp_types)
    i = 0

    print("Experiment starting in...")
    countdown(5)

    for exp_type in exp_types:
        if (i % 10 == 0) & (i != 0):
            print("Press any key to continue...")
            input()
            countdown(5)
        filename = str(i) + exp_type_to_str[exp_type] + ".csv"
        filename = os.path.join(path, exp_type_to_str, filename)

        #start_message = threading.Timer(3.0, print_action_start, [exp_type])
        #stop_message = threading.Timer(6.0, print_action_stop, [exp_type])
        #start_message.start()
        #stop_message.start()
        i += 1
        print("START ====================" + str(i) + "/" + str(LEFT + RIGHT + NEUTRAL))
        time_series = pull_time_series(stream, 6000, exp_type)
        np.savetxt(filename, time_series, delimiter=",")
        print("STOP  ====================")
        print("Starting next measurement...")
        countdown(2)





if __name__ == '__main__':
    stream = Stream()
    stream.connect()
    print("*******************************************")
    print("Experiment is starting.")
    print("*******************************************")

    print("Please write in your first name.")
    #print("Your name will not be stored and only used to create an unique anonymous ID")
    name = input('Name: ')
    #hashed_name = bcrypt.hashpw(str.encode(name), bcrypt.gensalt())

    #path = "/data/" + hashed_name.decode("utf-8")  + "/"
    path = os.path.join(dir, "Data", name)
    print(path)

    # If not yes, jump to beginning of while loop
    if os.path.exists(path):
        continue_name = input("This name already exists, continue? (y/n)")
        if continue_name != "y":
            sys.exit()
    else:
        try:
            os.makedirs(path)
        except ValueError:
            print(ValueError)
            sys.exit()

    run_experiment(stream, path)
    print("Experiment done")
    print("Exiting...")


