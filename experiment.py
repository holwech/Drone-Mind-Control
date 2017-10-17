import pandas as pd
from lib.stream.stream import Stream
import os.path
import sys
import bcrypt
from enum import Enum
import numpy as np
import time
import threading

dir = os.path.dirname(__file__)

# Number of each experiment type
single_exp_size = 70

# Prints whether to close left or right fist
# Defined as own function for threading purposes
def print_action_start(exp_type):
    if exp_type == 1:
        print("\n      RIGHT ---->\n")
    elif exp_type == 2:
        print("\n<---- LEFT\n")


# Prints when to stop closing fist
# Defined as own function for threading purposes
def print_action_stop(exp_type):
    if exp_type != 0:
        print("STOP")

# NEUTRAL = 0
# RIGHT = 1
# LEFT = 2
def run_experiment(stream, path):
    # Set up a number of each experiment and shuffle the order of them
    exp_size = single_exp_size * 3
    exp_types = np.zeros(exp_size, dtype=int)
    exp_types[single_exp_size:(single_exp_size * 2)] += 1
    exp_types[(2 * single_exp_size):(3 * single_exp_size)] += 2
    np.random.shuffle(exp_types)
    print(exp_types)
    i = 0
    for exp_type in exp_types:
        filename = str(i) + "_lf_" + str(exp_type) + ".csv"
        filename = os.path.join(path, filename)
        i += 1
        start_message = threading.Timer(3.0, print_action_start, [exp_type])
        stop_message = threading.Timer(7.0, print_action_stop, [exp_type])

        print("====================" + str(i) + "/" + str(single_exp_size * 3))
        print("Next experiment starting in...")
        countdown = 3
        while countdown > 0:
            time.sleep(1)
            print(countdown)
            countdown -= 1
        start_message.start()
        stop_message.start()
        time_series = stream.pull_time_series(10000)
        np.savetxt(filename, time_series, delimiter=",")
        time.sleep(2)





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


