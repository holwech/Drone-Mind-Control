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
LEFT = 0
RIGHT = 70
NEUTRAL = 0

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

def countdown(seconds):
    while seconds > 0:
        time.sleep(1)
        print(seconds)
        seconds -= 1


# NEUTRAL = 0
# RIGHT = 1
# LEFT = 2
def run_experiment(stream, path):
    # Set up a number of each experiment and shuffle the order of them
    exp_size = RIGHT + LEFT + NEUTRAL
    exp_types = np.zeros(exp_size, dtype=int)
    exp_types[NEUTRAL:(RIGHT * 2)] += 1
    exp_types[(RIGHT * 2):(RIGHT * 2 + 3 * LEFT)] += 2
    np.random.shuffle(exp_types)
    print(exp_types)
    i = 0

    print("Next experiment starting in...")
    countdown(3)

    for exp_type in exp_types:
        filename = str(i) + "_right_" + str(exp_type) + ".csv"
        filename = os.path.join(path, filename)
        start_message = threading.Timer(1.0, print_action_start, [exp_type])
        stop_message = threading.Timer(4.0, print_action_stop, [exp_type])

        if i % 10 == 0 & i != 0:
            print("Next experiment starting in...")
            countdown(10)
        i += 1
        print("====================" + str(i) + "/" + str(LEFT + RIGHT + NEUTRAL))

        start_message.start()
        stop_message.start()
        time_series = stream.pull_time_series(6000)
        np.savetxt(filename, time_series, delimiter=",")
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
    path = os.path.join(dir, "Data", name, "single_right")
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


