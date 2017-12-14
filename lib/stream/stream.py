from pylsl import StreamInlet, resolve_stream
import time
from timeit import default_timer as timer

# Default sample rate for OpenBCI-headset
SAMPLE_RATE = 250

# Handles everything regarding the data connection between the
# USB dongle and the OpenBCI headset.
class Stream():
    def __init__(self):
        print("====================")
        print("Nothing happening? Make sure that the lab")
        print("streaming layer application is running :)")
        print("===================")
        pass

    # Connect to LabStreamingLayer.
    # Requires LabStreamingLayer to already be initialized.
    def connect(self):
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        print("Connected to streams.")

    # Return a list of sample and corresponding timestamp.
    def pull_sample(self):
        return self.inlet.pull_sample()

    # Samples for a given duration of time in milliseconds
    def pull_time_series(self, duration):
        elapse_time = timer()
        print("Printing time_series for ", duration, " milliseconds.")

        num_samples = int(round(250 * (duration / 1000)))
        time_series = []
        count = 0

        while count <= num_samples:
            count += 1
            samples, timestamp = self.pull_sample()
            time_series.append([timestamp] + samples)

            # Wait for a given period based on the sampling frequency
            start_time = timer()
            curr_time = timer()
            while (curr_time - start_time) < (1 / SAMPLE_RATE):
                curr_time = timer()

        print("Timeseries complete")
        print("Time elapsed is ", timer() - elapse_time)
        print("Number of samples are ", len(time_series))
        return time_series


    # Samples for a given duration of time in milliseconds
    # Limited by time and not samples
    def pull_time_series_tm(self, duration):
        elapse_time = timer()
        print("Printing time_series for ", duration, " milliseconds.")

        time_series = []
        start_time = timer() * 1000
        curr_time = timer() * 1000
        while (curr_time - start_time) < duration:
            samples, timestamp = self.pull_sample()
            time_series.append([timestamp] + samples)
            curr_time = timer() * 1000

        print("Timeseries complete")
        print("Time elapsed is ", timer() - elapse_time)
        print("Number of samples are ", len(time_series))
        return time_series
