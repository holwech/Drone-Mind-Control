from lib.stream.stream import Stream


stream = Stream()
stream.connect()
print("Pulling samples...")
stream.pull_time_series(2000)

