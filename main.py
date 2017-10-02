
from PyQt5.QtWidgets import QApplication
import pandas as pd
from lib.stream.stream import Stream

if __name__ == '__main__':
    stream = Stream()
    stream.connect()
    time_series = stream.pull_time_series(1000)
    df = pd.DataFrame(time_series)
#    app = QApplication(sys.argv)
#    ex = GUI()
#    sys.exit(app.exec_())
