import pyqtgraph as pg
from PyQt5 import QtWidgets
import sys
import configuration.record_conf as conf
import time


class LivePlot:
    pen_color = [(205,133,63), (220,20,60), (255,140,0), (255,215,0), (50,205,50), (30,144,255),
                 (255,0,255), (169,169,169)]

    def __init__(self, data_function):
        self.channels = conf.eeg_channels
        self.data_function = data_function
        self.plot = []
        self.curve = []
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsWindow()
        for i in range(self.channels):
            self.plot.append(self.win.addPlot())
            self.plot[i].setRange(yRange=[-150, 150])
            self.win.nextRow()
            self.curve.append(self.plot[i].plot(pen=LivePlot.pen_color[i]))
        self.timer = pg.QtCore.QTimer()

    def update(self):
        for i in range(self.channels):
            self.curve[i].setData(self.data_function(i))

    def update_setup(self, update_timer):
        self.timer.timeout.connect(self.update)
        self.timer.start(update_timer)

    def process_plot(self):
        self.app.processEvents()

    def window_refresh(self, samp_num):
        if samp_num % conf.win_update_rate == 0:
            self.process_plot()



