import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


class LivePlot():
    def __init__(self):
        self.x = 0
        pass

    def start_plot(self):
        style.use('fivethirtyeight')
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(1,1,1)

    def update(self, data):
        plt.plot(self.x, data, 'ro')
        self.x += 1


def test_plot():
    lp = LivePlot()
    lp.start_plot()
    for i in range(10):
        lp.update(i*i)
        plt.pause(2)
    while True:
        plt.pause(0.05)
