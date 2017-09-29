import serial
import serial.tools.list_ports
import binascii
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from numpy import *

BOARD_FREQUENCY = 250
BOARD_SAMPLE = 1.0/250.0

SERIAL_ARDUINO = 'COM12'
SERIAL_OBCI = 'COM3'

filename_save = "example.csv"

t = 0
antallLest = 0
sampletime = 60


plot_ElectrodeOffset = 17000 # For plotting purpose
plot_xMin = 600
plot_xMax = sampletime*BOARD_FREQUENCY    # Set the time to record
plot_yMin = -150000
plot_yMax = 0


ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, time, mean, aux = ([] for i in range(11))


gain = 24
Vref = 4.5
scale_factor = Vref*1000000/(float((pow(2,23)-1))*gain)

data = np.array([])
total_data = np.array([[0,0,0,0,0,0,0,0,0,0]])

t = 'a'
r = 'a'

def portInfo():
    ports = list(serial.tools.list_ports.comports())

    for p in ports:
        try:
            print("Port info: "),
            print (p)
        except:
            print(sys.exc_info())

def RGB(n):
    with serial.Serial(SERIAL_ARDUINO, 9600, timeout=1, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
        if n == 1:
            ser.write('R')
        if n == 2:
            ser.write('G')
        if n == 3:
            ser.write('B')

def getEEG():
    with serial.Serial(SERIAL_OBCI, 115200, timeout=1, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
        ser.write('v')  # Reset board
        ser.write('b')  # Start streaming
        print("Streaming d_rec initiated")
        tidVedBlink = -1001
        for i in range(0, plot_xMax):

            # Check if AUX contains trigger value at position i
            #if aux[i] > 0 && aux[i] < 3:
                #RGB(aux[i])

            if (i % 250 == 0):
                print (i / 250)


            s = ser.read(1)
            if binascii.hexlify(s) == "c0":
                s = ser.read(1)
                if binascii.hexlify(s) == "a0":
                    antallLest += 1
                    sample = ser.read(31)
                    # d_rec.append(int(binascii.hexlify(ser.read(31), 16))
                    sampleID = sample[0]

                    t += 1;
                    time = t
                    # Adding channel d_rec into separate lists
                    ch1 = int(binascii.hexlify(sample[1:4]), 16) * scale_factor
                    ch2 = int(binascii.hexlify(sample[4:7]), 16) * scale_factor
                    ch3 = int(binascii.hexlify(sample[7:10]), 16) * scale_factor
                    ch4 = int(binascii.hexlify(sample[10:13]), 16) * scale_factor
                    ch5 = int(binascii.hexlify(sample[13:16]), 16) * scale_factor
                    ch6 = int(binascii.hexlify(sample[16:19]), 16) * scale_factor
                    ch7 = int(binascii.hexlify(sample[19:22]), 16) * scale_factor
                    ch8 = int(binascii.hexlify(sample[22:25]), 16) * scale_factor
                    aux = int(binascii.hexlify(sample[25]), 16)

                    data = np.array([time, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, aux])
                    total_data = np.append(total_data, [data], 0)

        np.savetxt(filename_save, total_data, delimiter=", ", fmt=('%s'), header='time, signal')

def main():

    portInfo()
    #getEEG()


    while t == 'a':



        with serial.Serial(SERIAL_ARDUINO, 9600, timeout = 1,parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
            r = ser.read(128)
            print(r)

            ser.write('O')
            print("Sending O \n")


            s = ser.read(1)
            print(s)
            if 'K' in s:
                print("Serial connection confirmed \n")

main()


