import binascii
import serial
import numpy as np
from serial.tools import list_ports
import configuration.record_conf as conf


def serial_close():
    ser_obci.write(b'v')  # Reset board
    ser_obci.close()


def device_start_stream():
    ser_obci.write(b'v')  # Reset board
    ser_obci.write(b'b')  # Start streaming
    print("Streaming data from OBCI initiated")


def device_data_retrieve(aux_data=False):
    s = ser_obci.read(1)
    if len(s) == 0:
        print("Obci serial timeout, no data received from OBCI")
        return 0
    if binascii.hexlify(s).decode('ascii') == "c0":
        s = ser_obci.read(1)
        if binascii.hexlify(s).decode('ascii') == "a0":
            sample = ser_obci.read(31)

            # Adding channel record_data into separate lists
            ch1 = int(binascii.hexlify(sample[1:4]).decode('ascii'), 16) * conf.scale_factor
            ch2 = int(binascii.hexlify(sample[4:7]).decode('ascii'), 16) * conf.scale_factor
            ch3 = int(binascii.hexlify(sample[7:10]).decode('ascii'), 16) * conf.scale_factor
            ch4 = int(binascii.hexlify(sample[10:13]).decode('ascii'), 16) * conf.scale_factor
            ch5 = int(binascii.hexlify(sample[13:16]).decode('ascii'), 16) * conf.scale_factor
            ch6 = int(binascii.hexlify(sample[16:19]).decode('ascii'), 16) * conf.scale_factor
            ch7 = int(binascii.hexlify(sample[19:22]).decode('ascii'), 16) * conf.scale_factor
            ch8 = int(binascii.hexlify(sample[22:25]).decode('ascii'), 16) * conf.scale_factor
            aux = sample[25]
            if aux_data:
                return np.array([ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, aux])
            else:
                return np.array([ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8])
        else:
            return 0
    else:
        return 0


def print_available_ports():
    print('''
================================
OBCI dongle: 'USB Serial Port'
Arduino    : 'USB-SERIAL CH340'
================================''')
    print("COM port list:")
    for port in list_ports.comports():
        print(port)
    print("\n")


try:
    ser_obci = serial.Serial(port=conf.COM_obci, baudrate=conf.BAUD_obci, parity=serial.PARITY_NONE,
                             stopbits=serial.STOPBITS_ONE,
                             bytesize=serial.EIGHTBITS, timeout=0.05)
except Exception as e:
    print(e)
    print_available_ports()
    print("Change COM ports in record_conf.py, exiting program")
    quit()
