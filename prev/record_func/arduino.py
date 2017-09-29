import time
import serial
import configuration.record_conf as conf
from record_func.obci import print_available_ports


def serial_wake():
    ser_arduino.write(b'w')  # Send dummy signal to wake up serial connection
    time.sleep(2)


def serial_close():
    ser_arduino.close()


def arduino_set_confmode():
    ser_arduino.write(b'q')


def arduino_character_send(text):
    b = bytearray()
    b.extend(map(ord, text))
    ser_arduino.write(b)


def arduino_read_char():
    if ser_arduino.in_waiting > 0:
        return ser_arduino.read(ser_arduino.in_waiting)


def timer_colordata_store():
    pass


# Prints the color lighted in the console at the given experiment time
def printLED(color, sample_time):
    color_text = ""
    if color == 'R': color_text = "RED"
    if color == 'G': color_text = "GREEN"
    if color == 'B': color_text = "BLUE"
    print("                                          " + color_text + " flashed at " + str(round(sample_time, 1)))


def config_send(mode):
    ser_arduino.write(b'c')
    ser_arduino.write(bytes([mode]))
    ser_arduino.write(bytes([int(conf.pulse_duration * 10)]))
    ser_arduino.write(bytes([int(conf.steadystate_duration)]))


try:
    ser_arduino = serial.Serial(port=conf.COM_arduino, baudrate=conf.BAUD_arduino, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                bytesize=serial.EIGHTBITS)
except Exception as e:
    print(e)
    print_available_ports()
    print("Change COM ports in record_conf.py, exiting program")
    quit()
