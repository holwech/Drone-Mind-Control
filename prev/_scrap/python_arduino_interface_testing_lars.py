import serial
import time

ser = serial.Serial(
	port='COM12',
	baudrate=115200,
	parity=serial.PARITY_NONE,
	stopbits=serial.STOPBITS_ONE,
	bytesize=serial.EIGHTBITS
)

def RGB_config(l):
    ser.write((str(l).encode()))
    ser.write(b'\r')


def RGB(n):
    #time.sleep(1.6)  # Will not work without the delaY?!
    ser.write(n)


def main():
    ser.write(b'a\r')       # Dummy signal at start of Python program needed to establish serial connection
    time.sleep(2)

    RGB_config(1000)
    time.sleep(2)

    RGB(b'r\r')
    time.sleep(3)


main()






