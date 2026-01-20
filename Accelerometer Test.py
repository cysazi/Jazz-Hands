import numpy
import serial
from dataclasses import dataclass

# Definition of constants
COM_PORT: str = 'COM5'
PACKET_SIZE: int = 34 # Number of bytes per packet:
# Header (1) + Device Number (1) + X,Y,Z Accel (4*3 = 12) + Ang Displacement (4*3 = 12) + Button State (1) + UWB (4x2=8)

ser = serial.Serial(port=COM_PORT, baudrate=115200)
@dataclass
class PacketData:
    device_number: int
    accel_x: float
    accel_y: float
    accel_z: float
    quaternion: tuple[float, float, float, float]
    button_state: bool

