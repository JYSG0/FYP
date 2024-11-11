import time
import struct
import math
from mag_base import mag_base

class QMC5883L(mag_base):
    # Probe the existence of const()
    try:
        _canary = const(0xfeed)
    except:
        const = lambda x: x

    # Default I2C address
    ADDR = const(0x0D)

    # QMC5883 Register numbers
    X_LSB = const(0)
    X_MSB = const(1)
    Y_LSB = const(2)
    Y_MSB = const(3)
    Z_LSB = const(4)
    Z_MSB = const(5)
    STATUS = const(6)
    T_LSB = const(7)
    T_MSB = const(8)
    CONFIG = const(9)
    CONFIG2 = const(10)
    RESET = const(11)
    STATUS2 = const(12)
    CHIP_ID = const(13)

    # Bit values for the STATUS register
    STATUS_DRDY = const(1)
    STATUS_OVL = const(2)
    STATUS_DOR = const(4)

    # Oversampling values for the CONFIG register
    CONFIG_OS512 = const(0b00000000)
    CONFIG_OS256 = const(0b01000000)
    CONFIG_OS128 = const(0b10000000)
    CONFIG_OS64 = const(0b11000000)

    # Range values for the CONFIG register
    CONFIG_2GAUSS = const(0b00000000)
    CONFIG_8GAUSS = const(0b00010000)

    # Rate values for the CONFIG register
    CONFIG_10HZ = const(0b00000000)
    CONFIG_50HZ = const(0b00000100)
    CONFIG_100HZ = const(0b00001000)
    CONFIG_200HZ = const(0b00001100)

    # Mode values for the CONFIG register
    CONFIG_STANDBY = const(0b00000000)
    CONFIG_CONT = const(0b00000001)

    # Mode values for the CONFIG2 register
    CONFIG2_INT_DISABLE = const(0b00000001)
    CONFIG2_ROL_PTR = const(0b01000000)
    CONFIG2_SOFT_RST = const(0b10000000)

    def __init__(self, i2c, offset=50.0):
        super().__init__(i2c)

        self.temp_offset = offset
        self.oversampling = QMC5883L.CONFIG_OS64
        self.range = QMC5883L.CONFIG_2GAUSS
        self.rate = QMC5883L.CONFIG_100HZ
        self.mode = QMC5883L.CONFIG_CONT
        self.register = bytearray(9)
        self.command = bytearray(1)
        self.reset()

        # Initialize calibration variables
        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0
        self.x_scale = 1
        self.y_scale = 1
        self.z_scale = 1

    def reset(self):
        self.command[0] = 1
        self.i2c.writeto_mem(QMC5883L.ADDR, QMC5883L.RESET, self.command)
        time.sleep(0.1)
        self.reconfig()

    def reconfig(self):
        self.command[0] = (self.oversampling | self.range |
                           self.rate | self.mode)
        self.i2c.writeto_mem(QMC5883L.ADDR, QMC5883L.CONFIG,
                             self.command)
        time.sleep(0.01)
        self.command[0] = QMC5883L.CONFIG2_INT_DISABLE
        self.i2c.writeto_mem(QMC5883L.ADDR, QMC5883L.CONFIG2,
                             self.command)
        time.sleep(0.01)

    def set_oversampling(self, sampling):
        if (sampling << 6) in (QMC5883L.CONFIG_OS512, QMC5883L.CONFIG_OS256,
                               QMC5883L.CONFIG_OS128, QMC5883L.CONFIG_OS64):
            self.oversampling = sampling << 6
            self.reconfig()
        else:
            raise ValueError("Invalid parameter")

    def set_range(self, rng):
        if (rng << 4) in (QMC5883L.CONFIG_2GAUSS, QMC5883L.CONFIG_8GAUSS):
            self.range = rng << 4
            self.reconfig()
        else:
            raise ValueError("Invalid parameter")

    def set_sampling_rate(self, rate):
        if (rate << 2) in (QMC5883L.CONFIG_10HZ, QMC5883L.CONFIG_50HZ,
                           QMC5883L.CONFIG_100HZ, QMC5883L.CONFIG_200HZ):
            self.rate = rate << 2
            self.reconfig()
        else:
            raise ValueError("Invalid parameter")

    def ready(self):
        status = self.i2c.readfrom_mem(QMC5883L.ADDR, QMC5883L.STATUS, 1)[0]
        if status == QMC5883L.STATUS_DOR:
            print("Incomplete read")
            return QMC5883L.STATUS_DRDY
        return status & QMC5883L.STATUS_DRDY

    def read_raw(self):
        try:
            while not self.ready():
                time.sleep(0.005)
            self.i2c.readfrom_mem_into(QMC5883L.ADDR, QMC5883L.X_LSB,
                                       self.register)
        except OSError as error:
            print("OSError", error)
            pass
        x, y, z, _, temp = struct.unpack('<hhhBh', self.register)
        return (x, y, z, temp)

    def read_scaled(self):
        x, y, z, temp = self.read_raw()
        scale = 12000 if self.range == QMC5883L.CONFIG_2GAUSS else 3000
        return (x / scale, y / scale, z / scale,
                (temp / 100 + self.temp_offset))

    def calibrate(self, num_samples=500):	#Number of read in raw values 500 by default
        #Initialise min, max with infinity
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        print("Move the sensor around to calibrate...")
        
        #Get min and max for each axis
        for _ in range(num_samples):
            x, y, z, _ = self.read_raw()

            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
            z_min, z_max = min(z_min, z), max(z_max, z)

            time.sleep(0.05)	#Delay for 0.05 seconds to not overwhelm the reading but not too slow

        #Calculate offsets and scales
        self.x_offset = (x_max + x_min) / 2
        self.y_offset = (y_max + y_min) / 2
        self.z_offset = (z_max + z_min) / 2

        self.x_scale = (x_max - x_min) / 2
        self.y_scale = (y_max - y_min) / 2
        self.z_scale = (z_max - z_min) / 2

        print("Calibration completed.")
        print(f"Offsets: X={self.x_offset}, Y={self.y_offset}, Z={self.z_offset}")
        print(f"Scales: X={self.x_scale}, Y={self.y_scale}, Z={self.z_scale}")

        return (self.x_offset, self.y_offset, self.z_offset,
                self.x_scale, self.y_scale, self.z_scale)
    
    #Apply offset and scale from calibration onto raw values to get calibrated values
    def apply_calibration(self, x, y, z):
        x_calibrated = (x - self.x_offset) / self.x_scale
        y_calibrated = (y - self.y_offset) / self.y_scale
        z_calibrated = (z - self.z_offset) / self.z_scale
        return x_calibrated, y_calibrated, z_calibrated
    
    def calculate_azimuth(self, x_calibrated, y_calibrated):
        azimuth = math.atan2(y_calibrated, x_calibrated) * (180 / math.pi)
        if azimuth < 0:
            azimuth += 360	#Normalise azimuth to 360
        return azimuth
    
    def get_cardinal_direction(self, azimuth):
        directions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']	#Try to match for every program that uses this cardinal directions array
        
        # Each direction corresponds to 45 degrees, with ±22.5° tolerance
        index = int((azimuth + 22.5) // 45) % 8  # Modulo 8 to wrap around after 360°
        
        return directions[index]