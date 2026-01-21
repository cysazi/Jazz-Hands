// Jazz Hand (1/2) (RIGHT/LEFT) <- to be decided

const uint8_t relayMAC[] = { 1, 2, 3, 4, 5 };  // The MAC address of the relay ESP-32, constant for this device.

struct __attribute__((__packed__)) SensorData {  // has some special syntax to tell arduino to leave the raw data in byte form
  uint16_t header = 0xAAAA;                      // 2 bytes
  uint8_t device_id;                             // 1 byte
  uint32_t timestamp;                            // 4 bytes
  float accel_x, accel_y, accel_z;               // 4*3 = 12 bytes
  float UWB_distance1, UWB_distance2;            // 4*2 = 8 bytes
  uint8_t button_state;                          // 1 byte
  float quat_W, quat_x, quat_y, quat_z;          // 4*4 = 16 bytes
};                                               // Total: 44 bytes

void setup() {
  Serial.begin(115200);  // This project uses a higher baud rate than we did in BME 60A
  // the rest of the pin initializations or whatever
}

void loop() {
  SensorData current_readings = collectData();  // collect the data
  sendPacket(current_readings);                 // send the data
}

SensorData collectData() {  // this fn returns the struct with all the data
  SensorData data;          // initializing a local instance of the struct inside this fn

  // The data collection stuff goes here

  return data;  // returning the instance of the data to the outside world
}

void sendPacket(SensorData data) {  // sends the struct to relay ESP-32 via ESP-NOW
}