

// All BNO085 Libraries
#include <Adafruit_BNO08x.h>
#include <Wire.h>

// All ESP NOW Libraries
#include <ESP32_NOW.h>
#include <ESP32_NOW_Serial.h>

// BNO085 definitions
#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22

// Jazz Hand (1/2) (RIGHT/LEFT) <- to be decided
#define HEADER = 0xAAAA;
#define DEVICE_ID = 1;


const uint8_t ESP_NOW_relayMAC[6] = { 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF };  // The MAC address of the relay ESP-32, constant for this device.


struct __attribute__((__packed__)) SensorData {          // has some special syntax to tell arduino to leave the raw data in byte form
  uint16_t header = HEADER;                              // 2 bytes
  uint8_t device_id = DEVICE_ID;                         // 1 byte
  uint32_t timestamp;                                    // 4 bytes
  float accel_x, accel_y, accel_z;                       // 4*3 = 12 bytes
  float UWB_distance1, UWB_distance2;                    // 4*2 = 8 bytes
  uint8_t button_state;                                  // 1 byte
  float quat_w, quat_i, quat_j, quat_k;                  // 4*4 = 16 bytes
  uint8_t error_handler;                                 // 1 byte
};                                                       // Total: 56 bytes

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

void setReports(sh2_SensorId_t reportType, long report_interval) {
  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(reportType, report_interval)) {
    Serial.println("Could not enable stabilized remote vector");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);  // pause until serial actually opens

  // Try to initialize!
  if (!bno08x.begin_I2C()) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO08x Found!");


  setReports(SH2_ROTATION_VECTOR, 10000); // 100Hz
  setReports(SH2_LINEAR_ACCELERATION, 10000); // 100Hz

  Serial.println("Reading events");
  delay(100);
}


struct SensorData current_readings = {};
current_readings.header = HEADER;
current_readings.device_id = DEVICE_ID;

void loop() {
  current_readings = collectData();  // collect the data
  // maybe check the integrity of the packet here? validatePacket()?
  if (ready_to_send) { sendPacket(current_readings); }  // send the data
}

SensorData collectData(struct SensorData data) {  // this fn returns the struct with all the data
  
  
  // The data collection stuff goes here
  data.timestamp = millis();  // do this last
  return data;                // returning the instance of the data to the outside world
}

void sendPacket(SensorData data) {  // sends the struct to relay ESP-32 via ESP-NOW

}