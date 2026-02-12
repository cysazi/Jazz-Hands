// All BNO085 Libraries
#include <Adafruit_BNO08x.h>
#include <Wire.h>

// All ESP NOW Libraries
#include <ESP32_NOW.h>
#include <ESP32_NOW_Serial.h>
#include <WiFi.h>

// ESP NOW Definitions
#define ESPNOW_WIFI_CHANNEL 6

// BNO085 definitions
#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22

// Jazz Hand (1/2) (LEFT/RIGHT) <- to be decided
#define HEADER 0xAAAA
#define DEVICE_ID 1


const uint8_t ESP_NOW_relayMAC[] = { 0x08, 0xF9, 0xE0, 0x92, 0xC0, 0x08 };  // The MAC address of the relay ESP-32 (device 4), constant for this device.


struct __attribute__((__packed__)) SensorData {  // has some special syntax to tell arduino to leave the raw data in byte form
  uint16_t header = HEADER;                      // 2 bytes
  uint8_t device_id = DEVICE_ID;                 // 1 byte
  uint8_t packet_type;                           // 1 byte
  uint32_t timestamp;                            // 4 bytes
  float accel_x, accel_y, accel_z;               // 4*3 = 12 bytes
  float UWB_distance1, UWB_distance2;            // 4*2 = 8 bytes
  uint8_t button_state;                          // 1 byte
  float quat_w, quat_i, quat_j, quat_k;          // 4*4 = 16 bytes
  uint8_t error_handler;                         // 1 byte
};                                               // Total: 57 bytes

// IMU Declarations
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Global Variables
struct SensorData current_readings = {};
current_readings.header = HEADER;
current_readings.device_id = DEVICE_ID;
bool ready_to_send;
bool has_rotation_vector = false;
uint32_t last_accel_time = 0;      // For 400Hz timing
uint32_t last_rotation_time = 0;   // For 100Hz timing


void setReports() {
  Serial.println("Setting desired reports");
  // 2500µs = 400Hz for acceleration, 10000µs = 100Hz for rotation vector
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    Serial.println("Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 2500)) {
    Serial.println("Could not enable linear acceleration");
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("BNO085 test");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);  // 400kHz for stability/faster I2C
  delay(100);

  // Initialize the IMU
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }

  // Intialize ESP-NOW
  WiFi.mode(WIFI_STA);
  esp_now_init();

  // Add relay as peer once
  esp_now_peer_info_t relay = {};
  memcpy(relay.peer_addr, ESP_NOW_relayMAC, 6);
  relay.channel = ESPNOW_WIFI_CHANNEL;      // Fixed channel for reliability
  relay.encrypt = false;  // No encryption for speed
  esp_now_add_peer(&relay);

  // Initialize the global struct properly
  current_readings.header = HEADER;
  current_readings.device_id = DEVICE_ID;
  current_readings.packet_type = 0x01;  // Start with accel-only packets

  Serial.println("BNO085 Found!");
  setReports();
  Serial.println("Reading events...");
}

void loop() {
  current_readings = collectData();  // collect the data
  // maybe check the integrity of the packet here? validatePacket()?
  if (ready_to_send) {
    ready_to_send = false;
    sendPacket(&current_readings);
  }  // send the data
  if (has_rotation_vector) {
  }
}

SensorData collectData() {  // this fn returns the struct with all the data
  SensorData data = {};
  if (bno08x.getSensorEvent(&sensorValue)) {
      ready_to_send = true;
      switch (sensorValue.sensorId) {
        case SH2_ROTATION_VECTOR:
          data.quat_w = sensorValue.un.rotationVector.real;
          data.quat_i = sensorValue.un.rotationVector.i;
          data.quat_j = sensorValue.un.rotationVector.j;
          data.quat_k = sensorValue.un.rotationVector.k;
          break;
        case SH2_LINEAR_ACCELERATION:
          data.accel_x = sensorValue.un.linearAcceleration.x;
          data.accel_y = sensorValue.un.linearAcceleration.y;
          data.accel_z = sensorValue.un.linearAcceleration.z;
          break;
      }
    }

  data.UWB_distance1 = 0;
  data.UWB_distance2 = 0;
  data.button_state = false;

  // The data collection stuff goes here
  data.timestamp = millis();  // do this last
  return data;                // returning the instance of the data to the outside world
}

void sendPacket(SensorData* data) {  // sends the struct to relay ESP-32 via ESP-NOW
  esp_now_send(ESP_NOW_relayMAC, (uint8_t*)data, sizeof(data))
}