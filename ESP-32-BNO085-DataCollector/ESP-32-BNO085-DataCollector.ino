// All BNO085 Libraries
#include <Adafruit_BNO08x.h>
#include <Wire.h>

// All ESP NOW Libraries
#include <ESP32_NOW.h>
#include <ESP32_NOW_Serial.h>
#include <WiFi.h>

// All UWB Libraries
#include <SPI.h>
#include "DW1000Ranging.h"

// ESP NOW Definitions
#define ESPNOW_WIFI_CHANNEL 6
const uint8_t ESP_NOW_relayMAC[] = { 0x08, 0xF9, 0xE0, 0x92, 0xC0, 0x08 };  // The MAC address of the relay ESP-32 (device 4); constant for this device.

// BNO085 definitions
#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22

// UWB Definitions
#define UWB_1_ADDRESS 0x1111
#define UWB_2_ADDRESS 0x2222

#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4

// connection pins
const uint8_t PIN_RST = 27;  // reset pin
const uint8_t PIN_IRQ = 34;  // irq pin
const uint8_t PIN_SS = 4;    // spi select pin

// Misc. definitions
#define button_pin 23

// Jazz Hand (1/2) (LEFT/RIGHT) <- to be decided
#define HEADER 0xAAAA
#define DEVICE_ID 1

// Packet Type Bitflags
#define PACKET_HAS_ACCEL 0b00000001
#define PACKET_HAS_QUAT 0b00000010
#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000

// Packet struct definition
typedef struct __attribute__((__packed__)) {  // has some special syntax to tell arduino to leave the raw data in byte form
  uint16_t header = HEADER;                   // 2 bytes
  uint8_t device_id = DEVICE_ID;              // 1 byte
  uint8_t packet_type;                        // 1 byte
  unsigned long timestamp;                    // 8 bytes
  uint8_t button_state;                       // 1 byte
  float accel_x, accel_y, accel_z;            // 4*3 = 12 bytes
  float UWB_distance1, UWB_distance2;         // 4*2 = 8 bytes
  float quat_w, quat_i, quat_j, quat_k;       // 4*4 = 16 bytes
  uint8_t error_handler;                      // 1 byte
} datapacket_t;                               // Total: 46 bytes

// IMU Declarations
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Global Variables
datapacket_t current_readings = {};
bool ready_to_send;
bool has_rotation_vector = false;
uint32_t last_accel_time = 0;     // For 400Hz timing
uint32_t last_rotation_time = 0;  // For 100Hz timing
// Global variables for filtered acceleration
float filtered_acc_x = 0, filtered_acc_y = 0, filtered_acc_z = 0;
const float acc_alpha = 0.2;  // Constant between 0 and 1.

// Set desired data for BNO085
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
  // Initialize Serial
  Serial.begin(115200);
  delay(100);

  // Initialize Wire
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);  // 400kHz for stability/faster I2C
  delay(100);

  // Initialize the IMU
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip, please RST the ESP32 and try again");
    while (1) { delay(10); }
  }
  Serial.println("BNO085 Found!");
  setReports();

  // Intialize ESP-NOW
  WiFi.mode(WIFI_STA);
  esp_now_init();

  // Add relay as peer once
  esp_now_peer_info_t relay = {};
  memcpy(relay.peer_addr, ESP_NOW_relayMAC, 6);
  relay.channel = ESPNOW_WIFI_CHANNEL;  // Fixed channel for reliability
  relay.encrypt = false;                // No encryption for speed
  esp_now_add_peer(&relay);

  // Initialize the global struct
  current_readings.header = HEADER;
  current_readings.device_id = DEVICE_ID;
  current_readings.packet_type = 0x00;

  // UWB Initialization
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);

  // Start the module as a tag
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C", DW1000.MODE_SHORTDATA_FAST_ACCURACY);
}

void loop() {

  // UWB Data Collection
  DW1000Ranging.loop();

  // Button State Reading
  current_readings.button_state = digitalRead(button_pin);
  if (current_readings.button_state) {

  }
  // IMU Data Collection
  while (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        if (!current_readings.packet_type & PACKET_HAS_QUAT) {
          current_readings.quat_w = sensorValue.un.rotationVector.real;
          current_readings.quat_i = sensorValue.un.rotationVector.i;
          current_readings.quat_j = sensorValue.un.rotationVector.j;
          current_readings.quat_k = sensorValue.un.rotationVector.k;
          current_readings.packet_type |= PACKET_HAS_QUAT;  // tells python that this packet has quaternion
        }
        break;
      case SH2_LINEAR_ACCELERATION:
        // One pole recursive Low-Pass Filter
        filtered_acc_x = (acc_alpha * sensorValue.un.linearAcceleration.x) + (1.0 - acc_alpha) * filtered_acc_x;
        filtered_acc_y = (acc_alpha * sensorValue.un.linearAcceleration.y) + (1.0 - acc_alpha) * filtered_acc_y;
        filtered_acc_z = (acc_alpha * sensorValue.un.linearAcceleration.z) + (1.0 - acc_alpha) * filtered_acc_z;

        // Assign the smoothed values to the data packet
        current_readings.accel_x = filtered_acc_x;
        current_readings.accel_y = filtered_acc_y;
        current_readings.accel_z = filtered_acc_z;

        current_readings.packet_type |= PACKET_HAS_ACCEL;  // tells python that this packet has accel data
        break;
    }
    // If both Rotation and Accel are acquired, break out of the while loop regardless of queue contents
    if (current_readings.packet_type & (PACKET_HAS_ACCEL | PACKET_HAS_QUAT) == 0b00000011) {
      break;
    }
  }
  if (current_readings.packet_type & PACKET_HAS_QUAT) {
    sendPacket(&current_readings);
  }
}

void sendPacket(datapacket_t* data) {
  esp_err_t result = esp_now_send(ESP_NOW_relayMAC, (uint8_t*)data, sizeof(&data));
  data->packet_type = 0;  // Reset the packet type
  // Optional Error Handling
  if (result != ESP_OK) {
    current_readings.error_handler = 1;  // Set error flag
  } else {
    current_readings.error_handler = 0;  // Clear error flag
  }
}


void newRange() {
  // Get new UWB Data when available
  // Get the specific device that just talked to us
  DW1000Device* device = DW1000Ranging.getDistantDevice();

  // Extract its address
  int shortAddress = device->getShortAddress();


  switch (shortAddress) {
    case UWB_1_ADDRESS:
      current_readings.UWB_distance1 = DW1000Ranging.getDistantDevice()->getRange();
      current_readings.packet_type |= PACKET_HAS_UWB_1;
      break;
    case UWB_2_ADDRESS:
      current_readings.UWB_distance2 = DW1000Ranging.getDistantDevice()->getRange();
      current_readings.packet_type |= PACKET_HAS_UWB_2;
      break;
  }
}

void newDevice(DW1000Device* device) {
  // Serial.print("ranging init; 1 device added ! -> ");
  // Serial.print(" short:");
  // Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device* device) {
  // Serial.print("deleting inactive device: ");
  // Serial.println(device->getShortAddress(), HEX);
}
