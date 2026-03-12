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

// Local Kalman Filter Library
#include "Kalman.h"

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
const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS = 4;

// Misc. definitions
#define BUTTON_PIN 33
#define HEADER 0xAAAA
#define DEVICE_ID 1

// Packet Type Bitflags
#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000
#define PACKET_HAS_ERROR 0b10000000

// New Packet Structure for Dead Reckoning
typedef struct __attribute__((__packed__)) {
  uint16_t header = HEADER;                   // 2 bytes
  uint8_t device_id = DEVICE_ID;              // 1 byte
  uint32_t timestamp;                         // 4 bytes
  uint8_t packet_type;                        // 1 byte
  uint8_t button_state;                       // 1 byte

  // Integrated State from Kalman Filter
  float pos_x, pos_y, pos_z;                  // 12 bytes
  float vel_x, vel_y, vel_z;                  // 12 bytes

  // Raw UWB distances (if available)
  float UWB_distance1, UWB_distance2;         // 8 bytes

  // Quaternion for rotation
  float quat_w, quat_i, quat_j, quat_k;       // 16 bytes

  uint8_t error_handler;                      // 1 byte
} datapacket_t;                               // Total: 58 bytes

// IMU Declarations
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Kalman Filter Instance
KalmanFilter kf;

// ZUPT (Zero-Velocity Update) state
const int ZUPT_BUFFER_SIZE = 15;
float accel_buffer[ZUPT_BUFFER_SIZE][3];
int zupt_buffer_index = 0;
bool is_stationary = false;
const float ZUPT_STATIONARY_THRESHOLD = 0.05f; // Variance threshold

// Global Variables
datapacket_t current_packet = {};
unsigned long last_send_time = 0;
const unsigned long SEND_INTERVAL_MS = 10; // 100Hz send rate

void setReports() {
  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) { // 100Hz
    Serial.println("Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 2500)) { // 400Hz
    Serial.println("Could not enable linear acceleration");
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  delay(100);

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO085 Found!");
  setReports();

  // UWB Initialization (remains the same)
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C", DW1000.MODE_SHORTDATA_FAST_ACCURACY);

  current_packet.header = HEADER;
  current_packet.device_id = DEVICE_ID;
}

// --- High-Frequency IMU Processing ---
void processIMU() {
  static unsigned long last_predict_time = 0;

  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
      float ax = sensorValue.un.linearAcceleration.x;
      float ay = sensorValue.un.linearAcceleration.y;
      float az = sensorValue.un.linearAcceleration.z;

      // Update ZUPT buffer
      accel_buffer[zupt_buffer_index][0] = ax;
      accel_buffer[zupt_buffer_index][1] = ay;
      accel_buffer[zupt_buffer_index][2] = az;
      zupt_buffer_index = (zupt_buffer_index + 1) % ZUPT_BUFFER_SIZE;

      updateZUPT();

      // Calculate dt for Kalman predict
      unsigned long now = micros();
      float dt = (last_predict_time == 0) ? 0.0025f : (now - last_predict_time) / 1000000.0f;
      last_predict_time = now;

      if (is_stationary) {
        // If stationary, force velocity to zero in the Kalman state
        kf.x[3] = 0; kf.x[4] = 0; kf.x[5] = 0;
        // Don't predict with acceleration noise
        kf.predict(0, 0, 0, dt);
      } else {
        // If moving, predict normally
        kf.predict(ax, ay, az, dt);
      }

    } else if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      current_packet.quat_w = sensorValue.un.rotationVector.real;
      current_packet.quat_i = sensorValue.un.rotationVector.i;
      current_packet.quat_j = sensorValue.un.rotationVector.j;
      current_packet.quat_k = sensorValue.un.rotationVector.k;
    }
  }
}

void updateZUPT() {
  float mean_x = 0, mean_y = 0, mean_z = 0;
  for (int i = 0; i < ZUPT_BUFFER_SIZE; ++i) {
    mean_x += accel_buffer[i][0];
    mean_y += accel_buffer[i][1];
    mean_z += accel_buffer[i][2];
  }
  mean_x /= ZUPT_BUFFER_SIZE;
  mean_y /= ZUPT_BUFFER_SIZE;
  mean_z /= ZUPT_BUFFER_SIZE;

  float var_x = 0, var_y = 0, var_z = 0;
  for (int i = 0; i < ZUPT_BUFFER_SIZE; ++i) {
    var_x += pow(accel_buffer[i][0] - mean_x, 2);
    var_y += pow(accel_buffer[i][1] - mean_y, 2);
    var_z += pow(accel_buffer[i][2] - mean_z, 2);
  }
  var_x /= ZUPT_BUFFER_SIZE;
  var_y /= ZUPT_BUFFER_SIZE;
  var_z /= ZUPT_BUFFER_SIZE;

  float total_variance = var_x + var_y + var_z;
  is_stationary = total_variance < ZUPT_STATIONARY_THRESHOLD;
}

// --- Low-Frequency Sending and Correction ---
void sendPacket() {
  current_packet.timestamp = micros();
  current_packet.button_state = digitalRead(BUTTON_PIN);

  // Populate packet with current Kalman state
  current_packet.pos_x = kf.x[0];
  current_packet.pos_y = kf.x[1];
  current_packet.pos_z = kf.x[2];
  current_packet.vel_x = kf.x[3];
  current_packet.vel_y = kf.x[4];
  current_packet.vel_z = kf.x[5];

  Serial.write((uint8_t*)&current_packet, sizeof(datapacket_t));

  // Reset UWB flags after sending
  current_packet.packet_type &= ~(PACKET_HAS_UWB_1 | PACKET_HAS_UWB_2);
}

void checkForCorrection() {
  if (Serial.available() > 0) {
    char header = Serial.read();
    if (header == 'C') { // 'C' for Correction
      float dx, dy, dz;
      if (Serial.readBytes((char*)&dx, sizeof(dx)) == sizeof(dx) &&
          Serial.readBytes((char*)&dy, sizeof(dy)) == sizeof(dy) &&
          Serial.readBytes((char*)&dz, sizeof(dz)) == sizeof(dz)) {

        // Apply correction to the Kalman filter's position state
        kf.x[0] += dx;
        kf.x[1] += dy;
        kf.x[2] += dz;
      }
    }
  }
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("BNO085 reset");
    setReports();
  }

  // High-frequency processing of all available IMU data
  processIMU();

  // Low-frequency tasks
  unsigned long now = millis();
  if (now - last_send_time >= SEND_INTERVAL_MS) {
    last_send_time = now;
    sendPacket();
  }

  // Always check for incoming corrections from PC
  checkForCorrection();

  // UWB ranging loop
  DW1000Ranging.loop();
}

// --- UWB Callbacks ---
void newRange() {
  DW1000Device* device = DW1000Ranging.getDistantDevice();
  int shortAddress = device->getShortAddress();

  if (shortAddress == UWB_1_ADDRESS) {
    current_packet.UWB_distance1 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_1;
  } else if (shortAddress == UWB_2_ADDRESS) {
    current_packet.UWB_distance2 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_2;
  }
}

// Empty UWB callbacks
void newDevice(DW1000Device* device) {}
void inactiveDevice(DW1000Device* device) {}
