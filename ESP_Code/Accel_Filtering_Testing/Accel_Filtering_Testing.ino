// BNO085 Libraries
#include <Adafruit_BNO08x.h>
#include <Wire.h>

// Use the lower-level ESP-NOW library
#include <esp_now.h>
#include <WiFi.h>

// UWB Libraries
#include <SPI.h>
#include "DW1000Ranging.h"

// Local Kalman Filter Library
#include "Kalman.h"

// =================================================================
// =================== DEVICE CONFIGURATION ========================
// =================================================================
// UNCOMMENT the device you are flashing this code to.
// This determines the DEVICE_ID in the packet and the UWB address.

#define CONFIG_GLOVE_1 
// #define CONFIG_GLOVE_2

#define NUMBER_UWB_ANCHORS 4
// =================================================================

#ifdef CONFIG_GLOVE_1
#define DEVICE_ID 1
#define UWB_TAG_ADDRESS "7D:00:22:EA:82:60:3B:9C"  // Example address
#endif

#ifdef CONFIG_GLOVE_2
#define DEVICE_ID 2
#define UWB_TAG_ADDRESS "8E:01:33:EA:82:60:4C:8D"  // Example address
#endif


// --- Pin, Address, and Channel Definitions ---
#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22
#define UWB_1_ADDRESS 0x1111  // Short address of Anchor 1
#define UWB_2_ADDRESS 0x2222  // Short address of Anchor 2
#define UWB_3_ADDRESS 0x3333  // Short address of Anchor 3
#define UWB_4_ADDRESS 0x4444  // Short address of Anchor 4
#define UWB_5_ADDRESS 0x5555  // Short address of Anchor 5
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4
#define DW_RST 27
#define DW_IRQ 34
#define BUTTON_PIN 26

#define HEADER 0xAAAA
#define ESPNOW_WIFI_CHANNEL 11

#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000
#define PACKET_HAS_UWB_3 0b00010000
#define PACKET_HAS_UWB_4 0b00100000
#define PACKET_HAS_UWB_5 0b01000000
#define PACKET_HAS_ERROR 0b10000000

#define ROTATION_THRESHOLD_RAD_S 2 // 40 deg/s in rad/s

// MAC Address of the receiver (Anchor/Relay)
uint8_t receiverMac[6] = { 0x34, 0x98, 0x7A, 0x72, 0x93, 0xD4 };

// --- Data Structures ---
typedef struct __attribute__((__packed__)) {
  uint16_t header = HEADER;
  uint8_t device_id = DEVICE_ID;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2, UWB_distance3, UWB_distance4, UWB_distance5;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} datapacket_t;

typedef struct __attribute__((__packed__)) {
  char header;  // 'C'
  uint8_t device_id;
  float dx, dy, dz;
} correction_t;

// --- Global Variables and Instances ---
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
KalmanFilter kf;
datapacket_t current_packet = {};
float gyro_rad_s[3] = {0.0f, 0.0f, 0.0f};


const int ZUPT_BUFFER_SIZE = 8;
float accel_buffer[ZUPT_BUFFER_SIZE][3];
int zupt_buffer_index = 0;
bool is_stationary = false;
const float ZUPT_RELAXED = 0.015f;
const float ZUPT_STRICT = 0.06f;
const unsigned long ZUPT_COOLDOWN_MS = 300; // milliseconds
unsigned long zupt_last_change_ms = 0;
float last_zupt_var_sum = 0.0f;

// ========================================
// ESP-NOW Callbacks
// ========================================
void onDataRecv(const esp_now_recv_info_t* info, const uint8_t* data, int len) {
  // Debug: print source MAC and length
  Serial.print("onDataRecv from ");
  for (int i = 0; i < 6; ++i) {
    if (i > 0) Serial.print(":");
    Serial.print(info->src_addr[i], HEX);
  }
  Serial.print(" len="); Serial.println(len);

  // Handle incoming correction packets (packed correction_t)
  if (len == sizeof(correction_t)) {
    Serial.println("Receive Success: correction packet size matches");
    correction_t correction;
    memcpy(&correction, data, sizeof(correction));

    Serial.print("Parsed correction for device_id="); Serial.println(correction.device_id);

    if (correction.device_id == DEVICE_ID) {
      // Desired position = current KF position + correction vector
      float z[3];
      z[0] = kf.x[0] + correction.dx;
      z[1] = kf.x[1] + correction.dy;
      z[2] = kf.x[2] + correction.dz;

      // Apply as a measurement update so Kalman state and covariance are consistent
      kf.update(z);

      // Zero velocities to avoid immediate re-drift after a large correction
      kf.x[3] = 0.0f;
      kf.x[4] = 0.0f;
      kf.x[5] = 0.0f;

      // Debug logging
      Serial.print("Correction applied: dx="); Serial.print(correction.dx);
      Serial.print(" dy="); Serial.print(correction.dy);
      Serial.print(" dz="); Serial.println(correction.dz);
      Serial.print("New pos: "); Serial.print(kf.x[0]); Serial.print(", "); Serial.print(kf.x[1]); Serial.print(", "); Serial.println(kf.x[2]);
    }
  } else {
    Serial.println("Received unexpected packet size");
  }
}

void onDataSent(const esp_now_send_info_t* tx_info, esp_now_send_status_t status) {
  // Optional: Handle send status
}

void setReports() {
  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {  // 100Hz
    Serial.println("Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 2500)) {  // 400Hz
    Serial.println("Could not enable linear acceleration");
  }
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) { // 100Hz
    Serial.println("Could not enable gyroscope");
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO085 Found!");
  setReports();

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);
  esp_now_register_send_cb(onDataSent);

  // Add relay peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = ESPNOW_WIFI_CHANNEL;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add relay peer");
    return;
  }
  Serial.println("ESP-NOW Initialized and Peer Added.");

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);

  DW1000Ranging.attachNewRange(newRange);

  DW1000Ranging.startAsTag(UWB_TAG_ADDRESS, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);

  current_packet.header = HEADER;
  current_packet.device_id = DEVICE_ID;
}

void processIMU() {
  static unsigned long last_predict_time = 0;
  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
      gyro_rad_s[0] = sensorValue.un.gyroscope.x;
      gyro_rad_s[1] = sensorValue.un.gyroscope.y;
      gyro_rad_s[2] = sensorValue.un.gyroscope.z;
    }
    else if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
      float ax = sensorValue.un.linearAcceleration.x;
      float ay = sensorValue.un.linearAcceleration.y;
      float az = sensorValue.un.linearAcceleration.z;

      accel_buffer[zupt_buffer_index][0] = ax;
      accel_buffer[zupt_buffer_index][1] = ay;
      accel_buffer[zupt_buffer_index][2] = az;
      zupt_buffer_index = (zupt_buffer_index + 1) % ZUPT_BUFFER_SIZE;
      updateZUPT();

      unsigned long now = micros();
      float dt = (last_predict_time == 0) ? 0.0025f : (now - last_predict_time) / 1000000.0f;
      last_predict_time = now;

      float rotation_mag = sqrt(gyro_rad_s[0] * gyro_rad_s[0] + gyro_rad_s[1] * gyro_rad_s[1] + gyro_rad_s[2] * gyro_rad_s[2]);

      if (rotation_mag > ROTATION_THRESHOLD_RAD_S) {
        // During large rotations, reset velocities and avoid integrating accelerations
        kf.x[3] = 0;
        kf.x[4] = 0;
        kf.x[5] = 0;
        kf.predict(0, 0, 0, dt);
      } else if (is_stationary) {
        // If deeply stationary (var below relaxed), hard zero velocities.
        if (last_zupt_var_sum < ZUPT_RELAXED) {
          kf.x[3] = 0;
          kf.x[4] = 0;
          kf.x[5] = 0;
          kf.predict(0, 0, 0, dt);
        } else {
          // In hysteresis zone: allow velocity to decay smoothly instead of instant zeroing.
          float damping = 1.0f - constrain(5.0f * dt, 0.0f, 0.5f); // scale with dt
          kf.x[3] *= damping;
          kf.x[4] *= damping;
          kf.x[5] *= damping;
          kf.predict(0, 0, 0, dt);
        }
      } else {
        kf.predict(ax, ay, az, dt);
      }
    } else if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      current_packet.quat_w = sensorValue.un.rotationVector.real;
      current_packet.quat_i = sensorValue.un.rotationVector.i;
      current_packet.quat_j = sensorValue.un.rotationVector.j;
      current_packet.quat_k = sensorValue.un.rotationVector.k;
      sendPacket();
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

  float var_sum = var_x + var_y + var_z;
  unsigned long now = millis();

  // Hysteresis: enter stationary when below relaxed threshold, exit when above strict threshold.
  if (!is_stationary) {
    if (var_sum < ZUPT_RELAXED && (now - zupt_last_change_ms) >= ZUPT_COOLDOWN_MS) {
      is_stationary = true;
      zupt_last_change_ms = now;
    }
  } else {
    if (var_sum > ZUPT_STRICT && (now - zupt_last_change_ms) >= ZUPT_COOLDOWN_MS) {
      is_stationary = false;
      zupt_last_change_ms = now;
    }
  }

  last_zupt_var_sum = var_sum;
}

void sendPacket() {
  current_packet.timestamp = micros();
  current_packet.button_state = digitalRead(BUTTON_PIN);
  current_packet.pos_x = kf.x[0];
  current_packet.pos_y = kf.x[1];
  current_packet.pos_z = kf.x[2];
  current_packet.vel_x = kf.x[3];
  current_packet.vel_y = kf.x[4];
  current_packet.vel_z = kf.x[5];

  esp_now_send(receiverMac, (uint8_t*)&current_packet, sizeof(datapacket_t));
  current_packet.packet_type &= ~(PACKET_HAS_UWB_1 | PACKET_HAS_UWB_2 | PACKET_HAS_UWB_3 | PACKET_HAS_UWB_4 | PACKET_HAS_UWB_5);
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("BNO085 reset");
    setReports();
  }
  processIMU();
  DW1000Ranging.loop();
}

void newRange() {
  DW1000Device* device = DW1000Ranging.getDistantDevice();
  int shortAddress = device->getShortAddress();
  if (shortAddress == UWB_1_ADDRESS) {
    current_packet.UWB_distance1 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_1;
  } else if (shortAddress == UWB_2_ADDRESS) {
    current_packet.UWB_distance2 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_2;
  } else if (shortAddress == UWB_3_ADDRESS) {
    current_packet.UWB_distance3 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_3;
  } else if (shortAddress == UWB_4_ADDRESS) {
    current_packet.UWB_distance4 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_4;
  } else if (shortAddress == UWB_5_ADDRESS) {
    current_packet.UWB_distance5 = device->getRange();
    current_packet.packet_type |= PACKET_HAS_UWB_5;
  }
}

void newDevice(DW1000Device* device) {}
void inactiveDevice(DW1000Device* device) {}
