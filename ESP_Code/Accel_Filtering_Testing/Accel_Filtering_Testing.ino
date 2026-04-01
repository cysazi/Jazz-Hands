// BNO085 Libraries
#include <Adafruit_BNO08x.h>
#include <Wire.h>

// Use the higher-level ESP-NOW wrapper libraries
#include <ESP32_NOW.h>
#include <ESP32_NOW_Serial.h>
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

// =================================================================

#ifdef CONFIG_GLOVE_1
  #define DEVICE_ID 1
  #define UWB_TAG_ADDRESS "7D:00:22:EA:82:60:3B:9C" // Example address
#endif

#ifdef CONFIG_GLOVE_2
  #define DEVICE_ID 2
  #define UWB_TAG_ADDRESS "8E:01:33:EA:82:60:4C:8D" // Example address
#endif


// --- Pin, Address, and Channel Definitions ---
#define BNO08X_RESET -1
#define SDA_PIN 21
#define SCL_PIN 22
#define UWB_1_ADDRESS 0x0084 // Short address of Anchor 1
#define UWB_2_ADDRESS 0x0085 // Short address of Anchor 2
#define UWB_3_ADDRESS 0x0086 // Short address of Anchor 3
#define UWB_4_ADDRESS 0x0087 // Short address of Receiver Anchor 4
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4
const uint8_t PIN_RST = 27;
const uint8_t PIN_IRQ = 34;
const uint8_t PIN_SS = 4;
#define BUTTON_PIN 33

#define HEADER 0xAAAA
#define ESPNOW_WIFI_CHANNEL 6

#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000
#define PACKET_HAS_UWB_3 0b00010000
#define PACKET_HAS_UWB_4 0b00100000
#define PACKET_HAS_ACCEL 0b00000001
#define PACKET_HAS_QUAT 0b00000010
#define PACKET_HAS_ERROR 0b10000000

// MAC Address of the receiver (Anchor/Relay)
uint8_t receiverMac[] = {0x08, 0xF9, 0xE0, 0x92, 0xC0, 0x08};

// --- Data Structures ---
typedef struct __attribute__((__packed__)) {
  uint16_t header = HEADER;
  uint8_t device_id = DEVICE_ID;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float accel_x, accel_y, accel_z;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2, UWB_distance3, UWB_distance4;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} datapacket_t; // 78 bytes

typedef struct __attribute__((__packed__)) {
  char header; // 'C'
  uint8_t device_number;
  float dx, dy, dz;
} correction_t;

// --- Global Variables and Instances ---
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
KalmanFilter kf;
datapacket_t current_packet = {};

const int ZUPT_BUFFER_SIZE = 15;
float accel_buffer[ZUPT_BUFFER_SIZE][3];
int zupt_buffer_index = 0;
bool is_stationary = false;
const float ZUPT_STATIONARY_THRESHOLD = 0.05f;

// ========================================
// ESP-NOW Peer Class for the Relay
// ========================================
class RelayPeer : public ESP_NOW_Peer {
public:
  RelayPeer(const uint8_t *mac_addr) : ESP_NOW_Peer(mac_addr, ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, nullptr) {}

  // Public wrapper to add this peer
  bool begin() {
    return add();
  }

  // Public wrapper to send data
  void send_data(const datapacket_t* data) {
    send((uint8_t*)data, sizeof(datapacket_t));
  }

protected:
  // Handle incoming correction packets
  void onReceive(const uint8_t *data, size_t len, bool broadcast) {
    if (len == sizeof(correction_t)) {
      correction_t correction;
      memcpy(&correction, data, sizeof(correction));
      if (correction.header == 'C') {
        kf.x[0] += correction.dx;
        kf.x[1] += correction.dy;
        kf.x[2] += correction.dz;
      }
    }
  }
};

// --- Global Instance of the Relay Peer ---
RelayPeer relay(receiverMac);

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

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO085 Found!");
  setReports();

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);
  while(!WiFi.STA.started()) { delay(100); }

  if (!ESP_NOW.begin()) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  if (!relay.begin()) {
    Serial.println("Failed to add relay peer");
    return;
  }
  Serial.println("ESP-NOW Initialized and Peer Added.");

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.startAsTag(UWB_TAG_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);

  current_packet.header = HEADER;
  current_packet.device_id = DEVICE_ID;
}

void processIMU() {
  static unsigned long last_predict_time = 0;
  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
      float ax = sensorValue.un.linearAcceleration.x;
      float ay = sensorValue.un.linearAcceleration.y;
      float az = sensorValue.un.linearAcceleration.z;
      current_packet.packet_type |= PACKET_HAS_ACCEL;
      current_packet.accel_x = ax;
      current_packet.accel_y = ay;
      current_packet.accel_z = az;

      accel_buffer[zupt_buffer_index][0] = ax;
      accel_buffer[zupt_buffer_index][1] = ay;
      accel_buffer[zupt_buffer_index][2] = az;
      zupt_buffer_index = (zupt_buffer_index + 1) % ZUPT_BUFFER_SIZE;
      updateZUPT();

      unsigned long now = micros();
      float dt = (last_predict_time == 0) ? 0.0025f : (now - last_predict_time) / 1000000.0f;
      last_predict_time = now;

      if (is_stationary) {
        kf.x[3] = 0; kf.x[4] = 0; kf.x[5] = 0;
        kf.predict(0, 0, 0, dt);
      } else {
        kf.predict(ax, ay, az, dt);
      }
    } else if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      current_packet.packet_type |= PACKET_HAS_QUAT;
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

  is_stationary = (var_x + var_y + var_z) < ZUPT_STATIONARY_THRESHOLD;
}

void sendPacket() {
  current_packet.timestamp = micros();
  current_packet.button_state = digitalRead(BUTTON_PIN);
  current_packet.packet_type |= (PACKET_HAS_ACCEL | PACKET_HAS_QUAT);
  current_packet.pos_x = kf.x[0];
  current_packet.pos_y = kf.x[1];
  current_packet.pos_z = kf.x[2];
  current_packet.vel_x = kf.x[3];
  current_packet.vel_y = kf.x[4];
  current_packet.vel_z = kf.x[5];

  relay.send_data(&current_packet);

  current_packet.packet_type &= ~(PACKET_HAS_UWB_1 | PACKET_HAS_UWB_2 | PACKET_HAS_UWB_3 | PACKET_HAS_UWB_4);
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
  }
}

void newDevice(DW1000Device* device) {}
void inactiveDevice(DW1000Device* device) {}
