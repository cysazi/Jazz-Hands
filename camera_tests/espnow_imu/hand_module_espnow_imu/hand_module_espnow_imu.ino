#include <Adafruit_BNO08x.h>
#include <Wire.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

// Flash one hand at a time. Change this one setting before uploading.
#define LEFT_HAND 1
#define RIGHT_HAND 2
#define HAND_TO_FLASH LEFT_HAND

#if HAND_TO_FLASH == LEFT_HAND
const uint8_t HAND_DEVICE_ID = 1;
const char HAND_NAME[] = "LEFT";
#elif HAND_TO_FLASH == RIGHT_HAND
const uint8_t HAND_DEVICE_ID = 2;
const char HAND_NAME[] = "RIGHT";
#else
#error "HAND_TO_FLASH must be LEFT_HAND or RIGHT_HAND."
#endif

const uint16_t PACKET_HEADER = 0xAAAA;
const uint8_t ESPNOW_WIFI_CHANNEL = 11;
const uint32_t SEND_RATE_HZ = 120;
const uint32_t SEND_PERIOD_US = 1000000UL / SEND_RATE_HZ;
const uint32_t IMU_REPORT_INTERVAL_US = 1000000UL / SEND_RATE_HZ;
const uint32_t IMU_DATA_STALE_US = 100000UL;

uint8_t RECEIVER_MAC[6] = {0x08, 0xF9, 0xE0, 0x92, 0xC0, 0x08};

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int BUTTON_PIN = 23;
const int BNO08X_RESET = -1;
const uint8_t BNO08X_I2C_ADDRESS = 0x4A;

const float ACCEL_FILTER_ALPHA = 0.20f;

const uint8_t PACKET_HAS_ACCEL = 0b00000001;
const uint8_t PACKET_HAS_QUAT = 0b00000010;
const uint8_t PACKET_HAS_BUTTON = 0b00000100;
const uint8_t PACKET_HAS_ERROR = 0b10000000;

const uint8_t ERROR_ESPNOW_SEND = 0b00000001;
const uint8_t ERROR_QUAT_STALE = 0b00000010;
const uint8_t ERROR_ACCEL_STALE = 0b00000100;

typedef struct __attribute__((packed)) {
  uint16_t header;
  uint8_t device_id;
  uint32_t timestamp_us;
  uint32_t sequence;
  uint8_t packet_type;
  uint8_t button_pressed;
  float accel_x;
  float accel_y;
  float accel_z;
  float quat_w;
  float quat_i;
  float quat_j;
  float quat_k;
  uint8_t error_handler;
} hand_imu_packet_t;

static_assert(sizeof(hand_imu_packet_t) == 42, "Unexpected hand_imu_packet_t size");

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensor_value;

hand_imu_packet_t packet = {};
uint32_t sequence_number = 0;
uint32_t last_send_us = 0;
uint32_t last_quat_us = 0;
uint32_t last_accel_us = 0;
bool have_quat = false;
bool have_accel = false;
bool last_send_ok = true;

float filtered_accel_x = 0.0f;
float filtered_accel_y = 0.0f;
float filtered_accel_z = 0.0f;

void onDataSent(const esp_now_send_info_t *tx_info, esp_now_send_status_t status) {
  (void)tx_info;
  last_send_ok = (status == ESP_NOW_SEND_SUCCESS);
}

void setReports() {
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, IMU_REPORT_INTERVAL_US)) {
    Serial.println("Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, IMU_REPORT_INTERVAL_US)) {
    Serial.println("Could not enable linear acceleration");
  }
}

void setupEspNow() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    while (true) {
      delay(100);
    }
  }

  esp_now_register_send_cb(onDataSent);

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, RECEIVER_MAC, 6);
  peer.channel = ESPNOW_WIFI_CHANNEL;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Failed to add ESP-NOW receiver peer");
    while (true) {
      delay(100);
    }
  }
}

void setupImu() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  delay(100);

  if (!bno08x.begin_I2C(BNO08X_I2C_ADDRESS, &Wire)) {
    Serial.println("Failed to find BNO085");
    while (true) {
      delay(100);
    }
  }

  Serial.println("BNO085 found");
  setReports();
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  packet.header = PACKET_HEADER;
  packet.device_id = HAND_DEVICE_ID;
  packet.quat_w = 1.0f;

  setupImu();
  setupEspNow();

  Serial.print("ESP-NOW IMU hand module ready: ");
  Serial.println(HAND_NAME);
}

void updateImu() {
  if (bno08x.wasReset()) {
    setReports();
  }

  while (bno08x.getSensorEvent(&sensor_value)) {
    switch (sensor_value.sensorId) {
      case SH2_ROTATION_VECTOR:
        packet.quat_w = sensor_value.un.rotationVector.real;
        packet.quat_i = sensor_value.un.rotationVector.i;
        packet.quat_j = sensor_value.un.rotationVector.j;
        packet.quat_k = sensor_value.un.rotationVector.k;
        last_quat_us = micros();
        have_quat = true;
        break;

      case SH2_LINEAR_ACCELERATION:
        filtered_accel_x = ACCEL_FILTER_ALPHA * sensor_value.un.linearAcceleration.x
          + (1.0f - ACCEL_FILTER_ALPHA) * filtered_accel_x;
        filtered_accel_y = ACCEL_FILTER_ALPHA * sensor_value.un.linearAcceleration.y
          + (1.0f - ACCEL_FILTER_ALPHA) * filtered_accel_y;
        filtered_accel_z = ACCEL_FILTER_ALPHA * sensor_value.un.linearAcceleration.z
          + (1.0f - ACCEL_FILTER_ALPHA) * filtered_accel_z;

        packet.accel_x = filtered_accel_x;
        packet.accel_y = filtered_accel_y;
        packet.accel_z = filtered_accel_z;
        last_accel_us = micros();
        have_accel = true;
        break;
    }
  }
}

void sendPacketIfDue() {
  uint32_t now_us = micros();
  if ((uint32_t)(now_us - last_send_us) < SEND_PERIOD_US) {
    return;
  }
  last_send_us = now_us;

  packet.timestamp_us = now_us;
  packet.sequence = sequence_number++;
  packet.button_pressed = (digitalRead(BUTTON_PIN) == LOW) ? 1 : 0;
  packet.packet_type = PACKET_HAS_BUTTON;
  packet.error_handler = 0;

  bool accel_is_fresh = have_accel && ((uint32_t)(now_us - last_accel_us) <= IMU_DATA_STALE_US);
  bool quat_is_fresh = have_quat && ((uint32_t)(now_us - last_quat_us) <= IMU_DATA_STALE_US);

  if (accel_is_fresh) {
    packet.packet_type |= PACKET_HAS_ACCEL;
  } else {
    packet.error_handler |= ERROR_ACCEL_STALE;
  }
  if (quat_is_fresh) {
    packet.packet_type |= PACKET_HAS_QUAT;
  } else {
    packet.error_handler |= ERROR_QUAT_STALE;
  }
  if (!last_send_ok) {
    packet.error_handler |= ERROR_ESPNOW_SEND;
  }
  if (packet.error_handler != 0) {
    packet.packet_type |= PACKET_HAS_ERROR;
  }

  esp_err_t result = esp_now_send(RECEIVER_MAC, (uint8_t *)&packet, sizeof(packet));
  if (result != ESP_OK) {
    last_send_ok = false;
  } else {
    last_send_ok = true;
  }
}

void loop() {
  updateImu();
  sendPacketIfDue();
}
