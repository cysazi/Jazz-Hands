#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

const uint16_t PACKET_HEADER = 0xAAAA;
const uint8_t ESPNOW_WIFI_CHANNEL = 11;
const uint32_t SERIAL_BAUD = 921600;
const bool DEBUG_SERIAL = false;

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

void printMacAddress() {
  uint8_t mac[6];
  WiFi.macAddress(mac);
  Serial.print("Receiver STA MAC: ");
  for (int i = 0; i < 6; ++i) {
    if (i > 0) {
      Serial.print(":");
    }
    if (mac[i] < 16) {
      Serial.print("0");
    }
    Serial.print(mac[i], HEX);
  }
  Serial.println();
}

void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  (void)info;
  if (len != sizeof(hand_imu_packet_t)) {
    return;
  }

  hand_imu_packet_t packet;
  memcpy(&packet, data, sizeof(packet));
  if (packet.header != PACKET_HEADER) {
    return;
  }

  Serial.write((const uint8_t *)&packet, sizeof(packet));
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(ESPNOW_WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (DEBUG_SERIAL) {
    printMacAddress();
  }

  if (esp_now_init() != ESP_OK) {
    if (DEBUG_SERIAL) {
      Serial.println("Error initializing ESP-NOW");
    }
    while (true) {
      delay(100);
    }
  }

  esp_now_register_recv_cb(onDataRecv);

  if (DEBUG_SERIAL) {
    Serial.println("ESP-NOW IMU receiver ready");
  }
}

void loop() {
  delay(1);
}
