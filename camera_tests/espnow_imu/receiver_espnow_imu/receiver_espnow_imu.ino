#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

const uint16_t PACKET_HEADER = 0xAAAA;
const uint16_t HAPTICS_COMMAND_HEADER = 0xCC33;
const uint8_t ESPNOW_WIFI_CHANNEL = 11;
const uint32_t SERIAL_BAUD = 921600;
const bool DEBUG_SERIAL = false;

uint8_t LEFT_GLOVE_MAC[6] = {0x34, 0x98, 0x7A, 0x74, 0x39, 0x00};
uint8_t RIGHT_GLOVE_MAC[6] = {0x34, 0x98, 0x7A, 0x73, 0x93, 0x14};

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

typedef struct __attribute__((packed)) {
  uint16_t header;
  uint8_t device_id;
  uint8_t intensity;
  uint16_t duration_ms;
} haptics_command_t;

typedef struct {
  uint8_t device_id;
  uint8_t *mac;
} glove_peer_t;

static_assert(sizeof(hand_imu_packet_t) == 42, "Unexpected hand_imu_packet_t size");
static_assert(sizeof(haptics_command_t) == 6, "Unexpected haptics_command_t size");

glove_peer_t glove_peers[2] = {
  {1, LEFT_GLOVE_MAC},
  {2, RIGHT_GLOVE_MAC},
};

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

bool addPeerIfMissing(const uint8_t *peer_mac) {
  if (esp_now_is_peer_exist(peer_mac)) {
    return true;
  }

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, peer_mac, 6);
  peer.channel = ESPNOW_WIFI_CHANNEL;
  peer.encrypt = false;
  return esp_now_add_peer(&peer) == ESP_OK;
}

const uint8_t *peerMacForDevice(uint8_t device_id) {
  for (const glove_peer_t &peer : glove_peers) {
    if (peer.device_id == device_id) {
      return peer.mac;
    }
  }
  return nullptr;
}

void forwardHapticsCommand(const haptics_command_t &command) {
  const uint8_t *target_mac = peerMacForDevice(command.device_id);
  if (target_mac == nullptr) {
    if (DEBUG_SERIAL) {
      Serial.printf("Unknown haptics device_id=%u\n", command.device_id);
    }
    return;
  }
  if (!addPeerIfMissing(target_mac)) {
    if (DEBUG_SERIAL) {
      Serial.printf("Could not add peer for device_id=%u\n", command.device_id);
    }
    return;
  }

  esp_now_send(target_mac, reinterpret_cast<const uint8_t *>(&command), sizeof(command));
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

  Serial.write(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
}

void processSerialCommands() {
  const size_t expected = sizeof(haptics_command_t);
  if (Serial.available() < static_cast<int>(expected)) {
    return;
  }

  while (Serial.available() >= static_cast<int>(expected)) {
    int first = Serial.peek();
    if (first < 0) {
      return;
    }

    if (static_cast<uint8_t>(first) != static_cast<uint8_t>(HAPTICS_COMMAND_HEADER & 0xFF)) {
      Serial.read();
      continue;
    }

    uint8_t buffer[sizeof(haptics_command_t)];
    size_t bytes_read = Serial.readBytes(reinterpret_cast<char *>(buffer), expected);
    if (bytes_read != expected) {
      return;
    }

    haptics_command_t command;
    memcpy(&command, buffer, sizeof(command));
    if (command.header != HAPTICS_COMMAND_HEADER) {
      continue;
    }
    forwardHapticsCommand(command);
  }
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

  for (const glove_peer_t &peer : glove_peers) {
    addPeerIfMissing(peer.mac);
  }

  if (DEBUG_SERIAL) {
    Serial.println("ESP-NOW IMU receiver ready");
  }
}

void loop() {
  processSerialCommands();
  delay(1);
}
