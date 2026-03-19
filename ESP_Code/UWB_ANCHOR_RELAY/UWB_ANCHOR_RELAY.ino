// Use the higher-level ESP-NOW wrapper libraries
#include <ESP32_NOW.h>
#include <WiFi.h>

// UWB Libraries
#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>

// ========================================
// Pin, Address, and Channel Definitions
// ========================================
#define SPI_SCK  18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS    4
#define DW_RST   27
#define DW_IRQ   3

#define ANCHOR_ADDRESS "84:00:5B:D5:A9:9A:11:11"
#define ESPNOW_WIFI_CHANNEL 6

// MAC addresses of the glove devices this anchor will communicate with
uint8_t glove1_mac[] = {0x34, 0x98, 0x7A, 0x74, 0x39, 0x14}; // Device 1
uint8_t glove2_mac[] = {0x34, 0x98, 0x7A, 0x73, 0x75, 0xB8}; // Device 2

// ========================================
// Data Structures
// ========================================
typedef struct __attribute__((__packed__)) {
  uint16_t header;
  uint8_t device_id;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} datapacket_t;

typedef struct __attribute__((__packed__)) {
  char header; // 'C'
  float dx, dy, dz;
} correction_t;

// ========================================
// ESP-NOW Peer Class for Gloves
// ========================================
class GlovePeer : public ESP_NOW_Peer {
public:
  GlovePeer(const uint8_t *mac_addr) : ESP_NOW_Peer(mac_addr, ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, nullptr) {}

  // Public wrapper to access the protected 'add' method
  bool pair() {
    return add();
  }

  // Public wrapper to access the protected 'send' method
  bool send_correction(const correction_t* correction) {
    send((uint8_t*)correction, sizeof(correction_t));
  }

protected:
  // This is called when data is received from THIS specific peer
  void onReceive(const uint8_t *data, size_t len, bool broadcast) {
    if (len == sizeof(datapacket_t)) {
      Serial.write(data, len);
    }
  }
};

// ========================================
// Global Variables
// ========================================
GlovePeer glove1(glove1_mac);
GlovePeer glove2(glove2_mac);
std::vector<GlovePeer*> glove_peers = {&glove1, &glove2};

// ========================================
// ESP-NOW Callback for New Peers
// ========================================
void onNewPeer(const esp_now_recv_info_t *info, const uint8_t *data, int len, void *arg) {
  for (auto peer : glove_peers) {
    // Check if the MAC address matches one of our known gloves
    if (memcmp(info->src_addr, peer->addr(), 6) == 0) {
      // If it's not already paired, pair with it.
      if (!peer->isPaired()) {
        Serial.printf("Pairing with new glove: %02X:%02X:%02X:%02X:%02X:%02X\n",
                      info->src_addr[0], info->src_addr[1], info->src_addr[2],
                      info->src_addr[3], info->src_addr[4], info->src_addr[5]);
        peer->pair();
      }
      break;
    }
  }
}

// ========================================
// Setup
// ========================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("UWB Anchor & ESP-NOW Relay Starting...");

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);
  while(!WiFi.STA.started()) { delay(100); }

  if (!ESP_NOW.begin()) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  ESP_NOW.onNewPeer(onNewPeer, nullptr);
  Serial.println("ESP-NOW Initialized. Waiting for gloves to connect...");

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY);

  Serial.println("Anchor ready!");
}

// ========================================
// Main Loop & Correction Logic
// ========================================
void loop() {
  DW1000Ranging.loop();
  checkForCorrectionCommand();
}

void sendCorrection(uint8_t* target_mac, float dx, float dy, float dz) {
  correction_t correction;
  correction.header = 'C';
  correction.dx = dx;
  correction.dy = dy;
  correction.dz = dz;

  for (auto peer : glove_peers) {
    if (memcmp(target_mac, peer->addr(), 6) == 0) {
      peer->send_correction(&correction);
      break;
    }
  }
}

void checkForCorrectionCommand() {
  if (Serial.available() >= 1 + 6 + 12) {
    if (Serial.read() == 'C') {
      uint8_t target_mac[6];
      Serial.readBytes(target_mac, 6);

      float dx, dy, dz;
      Serial.readBytes((char*)&dx, sizeof(dx));
      Serial.readBytes((char*)&dy, sizeof(dy));
      Serial.readBytes((char*)&dz, sizeof(dz));

      sendCorrection(target_mac, dx, dy, dz);
    }
  }
}

// ========================================
// UWB Callbacks (can be left empty)
// ========================================
void onNewRange() {}
void onNewDevice(DW1000Device* device) {}
void onInactiveDevice(DW1000Device* device) {}
