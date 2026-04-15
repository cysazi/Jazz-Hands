// Use the lower-level ESP-NOW library
#include <esp_now.h>
#include <WiFi.h>

// UWB Libraries
#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>

// =================================================================
// =================== ANCHOR CONFIGURATION ========================
// =================================================================
// UNCOMMENT the anchor you are flashing this code to.
// This determines the UWB address and calibration sequence.

// #define CONFIG_ANCHOR_1  // This is the RELAY (connected to PC)
#define CONFIG_ANCHOR_2
// #define CONFIG_ANCHOR_3
// #define CONFIG_ANCHOR_4

#if !defined(CONFIG_ANCHOR_1) && \
    !defined(CONFIG_ANCHOR_2) && \
    !defined(CONFIG_ANCHOR_3) && \
    !defined(CONFIG_ANCHOR_4)
  #error "No anchor config defined! Uncomment one CONFIG_ANCHOR_X at the top."
#endif

// =================================================================

// ========================================
// Pin, Address, and Channel Definitions
// ========================================
#define SPI_SCK  18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS    4
#define DW_RST   27
#define DW_IRQ   34

#ifdef CONFIG_ANCHOR_1
  #define ANCHOR_ADDRESS "11:11:11:11:11:11:11:11"
  #define ANCHOR_SHORT_ADDRESS 0x1111
  #define ANCHOR_ID 1
  #define ANCHOR_DELAY 16537
  #define NEXT_ANCHOR_ADDRESS "22:22:22:22:22:22:22:22"
#endif

#ifdef CONFIG_ANCHOR_2
  #define ANCHOR_ADDRESS "22:22:22:22:22:22:22:22"
  #define ANCHOR_SHORT_ADDRESS 0x2222
  #define ANCHOR_ID 2
  #define ANCHOR_DELAY 16536
  #define NEXT_ANCHOR_ADDRESS "33:33:33:33:33:33:33:33"
#endif

#ifdef CONFIG_ANCHOR_3
  #define ANCHOR_ADDRESS "33:33:33:33:33:33:33:33"
  #define ANCHOR_SHORT_ADDRESS 0x3333
  #define ANCHOR_ID 3
  #define ANCHOR_DELAY 16533
  #define NEXT_ANCHOR_ADDRESS "44:44:44:44:44:44:44:44"
#endif

#ifdef CONFIG_ANCHOR_4
  #define ANCHOR_ADDRESS "44:44:44:44:44:44:44:44"
  #define ANCHOR_SHORT_ADDRESS 0x4444
  #define ANCHOR_ID 4
  #define ANCHOR_DELAY 16535
  #define NEXT_ANCHOR_ADDRESS "55:55:55:55:55:55:55:55"
#endif

#ifdef CONFIG_ANCHOR_5
  #define ANCHOR_ADDRESS "55:55:55:55:55:55:55:55"
  #define ANCHOR_SHORT_ADDRESS 0x5555
  #define ANCHOR_ID 5
  #define ANCHOR_DELAY 16535
  #define NEXT_ANCHOR_ADDRESS "11:11:11:11:11:11:11:11"
#endif

#define ESPNOW_WIFI_CHANNEL 11
#define CALIBRATION_HEADER 0xBBBB
#define CALIBRATION_SAMPLES 100  // Number of samples to average (~10 seconds at 10Hz)
#define NUM_UWB_ANCHORS 3  // Set to 3 or 4 (must match Python configuration!)

// MAC addresses of the glove devices this anchor will communicate with
uint8_t glove1_mac[6] = {0x34, 0x98, 0x7A, 0x74, 0x39, 0x14}; // Device 1
uint8_t glove2_mac[6] = {0x34, 0x98, 0x7A, 0x73, 0x75, 0xB8}; // Device 2

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
  float UWB_distance1, UWB_distance2, UWB_distance3, UWB_distance4, UWB_distance5;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} datapacket_t;

typedef struct __attribute__((__packed__)) {
  char header; // 'C'
  uint8_t device_number;
  float dx, dy, dz;
} correction_t;

typedef struct __attribute__((__packed__)) {
  uint16_t header; // 0xBBBB
  uint8_t source_anchor;
  uint8_t dest_anchor;
  float distance;
} calibration_packet_t;

// ========================================
// Global Variables
// ========================================
struct GlovePeerInfo {
  uint8_t mac[6];
  bool paired;
};

GlovePeerInfo glove_peers[2] = {
  {{0x34, 0x98, 0x7A, 0x74, 0x39, 0x14}, false}, // glove1_mac
  {{0x34, 0x98, 0x7A, 0x73, 0x75, 0xB8}, false}  // glove2_mac
};

// Calibration state
bool calibration_mode = true;
struct {
  int count;
  float sum;
} calibration_data[5];  // Index by anchor ID (1-4), 0 unused
int total_calibration_packets_sent = 0;

// ========================================
// ESP-NOW Callbacks
// ========================================
void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  // Forward data packets to serial
  if (len == sizeof(datapacket_t)) {
    Serial.write(data, len);
  }

  // Check if we need to pair with this peer
  for (int i = 0; i < 2; i++) {
    if (memcmp(info->src_addr, glove_peers[i].mac, 6) == 0) {
      if (!glove_peers[i].paired) {
        Serial.printf("Pairing with new glove: %02X:%02X:%02X:%02X:%02X:%02X\n",
                      info->src_addr[0], info->src_addr[1], info->src_addr[2],
                      info->src_addr[3], info->src_addr[4], info->src_addr[5]);

        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, glove_peers[i].mac, 6);
        peerInfo.channel = ESPNOW_WIFI_CHANNEL;
        peerInfo.encrypt = false;

        if (esp_now_add_peer(&peerInfo) == ESP_OK) {
          glove_peers[i].paired = true;
        }
      }
      break;
    }
  }
}

void onDataSent(const esp_now_send_info_t *tx_info, esp_now_send_status_t status) {
  // Optional: Handle send status
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

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);
  esp_now_register_send_cb(onDataSent);
  Serial.println("ESP-NOW Initialized. Waiting for gloves to connect...");

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000.setAntennaDelay(ANCHOR_DELAY);

  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);

#ifdef CONFIG_ANCHOR_1
  // Anchor 1 (relay) starts calibration as TAG to measure other anchors
  Serial.println("Starting calibration mode as TAG...");
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  calibration_mode = false;
#else
  // Other anchors start as ANCHOR and wait for ranging requests
  Serial.println("Starting as ANCHOR, waiting for calibration...");
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  calibration_mode = false;
#endif

  Serial.println("Anchor ready!");
}

// ========================================
// Main Loop & Correction Logic
// ========================
void loop() {
  DW1000Ranging.loop();

#ifdef CONFIG_ANCHOR_1
  checkForCorrectionCommand();
#endif
}

void sendCorrection(uint8_t* target_mac, uint8_t device_number, float dx, float dy, float dz) {
  correction_t correction;
  correction.header = 'C';
  correction.device_number = device_number;
  correction.dx = dx;
  correction.dy = dy;
  correction.dz = dz;

  for (int i = 0; i < 2; i++) {
    if (memcmp(target_mac, glove_peers[i].mac, 6) == 0 && glove_peers[i].paired) {
      esp_now_send(glove_peers[i].mac, (uint8_t*)&correction, sizeof(correction_t));
      break;
    }
  }
}

void checkForCorrectionCommand() {
  if (Serial.available() >= 1 + 1 + 12) {
    if (Serial.read() == 'C') {
      int device_id = Serial.read();
      uint8_t *target_mac = nullptr;
      if (device_id == 1) {
        target_mac = glove1_mac;
      }
      else if (device_id == 2) {
        target_mac = glove2_mac;
      }

      float dx, dy, dz;
      Serial.readBytes((char*)&dx, sizeof(dx));
      Serial.readBytes((char*)&dy, sizeof(dy));
      Serial.readBytes((char*)&dz, sizeof(dz));

      sendCorrection(target_mac, device_id, dx, dy, dz);
    }
  }
}

// ========================================
// UWB Callbacks
// ========================================
void onNewRange() {
// #ifdef CONFIG_ANCHOR_1
//   if (calibration_mode) {
//     DW1000Device* device = DW1000Ranging.getDistantDevice();
//     float range = device->getRange();
//     int dest_anchor = 0;

//     // Identify which anchor we're measuring by short address
//     uint16_t short_addr = device->getShortAddress();
//     if (short_addr == 0x2222) dest_anchor = 2;
//     else if (short_addr == 0x3333) dest_anchor = 3;
//     else if (short_addr == 0x4444) dest_anchor = 4;
//     else return;  // Unknown anchor, ignore

//     // Accumulate samples for this specific anchor
//     calibration_data[dest_anchor].sum += range;
//     calibration_data[dest_anchor].count++;

//     // Check if we've collected enough samples for THIS anchor
//     if (calibration_data[dest_anchor].count == CALIBRATION_SAMPLES) {
//       // Calculate average distance
//       float avg_distance = calibration_data[dest_anchor].sum / CALIBRATION_SAMPLES;

//       // Send calibration packet to Python (only once per anchor)
//       calibration_packet_t cal_packet;
//       cal_packet.header = CALIBRATION_HEADER;
//       cal_packet.source_anchor = ANCHOR_ID;
//       cal_packet.dest_anchor = dest_anchor;
//       cal_packet.distance = avg_distance;

//       Serial.write((uint8_t*)&cal_packet, sizeof(calibration_packet_t));
//       Serial.printf("Calibration: Anchor %d -> Anchor %d: %.3f m\n",
//                     cal_packet.source_anchor, cal_packet.dest_anchor, avg_distance);

//       total_calibration_packets_sent++;

//       // Check if calibration is complete for all required anchors
//       int required_anchors = (NUM_UWB_ANCHORS == 4) ? 3 : 2;
//       if (total_calibration_packets_sent >= required_anchors) {
//         Serial.println("Calibration complete! Switching to ANCHOR mode...");
//         calibration_mode = false;

//         // Switch from TAG to ANCHOR mode
//         delay(100);
//         DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
//       }
//     }
//   }
// #endif
}

void onNewDevice(DW1000Device* device) {}
void onInactiveDevice(DW1000Device* device) {}
