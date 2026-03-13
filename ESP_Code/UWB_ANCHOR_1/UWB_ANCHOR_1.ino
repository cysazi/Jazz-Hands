#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>
#include <esp_now.h>
#include <WiFi.h>

// ========================================
// Makerfabs ESP32 UWB Pin Definitions
// ========================================
#define SPI_SCK  18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS    4
#define DW_RST   27
#define DW_IRQ   34

// ========================================
// ANCHOR ADDRESS - Change per device!
// ========================================
// Flash Anchor 1 with ANCHOR_1_ADDRESS
// Flash Anchor 2 with ANCHOR_2_ADDRESS
// Comment/uncomment accordingly:

// #define ANCHOR_ADDRESS "84:00:5B:D5:A9:9A:11:11"   // ← Anchor 1
#define ANCHOR_ADDRESS "85:00:5B:D5:A9:9A:22:22" // ← Anchor 2

// ========================================
// Known Physical Positions (meters)
// Update these to your actual setup!
// ========================================
// Incoming data packet from a glove
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

// Outgoing correction packet to a glove
typedef struct __attribute__((__packed__)) {
  char header; // 'C'
  float dx, dy, dz;
} correction_t;

// ========================================
// ESP-NOW Callbacks
// ========================================

// Called when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  // When a packet is received, write its raw bytes directly to the serial port
  // The Python script will be responsible for parsing this
  Serial.write(incomingData, len);
}

// Called when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // You can add debug prints here if needed
  // Serial.print("Last Packet Send Status: ");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

// ========================================
// Setup
// ========================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("UWB Anchor & ESP-NOW Relay Starting...");

  // --- Initialize ESP-NOW ---
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_recv_cb(OnDataRecv);
  esp_now_register_send_cb(OnDataSent);

  // --- Register Peers ---
  esp_now_peer_info_t peerInfo = {};
  peerInfo.channel = ESPNOW_WIFI_CHANNEL;
  peerInfo.encrypt = false;

  // Add glove 1 as a peer
  memcpy(peerInfo.peer_addr, glove1_mac, 6);
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer: Glove 1");
    return;
  }
  // Add other gloves here

  Serial.println("ESP-NOW Initialized.");

  // --- Initialize UWB ---
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY);

  Serial.println("Anchor ready!");
}

// ========================================
// Main Loop
// ========================================
void loop() {
  // Keep UWB ranging active
  DW1000Ranging.loop();

  // Check for correction commands from the Python script over serial
  checkForCorrectionCommand();
}

void sendCorrection(uint8_t* target_mac, float dx, float dy, float dz) {
  correction_t correction;
  correction.header = 'C';
  correction.dx = dx;
  correction.dy = dy;
  correction.dz = dz;
  esp_now_send(target_mac, (uint8_t *) &correction, sizeof(correction));
}

void checkForCorrectionCommand() {
  // Format from Python: 'C' <6-byte MAC> <3 floats for dx,dy,dz>
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
