#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>

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
// Anchor 1: (0, 0, 2.0)
// Anchor 2: (5, 0, 2.0)  ← 5 meters apart

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("============================");
  Serial.println("  DW1000 Anchor Starting...");
  Serial.println("============================");
  Serial.print("Address: ");
  Serial.println(ANCHOR_ADDRESS);

  // Initialize SPI
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);

  // Initialize DW1000
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);

  // Register callbacks
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);

  // Start as anchor
  // All anchors + tags must share the same network ID and channel!
  DW1000Ranging.startAsAnchor(
    ANCHOR_ADDRESS,
    DW1000.MODE_SHORTDATA_FAST_ACCURACY,  // Mode (range vs speed trade-off)
    false                                  // false = not using OTP calibration
  );

  Serial.println("Anchor ready! Waiting for tags...");
}

void loop() {
  DW1000Ranging.loop();
}

// ========================================
// Callbacks
// ========================================

// Called every time a new range is measured
void onNewRange() {

}

// Called when a new tag is detected
void onNewDevice(DW1000Device* device) {
  Serial.print("[NEW DEVICE] Tag connected: 0x");
  Serial.println(device->getShortAddress(), HEX);
}

// Called when a tag is no longer responding
void onInactiveDevice(DW1000Device* device) {
  Serial.print("[INACTIVE] Tag disconnected: 0x");
  Serial.println(device->getShortAddress(), HEX);
}