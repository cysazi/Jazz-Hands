#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>

// ========================================
// Makerfabs ESP32 UWB pin definitions
// ========================================
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4
#define DW_RST 27
#define DW_IRQ 34

// ========================================
// Survey / ESPNOW config
// ========================================
#define ESPNOW_WIFI_CHANNEL 6
#define PRINT_RANGE_DEBUG true
#define PRINT_RANGE_DEBUG_EVERY_MS 1000

#define SURVEY_CMD_MAGIC 0x5343
#define SURVEY_CMD_STEP 1
#define SURVEY_CMD_STOP 2
#define SURVEY_REPORT_MAGIC 0x5352
#define SURVEY_CMD_REPEAT_MS 300
#define SURVEY_REPORT_REPEAT_MS 60
#define SURVEY_STEP_COUNT 4

#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000
#define PACKET_HAS_UWB_3 0b00010000
#define PACKET_HAS_UWB_4 0b00100000

const uint8_t RECEIVER_RELAY_MAC[6] = {0x08, 0xF9, 0xE0, 0x92, 0xC0, 0x08};
const uint8_t ESPNOW_BROADCAST_MAC[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

typedef struct __attribute__((__packed__)) {
  uint16_t magic;
  uint8_t cmd;
  uint8_t session_id;
  uint8_t step;
  uint16_t step_ms;
} survey_cmd_t;  // 7 bytes

typedef struct __attribute__((__packed__)) {
  uint16_t magic;
  uint8_t anchor_id;
  uint8_t session_id;
  uint8_t step;
  uint8_t uwb_flags;
  float d1;
  float d2;
  float d3;
  float d4;
  uint32_t timestamp_us;
} survey_report_t;  // 26 bytes

// ========================================
// Anchor role selection
// Flash this same file to each stationary UWB anchor board,
// changing ANCHOR_ROLE to 1, 2, 3, or 4.
// ========================================
#define ANCHOR_ROLE 3

char ANCHOR_1_ADDRESS[] = "84:00:5B:D5:A9:9A:11:11";  // short 0x0084
char ANCHOR_2_ADDRESS[] = "85:00:5B:D5:A9:9A:22:22";  // short 0x0085
char ANCHOR_3_ADDRESS[] = "86:00:5B:D5:A9:9A:33:33";  // short 0x0086
char ANCHOR_4_ADDRESS[] = "87:00:5B:D5:A9:9A:44:44";  // short 0x0087

char TAG_1_ADDRESS[] = "91:00:22:EA:82:60:3B:91";  // short 0x0091
char TAG_2_ADDRESS[] = "92:00:22:EA:82:60:3B:92";  // short 0x0092
char TAG_3_ADDRESS[] = "93:00:22:EA:82:60:3B:93";  // short 0x0093
char TAG_4_ADDRESS[] = "94:00:22:EA:82:60:3B:94";  // short 0x0094

#if (ANCHOR_ROLE == 1)
char* ANCHOR_ADDRESS = ANCHOR_1_ADDRESS;
char* TAG_ADDRESS = TAG_1_ADDRESS;
const uint16_t EXPECTED_ANCHOR_SHORT = 0x0084;
const uint16_t EXPECTED_TAG_SHORT = 0x0091;
#elif (ANCHOR_ROLE == 2)
char* ANCHOR_ADDRESS = ANCHOR_2_ADDRESS;
char* TAG_ADDRESS = TAG_2_ADDRESS;
const uint16_t EXPECTED_ANCHOR_SHORT = 0x0085;
const uint16_t EXPECTED_TAG_SHORT = 0x0092;
#elif (ANCHOR_ROLE == 3)
char* ANCHOR_ADDRESS = ANCHOR_3_ADDRESS;
char* TAG_ADDRESS = TAG_3_ADDRESS;
const uint16_t EXPECTED_ANCHOR_SHORT = 0x0086;
const uint16_t EXPECTED_TAG_SHORT = 0x0093;
#elif (ANCHOR_ROLE == 4)
char* ANCHOR_ADDRESS = ANCHOR_4_ADDRESS;
char* TAG_ADDRESS = TAG_4_ADDRESS;
const uint16_t EXPECTED_ANCHOR_SHORT = 0x0087;
const uint16_t EXPECTED_TAG_SHORT = 0x0094;
#else
#error "ANCHOR_ROLE must be 1, 2, 3, or 4"
#endif

// ========================================
// Runtime state
// ========================================
volatile uint32_t range_count = 0;
volatile uint32_t new_count = 0;
volatile uint32_t inactive_count = 0;
volatile float last_range_m = 0.0f;
volatile uint16_t last_short = 0;
volatile bool have_range = false;

bool survey_active = false;
bool is_tag_role = false;
uint8_t survey_session_id = 0;
uint8_t survey_step = 0;
uint16_t survey_step_ms = 0;
uint32_t survey_step_started_ms = 0;
uint32_t survey_last_cmd_ms = 0;
uint32_t survey_last_report_ms = 0;
portMUX_TYPE surveyMux = portMUX_INITIALIZER_UNLOCKED;

volatile float survey_d1 = 0.0f;
volatile float survey_d2 = 0.0f;
volatile float survey_d3 = 0.0f;
volatile float survey_d4 = 0.0f;
volatile uint8_t survey_flags = 0;
volatile bool survey_dirty = false;

// Forward declarations
void onNewRange();
void onNewDevice(DW1000Device* device);
void onInactiveDevice(DW1000Device* device);

// ========================================
// ESP-NOW helpers
// ========================================
bool setWifiChannel(uint8_t channel) {
  if (esp_wifi_set_promiscuous(true) != ESP_OK) return false;
  esp_err_t ch = esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  return ch == ESP_OK;
}

bool ensurePeer(const uint8_t mac[6]) {
  if (esp_now_is_peer_exist(mac)) return true;
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, mac, 6);
  peer.channel = ESPNOW_WIFI_CHANNEL;
  peer.encrypt = false;
  return esp_now_add_peer(&peer) == ESP_OK;
}

void clearSurveyState() {
  survey_d1 = 0.0f;
  survey_d2 = 0.0f;
  survey_d3 = 0.0f;
  survey_d4 = 0.0f;
  survey_flags = 0;
  survey_dirty = false;
  survey_last_report_ms = 0;
}

void startUwbAnchorRole() {
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  is_tag_role = false;
}

void startUwbTagRole() {
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);
  DW1000Ranging.startAsTag(TAG_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  is_tag_role = true;
}

void applyRoleForSurveyStep(uint8_t step) {
  if (step == ANCHOR_ROLE) {
    startUwbTagRole();
  } else {
    startUwbAnchorRole();
  }
}

void sendSurveyReport() {
  uint8_t flags;
  float d1, d2, d3, d4;
  portENTER_CRITICAL(&surveyMux);
  flags = survey_flags;
  d1 = survey_d1;
  d2 = survey_d2;
  d3 = survey_d3;
  d4 = survey_d4;
  portEXIT_CRITICAL(&surveyMux);

  survey_report_t report = {};
  report.magic = SURVEY_REPORT_MAGIC;
  report.anchor_id = (uint8_t)ANCHOR_ROLE;
  report.session_id = survey_session_id;
  report.step = survey_step;
  report.uwb_flags = flags;
  report.d1 = d1;
  report.d2 = d2;
  report.d3 = d3;
  report.d4 = d4;
  report.timestamp_us = micros();

  ensurePeer(RECEIVER_RELAY_MAC);
  esp_now_send(RECEIVER_RELAY_MAC, (const uint8_t*)&report, sizeof(report));
}

void maybeEmitSurveyReport() {
  if (!survey_active || !is_tag_role || survey_step != ANCHOR_ROLE) return;
  uint32_t now_ms = millis();
  if ((uint32_t)(now_ms - survey_last_report_ms) < SURVEY_REPORT_REPEAT_MS) return;

  bool dirty;
  uint8_t flags;
  portENTER_CRITICAL(&surveyMux);
  dirty = survey_dirty;
  flags = survey_flags;
  survey_dirty = false;
  portEXIT_CRITICAL(&surveyMux);

  if (dirty && flags != 0) {
    sendSurveyReport();
    survey_last_report_ms = now_ms;
  }
}

// ========================================
// UWB callbacks
// ========================================
void onNewRange() {
  DW1000Device* device = DW1000Ranging.getDistantDevice();
  if (!device) return;

  range_count++;
  last_range_m = device->getRange();
  last_short = device->getShortAddress();
  have_range = true;

  if (!survey_active) return;

  uint16_t short_addr = last_short;
  float range_m = last_range_m;
  uint8_t flags = 0;

  portENTER_CRITICAL(&surveyMux);
  if (is_tag_role) {
    if (short_addr == 0x0084) {
      survey_d1 = range_m;
      flags |= PACKET_HAS_UWB_1;
    } else if (short_addr == 0x0085) {
      survey_d2 = range_m;
      flags |= PACKET_HAS_UWB_2;
    } else if (short_addr == 0x0086) {
      survey_d3 = range_m;
      flags |= PACKET_HAS_UWB_3;
    } else if (short_addr == 0x0087) {
      survey_d4 = range_m;
      flags |= PACKET_HAS_UWB_4;
    }
  } else {
    if (short_addr == 0x0091) {
      survey_d1 = range_m;
      flags |= PACKET_HAS_UWB_1;
    } else if (short_addr == 0x0092) {
      survey_d2 = range_m;
      flags |= PACKET_HAS_UWB_2;
    } else if (short_addr == 0x0093) {
      survey_d3 = range_m;
      flags |= PACKET_HAS_UWB_3;
    } else if (short_addr == 0x0094) {
      survey_d4 = range_m;
      flags |= PACKET_HAS_UWB_4;
    }
  }

  if (flags != 0) {
    survey_flags |= flags;
    survey_dirty = true;
  }
  portEXIT_CRITICAL(&surveyMux);
}

void onNewDevice(DW1000Device* device) {
  if (!device) return;
  new_count++;
}

void onInactiveDevice(DW1000Device* device) {
  if (!device) return;
  inactive_count++;
}

// ========================================
// Survey command handling
// ========================================
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
void onEspNowRecv(const esp_now_recv_info_t* info, const uint8_t* incomingData, int len) {
  (void)info;
#else
void onEspNowRecv(const uint8_t* mac, const uint8_t* incomingData, int len) {
  (void)mac;
#endif
  if (len != (int)sizeof(survey_cmd_t)) return;

  survey_cmd_t cmd = {};
  memcpy(&cmd, incomingData, sizeof(cmd));
  if (cmd.magic != SURVEY_CMD_MAGIC) return;

  if (cmd.cmd == SURVEY_CMD_STOP) {
    survey_active = false;
    survey_step = 0;
    clearSurveyState();
    startUwbAnchorRole();
    Serial.printf("[ANCHOR %d] survey stop session=%u\n", ANCHOR_ROLE, survey_session_id);
    return;
  }

  if (cmd.cmd != SURVEY_CMD_STEP) return;

  bool new_session = (!survey_active) || (cmd.session_id != survey_session_id);
  bool new_step = (!survey_active) || (cmd.step != survey_step);

  survey_session_id = cmd.session_id;
  survey_step_ms = cmd.step_ms;
  survey_active = true;
  survey_step_started_ms = millis();
  survey_last_cmd_ms = survey_step_started_ms;

  if (new_session || new_step) {
    survey_step = cmd.step;
    clearSurveyState();
    applyRoleForSurveyStep(survey_step);
    Serial.printf("[ANCHOR %d] survey step=%u/%u role=%s session=%u\n",
                  ANCHOR_ROLE,
                  survey_step,
                  SURVEY_STEP_COUNT,
                  is_tag_role ? "tag" : "anchor",
                  survey_session_id);
  }
}

void serviceSurveyStepTimeout() {
  if (!survey_active) return;
  uint32_t now_ms = millis();
  if ((uint32_t)(now_ms - survey_last_cmd_ms) > (uint32_t)(SURVEY_CMD_REPEAT_MS * 4)) {
    // If the coordinator disappears, fall back to normal anchor mode.
    survey_active = false;
    survey_step = 0;
    clearSurveyState();
    startUwbAnchorRole();
    Serial.printf("[ANCHOR %d] survey timeout -> anchor mode\n", ANCHOR_ROLE);
  }
}

void printSurveyReportIfDue() {
  if (!PRINT_RANGE_DEBUG) return;
  if (!survey_active || !is_tag_role || survey_step != ANCHOR_ROLE) return;

  uint32_t now_ms = millis();
  static uint32_t last_print_ms = 0;
  if ((uint32_t)(now_ms - last_print_ms) < PRINT_RANGE_DEBUG_EVERY_MS) return;
  last_print_ms = now_ms;

  Serial.printf(
      "[ANCHOR %d] survey report step=%u flags=0b%08b d=[%.3f %.3f %.3f %.3f] ranges=%lu new=%lu inactive=%lu\n",
      ANCHOR_ROLE,
      survey_step,
      survey_flags,
      survey_d1,
      survey_d2,
      survey_d3,
      survey_d4,
      range_count,
      new_count,
      inactive_count);
}

void printRangeDebugIfDue() {
  if (!PRINT_RANGE_DEBUG) return;
  static uint32_t last_print_ms = 0;
  uint32_t now_ms = millis();
  if ((uint32_t)(now_ms - last_print_ms) < PRINT_RANGE_DEBUG_EVERY_MS) return;
  last_print_ms = now_ms;

  if (have_range) {
    Serial.printf(
        "[ANCHOR %d] role=%s expected_short=0x%04X last_short=0x%04X last=%.3f m ranges=%lu new=%lu inactive=%lu survey=%s step=%u/%u\n",
        ANCHOR_ROLE,
        is_tag_role ? "tag" : "anchor",
        is_tag_role ? EXPECTED_TAG_SHORT : EXPECTED_ANCHOR_SHORT,
        last_short,
        last_range_m,
        range_count,
        new_count,
        inactive_count,
        survey_active ? "yes" : "no",
        survey_step,
        SURVEY_STEP_COUNT);
  } else {
    Serial.printf(
        "[ANCHOR %d] role=%s expected_short=0x%04X waiting... ranges=%lu new=%lu inactive=%lu survey=%s step=%u/%u\n",
        ANCHOR_ROLE,
        is_tag_role ? "tag" : "anchor",
        is_tag_role ? EXPECTED_TAG_SHORT : EXPECTED_ANCHOR_SHORT,
        range_count,
        new_count,
        inactive_count,
        survey_active ? "yes" : "no",
        survey_step,
        SURVEY_STEP_COUNT);
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("=== JazzHands UWB Generic Anchor Boot ===");
  Serial.printf("Role: %d\n", ANCHOR_ROLE);
  Serial.printf("Anchor short: 0x%04X\n", EXPECTED_ANCHOR_SHORT);
  Serial.printf("Tag short: 0x%04X\n", EXPECTED_TAG_SHORT);
  Serial.print("Anchor address: ");
  Serial.println(ANCHOR_ADDRESS);
  Serial.print("Tag address: ");
  Serial.println(TAG_ADDRESS);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.disconnect(false, true);
  if (!setWifiChannel(ESPNOW_WIFI_CHANNEL)) {
    Serial.println("Failed to set WiFi channel for ESP-NOW");
    while (true) delay(1000);
  }
  if (esp_now_init() != ESP_OK) {
    Serial.println("Failed to initialize ESP-NOW");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onEspNowRecv);
  ensurePeer(RECEIVER_RELAY_MAC);
  ensurePeer(ESPNOW_BROADCAST_MAC);

  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(DW_RST, DW_CS, DW_IRQ);
  DW1000Ranging.attachNewRange(onNewRange);
  DW1000Ranging.attachBlinkDevice(onNewDevice);
  DW1000Ranging.attachInactiveDevice(onInactiveDevice);
  DW1000Ranging.startAsAnchor(ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);

  Serial.println("Anchor ready.");
}

void loop() {
  DW1000Ranging.loop();
  maybeEmitSurveyReport();
  serviceSurveyStepTimeout();
  printRangeDebugIfDue();
  printSurveyReportIfDue();
}
