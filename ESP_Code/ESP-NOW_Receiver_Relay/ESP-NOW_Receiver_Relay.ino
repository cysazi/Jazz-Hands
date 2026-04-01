#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

#include <SPI.h>
#include <DW1000Ranging.h>
#include <DW1000.h>

// ----------------------------- Config -----------------------------
#define SERIAL_BAUD 921600
#define ESPNOW_WIFI_CHANNEL 6
#define PACKET_HEADER 0xAAAA

// false = binary relay mode for Python (recommended)
// true  = text debug mode in Arduino Serial Monitor
#define DEBUG_SERIAL_MONITOR_MODE false

#define PRINT_STATS_EVERY_MS 1000
#define RELAY_FIFO_CAPACITY 64

// ----------------------------- UWB config -----------------------------
#define ENABLE_RECEIVER_UWB_ANCHOR 1
#if ENABLE_RECEIVER_UWB_ANCHOR
#define UWB_SPI_SCK 18
#define UWB_SPI_MISO 19
#define UWB_SPI_MOSI 23
#define UWB_DW_CS 4
#define UWB_DW_RST 27
#define UWB_DW_IRQ 34

// Receiver board acts as anchor #4 in normal runtime.
char RECEIVER_UWB_ANCHOR_ADDRESS[] = "87:00:5B:D5:A9:9A:44:44";  // short 0x0087
// During survey step 4 receiver temporarily acts as tag #4.
char RECEIVER_UWB_TAG_ADDRESS[] = "94:00:22:EA:82:60:3B:C4";     // short 0x0094

const uint16_t UWB_SHORT_ANCHOR_1 = 0x0084;
const uint16_t UWB_SHORT_ANCHOR_2 = 0x0085;
const uint16_t UWB_SHORT_ANCHOR_3 = 0x0086;
const uint16_t UWB_SHORT_ANCHOR_4 = 0x0087;
const uint16_t UWB_SHORT_TAG_1 = 0x0091;
const uint16_t UWB_SHORT_TAG_2 = 0x0092;
const uint16_t UWB_SHORT_TAG_3 = 0x0093;
const uint16_t UWB_SHORT_TAG_4 = 0x0094;
#endif

// ----------------------------- Packet flags -----------------------------
#define PACKET_HAS_ACCEL 0b00000001
#define PACKET_HAS_QUAT  0b00000010
#define PACKET_HAS_UWB_1 0b00000100
#define PACKET_HAS_UWB_2 0b00001000
#define PACKET_HAS_UWB_3 0b00010000
#define PACKET_HAS_UWB_4 0b00100000
#define PACKET_FLAG_ANCHOR_SURVEY 0b01000000

// ----------------------------- Survey protocol -----------------------------
#define HOST_CMD_MAGIC 0x4843        // "HC"
#define HOST_CMD_START_SURVEY 1
#define HOST_CMD_STOP_SURVEY 2

#define SURVEY_CMD_MAGIC 0x5343       // "SC"
#define SURVEY_CMD_STEP 1
#define SURVEY_CMD_STOP 2

#define SURVEY_REPORT_MAGIC 0x5352    // "SR"

#define SURVEY_DEFAULT_STEP_MS 3500
#define SURVEY_MIN_STEP_MS 1200
#define SURVEY_MAX_STEP_MS 12000
#define SURVEY_STEP_COUNT 4
#define SURVEY_CMD_REPEAT_MS 300
#define SURVEY_LOCAL_REPORT_MS 60

const uint8_t ESPNOW_BROADCAST_MAC[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

typedef struct __attribute__((__packed__)) {
  uint16_t magic;
  uint8_t cmd;
  uint8_t session_id;
  uint16_t step_ms;
  uint16_t reserved;
} host_cmd_t;  // 8 bytes

typedef struct __attribute__((__packed__)) {
  uint16_t magic;
  uint8_t cmd;
  uint8_t session_id;
  uint8_t step;
  uint16_t step_ms;
} survey_cmd_t;  // 7 bytes

typedef struct __attribute__((__packed__)) {
  uint16_t magic;
  uint8_t anchor_id;   // 1..4
  uint8_t session_id;
  uint8_t step;        // 1..4
  uint8_t uwb_flags;   // PACKET_HAS_UWB_1/2/3/4 bits
  float d1;
  float d2;
  float d3;
  float d4;
  uint32_t timestamp_us;
} survey_report_t;  // 26 bytes

// ----------------------------- Relay packet types -----------------------------
typedef struct __attribute__((__packed__)) {
  uint16_t header;
  uint8_t device_id;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float accel_x, accel_y, accel_z;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2, UWB_distance3, UWB_distance4;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} datapacket_t;  // 78 bytes

typedef struct __attribute__((__packed__)) {
  uint16_t header;
  uint8_t device_id;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float accel_x, accel_y, accel_z;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2, UWB_distance3;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} legacy_datapacket_74_t;  // 74 bytes

typedef struct __attribute__((__packed__)) {
  uint16_t header;
  uint8_t device_id;
  uint32_t timestamp;
  uint8_t packet_type;
  uint8_t button_state;
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float UWB_distance1, UWB_distance2, UWB_distance3;
  float quat_w, quat_i, quat_j, quat_k;
  uint8_t error_handler;
} legacy_datapacket_62_t;  // 62 bytes

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
} legacy_datapacket_58_t;  // 58 bytes

typedef struct {
  uint8_t len;
  uint8_t data[sizeof(datapacket_t)];
} relay_frame_t;

// ----------------------------- Runtime state -----------------------------
volatile uint32_t rx_total = 0;
volatile uint32_t rx_valid = 0;
volatile uint32_t rx_bad_size = 0;
volatile uint32_t rx_bad_header = 0;
volatile uint32_t rx_dev1 = 0;
volatile uint32_t rx_dev2 = 0;
volatile uint32_t rx_survey_reports = 0;
volatile uint32_t rx_host_commands = 0;
volatile uint32_t tx_total = 0;
volatile uint32_t tx_ok = 0;
volatile uint32_t tx_fail = 0;
volatile uint32_t relay_fifo_drop = 0;

relay_frame_t relay_fifo[RELAY_FIFO_CAPACITY];
volatile uint16_t relay_fifo_head = 0;
volatile uint16_t relay_fifo_tail = 0;
portMUX_TYPE relayFifoMux = portMUX_INITIALIZER_UNLOCKED;
portMUX_TYPE surveyMux = portMUX_INITIALIZER_UNLOCKED;

volatile bool latest_packet_valid = false;
datapacket_t latest_packet = {};
uint32_t latest_packet_ms = 0;

const bool RELAY_BINARY_MODE = !DEBUG_SERIAL_MONITOR_MODE;

// ----------------------------- Survey orchestration state -----------------------------
bool survey_active = false;
uint8_t survey_session_id = 0;
uint8_t survey_step = 0;
uint16_t survey_step_ms = SURVEY_DEFAULT_STEP_MS;
uint32_t survey_step_started_ms = 0;
uint32_t survey_last_cmd_broadcast_ms = 0;

#if ENABLE_RECEIVER_UWB_ANCHOR
enum ReceiverUwbRole : uint8_t {
  RECEIVER_UWB_ROLE_ANCHOR = 0,
  RECEIVER_UWB_ROLE_TAG = 1,
};
ReceiverUwbRole receiver_uwb_role = RECEIVER_UWB_ROLE_ANCHOR;

volatile uint32_t uwb_range_count = 0;
volatile uint32_t uwb_new_count = 0;
volatile uint32_t uwb_inactive_count = 0;
volatile float uwb_last_range_m = 0.0f;
volatile uint16_t uwb_last_short = 0;
volatile bool uwb_have_range = false;

volatile float survey_local_d1 = 0.0f;
volatile float survey_local_d2 = 0.0f;
volatile float survey_local_d3 = 0.0f;
volatile float survey_local_d4 = 0.0f;
volatile uint8_t survey_local_flags = 0;
volatile bool survey_local_dirty = false;
volatile uint32_t survey_local_last_update_ms = 0;
#endif

// ----------------------------- Helpers -----------------------------
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

bool enqueueRelayFrame(const uint8_t* data, int len) {
  if (len <= 0 || len > (int)sizeof(datapacket_t)) return false;

  bool ok = false;
  portENTER_CRITICAL(&relayFifoMux);
  uint16_t next = (uint16_t)((relay_fifo_head + 1) % RELAY_FIFO_CAPACITY);
  if (next != relay_fifo_tail) {
    relay_fifo[relay_fifo_head].len = (uint8_t)len;
    memcpy(relay_fifo[relay_fifo_head].data, data, (size_t)len);
    relay_fifo_head = next;
    ok = true;
  } else {
    relay_fifo_drop++;
  }
  portEXIT_CRITICAL(&relayFifoMux);
  return ok;
}

void flushRelayFifoToSerial() {
  while (true) {
    relay_frame_t frame = {};
    bool has_frame = false;
    portENTER_CRITICAL(&relayFifoMux);
    if (relay_fifo_tail != relay_fifo_head) {
      frame = relay_fifo[relay_fifo_tail];
      relay_fifo_tail = (uint16_t)((relay_fifo_tail + 1) % RELAY_FIFO_CAPACITY);
      has_frame = true;
    }
    portEXIT_CRITICAL(&relayFifoMux);
    if (!has_frame) break;
    Serial.write(frame.data, frame.len);
  }
}

void fillSurveyDatapacket(datapacket_t* out, uint8_t anchor_id, uint32_t ts_us, uint8_t uwb_flags, float d1, float d2, float d3, float d4) {
  memset(out, 0, sizeof(datapacket_t));
  out->header = PACKET_HEADER;
  out->device_id = (uint8_t)(100 + anchor_id);  // 101..104
  out->timestamp = ts_us;
  out->packet_type = (uint8_t)(PACKET_FLAG_ANCHOR_SURVEY | (uwb_flags & (PACKET_HAS_UWB_1 | PACKET_HAS_UWB_2 | PACKET_HAS_UWB_3 | PACKET_HAS_UWB_4)));
  out->button_state = 1;
  out->UWB_distance1 = d1;
  out->UWB_distance2 = d2;
  out->UWB_distance3 = d3;
  out->UWB_distance4 = d4;
  out->quat_w = 1.0f;
  out->quat_i = 0.0f;
  out->quat_j = 0.0f;
  out->quat_k = 0.0f;
}

void emitSurveyPacketToUsb(uint8_t anchor_id, uint8_t uwb_flags, float d1, float d2, float d3, float d4, uint32_t ts_us) {
  datapacket_t pkt = {};
  fillSurveyDatapacket(&pkt, anchor_id, ts_us, uwb_flags, d1, d2, d3, d4);
  enqueueRelayFrame((const uint8_t*)&pkt, sizeof(pkt));
  latest_packet = pkt;
  latest_packet_ms = millis();
  latest_packet_valid = true;
}

uint16_t getSurveyTagShortForStep(uint8_t step) {
  if (step == 1) return UWB_SHORT_TAG_1;
  if (step == 2) return UWB_SHORT_TAG_2;
  if (step == 3) return UWB_SHORT_TAG_3;
  if (step == 4) return UWB_SHORT_TAG_4;
  return 0;
}

bool receiverIsTagForStep(uint8_t step) {
  return step == 4;
}

void broadcastSurveyCommand(uint8_t cmd, uint8_t session_id, uint8_t step, uint16_t step_ms) {
  survey_cmd_t out = {};
  out.magic = SURVEY_CMD_MAGIC;
  out.cmd = cmd;
  out.session_id = session_id;
  out.step = step;
  out.step_ms = step_ms;
  ensurePeer(ESPNOW_BROADCAST_MAC);
  esp_now_send(ESPNOW_BROADCAST_MAC, (const uint8_t*)&out, sizeof(out));
}

#if ENABLE_RECEIVER_UWB_ANCHOR
void onUwbNewRange() {
  DW1000Device* device = DW1000Ranging.getDistantDevice();
  if (!device) return;

  uwb_range_count++;
  uwb_last_range_m = device->getRange();
  uwb_last_short = device->getShortAddress();
  uwb_have_range = true;

  if (!survey_active) return;

  uint16_t short_addr = uwb_last_short;
  float range_m = uwb_last_range_m;
  uint8_t flags = 0;

  portENTER_CRITICAL(&surveyMux);
  if (receiver_uwb_role == RECEIVER_UWB_ROLE_ANCHOR) {
    if (short_addr == UWB_SHORT_TAG_1) {
      survey_local_d1 = range_m;
      flags |= PACKET_HAS_UWB_1;
    } else if (short_addr == UWB_SHORT_TAG_2) {
      survey_local_d2 = range_m;
      flags |= PACKET_HAS_UWB_2;
    } else if (short_addr == UWB_SHORT_TAG_3) {
      survey_local_d3 = range_m;
      flags |= PACKET_HAS_UWB_3;
    } else if (short_addr == UWB_SHORT_TAG_4) {
      survey_local_d4 = range_m;
      flags |= PACKET_HAS_UWB_4;
    }
  } else {
    if (short_addr == UWB_SHORT_ANCHOR_1) {
      survey_local_d1 = range_m;
      flags |= PACKET_HAS_UWB_1;
    } else if (short_addr == UWB_SHORT_ANCHOR_2) {
      survey_local_d2 = range_m;
      flags |= PACKET_HAS_UWB_2;
    } else if (short_addr == UWB_SHORT_ANCHOR_3) {
      survey_local_d3 = range_m;
      flags |= PACKET_HAS_UWB_3;
    } else if (short_addr == UWB_SHORT_ANCHOR_4) {
      survey_local_d4 = range_m;
      flags |= PACKET_HAS_UWB_4;
    }
  }

  if (flags != 0) {
    survey_local_flags |= flags;
    survey_local_dirty = true;
    survey_local_last_update_ms = millis();
  }
  portEXIT_CRITICAL(&surveyMux);
}

void onUwbNewDevice(DW1000Device* device) {
  if (!device) return;
  uwb_new_count++;
}

void onUwbInactiveDevice(DW1000Device* device) {
  if (!device) return;
  uwb_inactive_count++;
}

void startReceiverAsAnchorRole() {
  SPI.begin(UWB_SPI_SCK, UWB_SPI_MISO, UWB_SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_DW_RST, UWB_DW_CS, UWB_DW_IRQ);
  DW1000Ranging.attachNewRange(onUwbNewRange);
  DW1000Ranging.attachNewDevice(onUwbNewDevice);
  DW1000Ranging.attachInactiveDevice(onUwbInactiveDevice);
  DW1000Ranging.startAsAnchor(RECEIVER_UWB_ANCHOR_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  receiver_uwb_role = RECEIVER_UWB_ROLE_ANCHOR;
}

void startReceiverAsTagRole() {
  SPI.begin(UWB_SPI_SCK, UWB_SPI_MISO, UWB_SPI_MOSI);
  DW1000Ranging.initCommunication(UWB_DW_RST, UWB_DW_CS, UWB_DW_IRQ);
  DW1000Ranging.attachNewRange(onUwbNewRange);
  DW1000Ranging.attachNewDevice(onUwbNewDevice);
  DW1000Ranging.attachInactiveDevice(onUwbInactiveDevice);
  DW1000Ranging.startAsTag(RECEIVER_UWB_TAG_ADDRESS, DW1000.MODE_SHORTDATA_FAST_ACCURACY, false);
  receiver_uwb_role = RECEIVER_UWB_ROLE_TAG;
}

void applyReceiverRoleForSurveyStep(uint8_t step) {
  if (receiverIsTagForStep(step)) {
    startReceiverAsTagRole();
  } else {
    startReceiverAsAnchorRole();
  }
}

void clearLocalSurveyState() {
  portENTER_CRITICAL(&surveyMux);
  survey_local_d1 = 0.0f;
  survey_local_d2 = 0.0f;
  survey_local_d3 = 0.0f;
  survey_local_d4 = 0.0f;
  survey_local_flags = 0;
  survey_local_dirty = false;
  survey_local_last_update_ms = 0;
  portEXIT_CRITICAL(&surveyMux);
}

void emitLocalSurveyReportIfDue() {
  static uint32_t last_emit_ms = 0;
  uint32_t now_ms = millis();
  if (!survey_active || receiver_uwb_role != RECEIVER_UWB_ROLE_TAG || survey_step != 4) return;
  if ((uint32_t)(now_ms - last_emit_ms) < SURVEY_LOCAL_REPORT_MS) return;
  last_emit_ms = now_ms;

  uint8_t flags = 0;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  float d4 = 0.0f;
  bool dirty = false;

  portENTER_CRITICAL(&surveyMux);
  flags = survey_local_flags;
  d1 = survey_local_d1;
  d2 = survey_local_d2;
  d3 = survey_local_d3;
  d4 = survey_local_d4;
  dirty = survey_local_dirty;
  survey_local_dirty = false;
  portEXIT_CRITICAL(&surveyMux);

  if (dirty && flags != 0) {
    emitSurveyPacketToUsb(4, flags, d1, d2, d3, d4, micros());
  }
}
#endif

void beginSurvey(uint8_t session_id, uint16_t requested_step_ms) {
#if ENABLE_RECEIVER_UWB_ANCHOR
  survey_session_id = session_id;
  survey_step_ms = (uint16_t)constrain((int)requested_step_ms, SURVEY_MIN_STEP_MS, SURVEY_MAX_STEP_MS);
  survey_step = 1;
  survey_step_started_ms = millis();
  survey_last_cmd_broadcast_ms = 0;
  survey_active = true;
  clearLocalSurveyState();
  applyReceiverRoleForSurveyStep(survey_step);
  broadcastSurveyCommand(SURVEY_CMD_STEP, survey_session_id, survey_step, survey_step_ms);
  Serial.printf("[SURVEY] start session=%u step=%u/%u step_ms=%u role=%s\n",
                survey_session_id,
                survey_step,
                SURVEY_STEP_COUNT,
                survey_step_ms,
                receiver_uwb_role == RECEIVER_UWB_ROLE_TAG ? "tag" : "anchor");
#else
  (void)session_id;
  (void)requested_step_ms;
#endif
}

void stopSurvey(bool notify_anchors) {
#if ENABLE_RECEIVER_UWB_ANCHOR
  if (notify_anchors) {
    broadcastSurveyCommand(SURVEY_CMD_STOP, survey_session_id, 0, 0);
  }
  survey_active = false;
  survey_step = 0;
  clearLocalSurveyState();
  startReceiverAsAnchorRole();
  Serial.printf("[SURVEY] stop session=%u notify=%s\n", survey_session_id, notify_anchors ? "yes" : "no");
#else
  (void)notify_anchors;
#endif
}

void serviceSurveyScheduler() {
#if ENABLE_RECEIVER_UWB_ANCHOR
  if (!survey_active) return;
  uint32_t now_ms = millis();

  if ((uint32_t)(now_ms - survey_last_cmd_broadcast_ms) >= SURVEY_CMD_REPEAT_MS) {
    survey_last_cmd_broadcast_ms = now_ms;
    broadcastSurveyCommand(SURVEY_CMD_STEP, survey_session_id, survey_step, survey_step_ms);
  }

  if ((uint32_t)(now_ms - survey_step_started_ms) >= survey_step_ms) {
    if (survey_step < SURVEY_STEP_COUNT) {
      survey_step++;
      survey_step_started_ms = now_ms;
      clearLocalSurveyState();
      applyReceiverRoleForSurveyStep(survey_step);
      broadcastSurveyCommand(SURVEY_CMD_STEP, survey_session_id, survey_step, survey_step_ms);
      Serial.printf("[SURVEY] step=%u/%u role=%s\n",
                    survey_step,
                    SURVEY_STEP_COUNT,
                    receiver_uwb_role == RECEIVER_UWB_ROLE_TAG ? "tag" : "anchor");
    } else {
      stopSurvey(true);
    }
  }
#endif
}

void handleHostCommands() {
  while (Serial.available() >= (int)sizeof(host_cmd_t)) {
    host_cmd_t cmd = {};
    if (Serial.readBytes((char*)&cmd, sizeof(cmd)) != sizeof(cmd)) return;
    if (cmd.magic != HOST_CMD_MAGIC) continue;
    rx_host_commands++;

    if (cmd.cmd == HOST_CMD_START_SURVEY) {
      beginSurvey(cmd.session_id, cmd.step_ms == 0 ? SURVEY_DEFAULT_STEP_MS : cmd.step_ms);
    } else if (cmd.cmd == HOST_CMD_STOP_SURVEY) {
      if (!survey_active || cmd.session_id == survey_session_id) {
        stopSurvey(true);
      }
    }
  }
}

void handleIncomingSurveyReport(const uint8_t* incomingData, int len) {
  if (len != (int)sizeof(survey_report_t)) return;
  survey_report_t report = {};
  memcpy(&report, incomingData, sizeof(report));
  if (report.magic != SURVEY_REPORT_MAGIC) return;
  if (report.anchor_id < 1 || report.anchor_id > 4) return;

  rx_survey_reports++;
  emitSurveyPacketToUsb(
      report.anchor_id,
      report.uwb_flags,
      report.d1,
      report.d2,
      report.d3,
      report.d4,
      report.timestamp_us);
}

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
void OnDataRecv(const esp_now_recv_info_t* info, const uint8_t* incomingData, int len) {
  (void)info;
#else
void OnDataRecv(const uint8_t* mac, const uint8_t* incomingData, int len) {
  (void)mac;
#endif
  rx_total++;

  if (len == (int)sizeof(survey_report_t)) {
    handleIncomingSurveyReport(incomingData, len);
    return;
  }

  bool is_new_packet = (len == (int)sizeof(datapacket_t));
  bool is_legacy_74 = (len == (int)sizeof(legacy_datapacket_74_t));
  bool is_legacy_62 = (len == (int)sizeof(legacy_datapacket_62_t));
  bool is_legacy_58 = (len == (int)sizeof(legacy_datapacket_58_t));
  if (!is_new_packet && !is_legacy_74 && !is_legacy_62 && !is_legacy_58) {
    rx_bad_size++;
    return;
  }

  datapacket_t packet = {};
  if (is_new_packet) {
    memcpy(&packet, incomingData, sizeof(packet));
  } else if (is_legacy_74) {
    legacy_datapacket_74_t legacy = {};
    memcpy(&legacy, incomingData, sizeof(legacy));
    packet.header = legacy.header;
    packet.device_id = legacy.device_id;
    packet.timestamp = legacy.timestamp;
    packet.packet_type = legacy.packet_type;
    packet.button_state = legacy.button_state;
    packet.accel_x = legacy.accel_x;
    packet.accel_y = legacy.accel_y;
    packet.accel_z = legacy.accel_z;
    packet.pos_x = legacy.pos_x;
    packet.pos_y = legacy.pos_y;
    packet.pos_z = legacy.pos_z;
    packet.vel_x = legacy.vel_x;
    packet.vel_y = legacy.vel_y;
    packet.vel_z = legacy.vel_z;
    packet.UWB_distance1 = legacy.UWB_distance1;
    packet.UWB_distance2 = legacy.UWB_distance2;
    packet.UWB_distance3 = legacy.UWB_distance3;
    packet.UWB_distance4 = 0.0f;
    packet.quat_w = legacy.quat_w;
    packet.quat_i = legacy.quat_i;
    packet.quat_j = legacy.quat_j;
    packet.quat_k = legacy.quat_k;
    packet.error_handler = legacy.error_handler;
  } else if (is_legacy_62) {
    legacy_datapacket_62_t legacy = {};
    memcpy(&legacy, incomingData, sizeof(legacy));
    packet.header = legacy.header;
    packet.device_id = legacy.device_id;
    packet.timestamp = legacy.timestamp;
    packet.packet_type = legacy.packet_type;
    packet.button_state = legacy.button_state;
    packet.accel_x = 0.0f;
    packet.accel_y = 0.0f;
    packet.accel_z = 0.0f;
    packet.pos_x = legacy.pos_x;
    packet.pos_y = legacy.pos_y;
    packet.pos_z = legacy.pos_z;
    packet.vel_x = legacy.vel_x;
    packet.vel_y = legacy.vel_y;
    packet.vel_z = legacy.vel_z;
    packet.UWB_distance1 = legacy.UWB_distance1;
    packet.UWB_distance2 = legacy.UWB_distance2;
    packet.UWB_distance3 = legacy.UWB_distance3;
    packet.UWB_distance4 = 0.0f;
    packet.quat_w = legacy.quat_w;
    packet.quat_i = legacy.quat_i;
    packet.quat_j = legacy.quat_j;
    packet.quat_k = legacy.quat_k;
    packet.error_handler = legacy.error_handler;
  } else {
    legacy_datapacket_58_t legacy = {};
    memcpy(&legacy, incomingData, sizeof(legacy));
    packet.header = legacy.header;
    packet.device_id = legacy.device_id;
    packet.timestamp = legacy.timestamp;
    packet.packet_type = legacy.packet_type;
    packet.button_state = legacy.button_state;
    packet.accel_x = 0.0f;
    packet.accel_y = 0.0f;
    packet.accel_z = 0.0f;
    packet.pos_x = legacy.pos_x;
    packet.pos_y = legacy.pos_y;
    packet.pos_z = legacy.pos_z;
    packet.vel_x = legacy.vel_x;
    packet.vel_y = legacy.vel_y;
    packet.vel_z = legacy.vel_z;
    packet.UWB_distance1 = legacy.UWB_distance1;
    packet.UWB_distance2 = legacy.UWB_distance2;
    packet.UWB_distance3 = 0.0f;
    packet.UWB_distance4 = 0.0f;
    packet.quat_w = legacy.quat_w;
    packet.quat_i = legacy.quat_i;
    packet.quat_j = legacy.quat_j;
    packet.quat_k = legacy.quat_k;
    packet.error_handler = legacy.error_handler;
  }

  if (packet.header != PACKET_HEADER) {
    rx_bad_header++;
    return;
  }

  rx_valid++;
  if (packet.device_id == 1) rx_dev1++;
  if (packet.device_id == 2) rx_dev2++;

  latest_packet = packet;
  latest_packet_ms = millis();
  latest_packet_valid = true;

  if (RELAY_BINARY_MODE) {
    if (is_new_packet) {
      enqueueRelayFrame(incomingData, len);
    } else {
      enqueueRelayFrame((const uint8_t*)&packet, sizeof(datapacket_t));
    }
  }
}

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
void OnDataSent(const wifi_tx_info_t* tx_info, esp_now_send_status_t status) {
  (void)tx_info;
#else
void OnDataSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
  (void)mac_addr;
#endif
  tx_total++;
  if (status == ESP_NOW_SEND_SUCCESS) {
    tx_ok++;
  } else {
    tx_fail++;
  }
}

void printStatsIfDue() {
  if (RELAY_BINARY_MODE) return;

  static uint32_t last_print = 0;
  uint32_t now = millis();
  if ((uint32_t)(now - last_print) < PRINT_STATS_EVERY_MS) return;
  last_print = now;

  uint32_t fifo_depth = (uint16_t)((relay_fifo_head + RELAY_FIFO_CAPACITY - relay_fifo_tail) % RELAY_FIFO_CAPACITY);
  Serial.printf(
      "[RX] total=%lu valid=%lu bad_size=%lu bad_header=%lu dev1=%lu dev2=%lu survey=%lu host_cmd=%lu fifo_drop=%lu fifo_depth=%lu | "
      "[TX] total=%lu ok=%lu fail=%lu\n",
      rx_total,
      rx_valid,
      rx_bad_size,
      rx_bad_header,
      rx_dev1,
      rx_dev2,
      rx_survey_reports,
      rx_host_commands,
      relay_fifo_drop,
      fifo_depth,
      tx_total,
      tx_ok,
      tx_fail);

  Serial.printf(
      "[SURVEY] active=%s session=%u step=%u/%u step_ms=%u role=%s\n",
      survey_active ? "yes" : "no",
      survey_session_id,
      survey_step,
      SURVEY_STEP_COUNT,
      survey_step_ms,
      receiver_uwb_role == RECEIVER_UWB_ROLE_TAG ? "tag" : "anchor");

  if (latest_packet_valid) {
    uint32_t age = now - latest_packet_ms;
    Serial.printf(
        "[PKT] dev=%u type=0x%02X uwb=[%.3f %.3f %.3f %.3f] age=%lums\n",
        latest_packet.device_id,
        latest_packet.packet_type,
        latest_packet.UWB_distance1,
        latest_packet.UWB_distance2,
        latest_packet.UWB_distance3,
        latest_packet.UWB_distance4,
        age);
  } else {
    Serial.println("[PKT] waiting for valid packet...");
  }

#if ENABLE_RECEIVER_UWB_ANCHOR
  if (uwb_have_range) {
    Serial.printf(
        "[UWB] role=%s last_short=0x%04X last=%.3fm ranges=%lu new=%lu inactive=%lu\n",
        receiver_uwb_role == RECEIVER_UWB_ROLE_TAG ? "tag" : "anchor",
        uwb_last_short,
        uwb_last_range_m,
        uwb_range_count,
        uwb_new_count,
        uwb_inactive_count);
  } else {
    Serial.printf(
        "[UWB] role=%s waiting... ranges=%lu new=%lu inactive=%lu\n",
        receiver_uwb_role == RECEIVER_UWB_ROLE_TAG ? "tag" : "anchor",
        uwb_range_count,
        uwb_new_count,
        uwb_inactive_count);
  }
#endif
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(15);
  delay(250);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.disconnect(false, true);

  if (!setWifiChannel(ESPNOW_WIFI_CHANNEL)) {
    while (true) delay(1000);
  }
  if (esp_now_init() != ESP_OK) {
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(OnDataRecv);
  esp_now_register_send_cb(OnDataSent);
  setWifiChannel(ESPNOW_WIFI_CHANNEL);
  ensurePeer(ESPNOW_BROADCAST_MAC);

#if ENABLE_RECEIVER_UWB_ANCHOR
  startReceiverAsAnchorRole();
#endif

  if (!RELAY_BINARY_MODE) {
    Serial.println("=== JazzHands Receiver Relay Boot ===");
    Serial.printf("Relay mode: %s\n", RELAY_BINARY_MODE ? "binary-to-USB" : "debug-text");
    Serial.printf("Expected packet bytes: %u\n", (unsigned)sizeof(datapacket_t));
    Serial.printf("WiFi channel: %u\n", (unsigned)ESPNOW_WIFI_CHANNEL);
    Serial.print("Receiver STA MAC: ");
    Serial.println(WiFi.macAddress());
#if ENABLE_RECEIVER_UWB_ANCHOR
    Serial.println("Receiver UWB survey coordinator: enabled");
    Serial.print("Receiver anchor address: ");
    Serial.println(RECEIVER_UWB_ANCHOR_ADDRESS);
    Serial.print("Receiver survey tag address: ");
    Serial.println(RECEIVER_UWB_TAG_ADDRESS);
#else
    Serial.println("Receiver UWB survey coordinator: disabled");
#endif
  }
}

void loop() {
  handleHostCommands();
  serviceSurveyScheduler();

  if (RELAY_BINARY_MODE) {
    flushRelayFifoToSerial();
  } else {
    printStatsIfDue();
  }

#if ENABLE_RECEIVER_UWB_ANCHOR
  emitLocalSurveyReportIfDue();
  DW1000Ranging.loop();
#endif
}
