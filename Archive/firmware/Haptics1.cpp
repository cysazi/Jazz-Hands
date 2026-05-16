/*
Serial haptics receiver for Jazz Hands.

Wiring (ESP32 -> DRV2605L):
- 3.3V  -> VCC
- GND   -> GND
- GPIO25 -> SDA
- GPIO26 -> SCL

Serial command:
P,<HAND>,<INTENSITY>,<DURATION_MS>\n
Example:
P,LEFT,150,55
*/

#include <Wire.h>

#define HAPTICS_ON true
#define HAPTICS_OFF false

const uint32_t SERIAL_BAUD = 115200;
const int HAPTICS_SDA_PIN = 25;
const int HAPTICS_SCL_PIN = 26;

class Haptics {
  private:
    const uint8_t DRV_ADDR = 0x5A;

    void writeReg(uint8_t reg, uint8_t val) {
      Wire.beginTransmission(DRV_ADDR);
      Wire.write(reg);
      Wire.write(val);
      Wire.endTransmission();
    }

  public:
    void begin(int sdaPin = 21, int sclPin = 22) {
      Wire.begin(sdaPin, sclPin);
      writeReg(0x01, 0x05); // RTP mode
      writeReg(0x1A, 0x80); // LRA motor mode
      haptics(HAPTICS_OFF);
    }

    void haptics(bool state, uint8_t strength = 127) {
      writeReg(0x02, state == HAPTICS_ON ? strength : 0);
    }
};

Haptics motor;

String serialLine;
uint32_t pulseEndMs = 0;

void startPulse(uint8_t intensity, uint16_t durationMs) {
  motor.haptics(HAPTICS_ON, intensity);
  pulseEndMs = millis() + durationMs;
}

void handleCommand(const String &line) {
  if (!line.startsWith("P,")) {
    return;
  }

  int firstComma = line.indexOf(',');
  int secondComma = line.indexOf(',', firstComma + 1);
  int thirdComma = line.indexOf(',', secondComma + 1);
  if (firstComma < 0 || secondComma < 0 || thirdComma < 0) {
    return;
  }

  int intensity = line.substring(secondComma + 1, thirdComma).toInt();
  int durationMs = line.substring(thirdComma + 1).toInt();
  intensity = constrain(intensity, 0, 255);
  durationMs = constrain(durationMs, 1, 1000);
  startPulse((uint8_t)intensity, (uint16_t)durationMs);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  motor.begin(HAPTICS_SDA_PIN, HAPTICS_SCL_PIN);
  serialLine.reserve(64);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      serialLine.trim();
      handleCommand(serialLine);
      serialLine = "";
    } else if (c != '\r' && serialLine.length() < 63) {
      serialLine += c;
    }
  }

  if (pulseEndMs != 0 && (int32_t)(millis() - pulseEndMs) >= 0) {
    motor.haptics(HAPTICS_OFF);
    pulseEndMs = 0;
  }
}
