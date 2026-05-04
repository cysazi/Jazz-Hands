/*
==================== SETUP ====================

WIRING (ESP32 → DRV2605L):
- 3.3V  → VCC
- GND   → GND
- GPIO21 → SDA
- GPIO22 → SCL

MOTOR:
- Connect motor leads to OUT+ and OUT- on the driver

HOW TO USE:
- Write: motor.haptics(HAPTICS_ON, intensity);  = turn ON 
- Write: motor.haptics(HAPTICS_OFF);            = turn OFF 

INTENSITY:
- Range: 0–255
- Example: motor.haptics(HAPTICS_ON, 150); 

======================================================
*/

#include <Wire.h>

#define HAPTICS_ON  true
#define HAPTICS_OFF false

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

      writeReg(0x01, 0x05); // RTP mode (sustained control)
      writeReg(0x1A, 0x80); // LRA motor mode

      haptics(HAPTICS_OFF); // start off
    }

    void haptics(bool state, uint8_t strength = 127) {
      // state: ON/OFF
      // strength: vibration intensity (0–255)

      if (state == HAPTICS_ON) {
        writeReg(0x02, strength); // apply intensity
      } else {
        writeReg(0x02, 0);        // stop vibration
      }
    }
};

Haptics motor;

void setup() {
  motor.begin();
}

void loop() {
  motor.haptics(HAPTICS_ON, 150);
  delay(3000);

  motor.haptics(HAPTICS_OFF);
  delay(3000);
}