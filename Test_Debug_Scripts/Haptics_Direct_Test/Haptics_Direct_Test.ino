/*
  Simple Jazz Hands haptics test.

  Pick ONE pin pair below, upload, and the motor should pulse forever.
  If it does not work, comment that pair out and try the swapped pair.
*/

#include <Wire.h>

// Current glove wiring:
// const int SDA_PIN = 33;
// const int SCL_PIN = 32;

// Current glove wiring, swapped:
// const int SDA_PIN = 32;
// const int SCL_PIN = 33;

// Older wiring:
// const int SDA_PIN = 25;
// const int SCL_PIN = 26;

// Older wiring, swapped:
const int SDA_PIN = 26;
const int SCL_PIN = 25;

const byte DRV2605_ADDR = 0x5A;
const byte STRENGTH = 220;  // 0-255

void writeDrv(byte reg, byte value) {
  Wire.beginTransmission(DRV2605_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void motorOn() {
  writeDrv(0x02, STRENGTH);  // RTP input strength
}

void motorOff() {
  writeDrv(0x02, 0);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("Simple haptics test");
  Serial.print("SDA=");
  Serial.print(SDA_PIN);
  Serial.print(" SCL=");
  Serial.println(SCL_PIN);

  Wire.begin(SDA_PIN, SCL_PIN);

  writeDrv(0x01, 0x05);  // RTP mode
  writeDrv(0x1A, 0x80);  // LRA mode
  motorOff();
}

void loop() {
  Serial.println("buzz");
  motorOn();
  delay(500);

  Serial.println("unbuzz");
  motorOff();
  delay(500);
}
