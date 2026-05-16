#include <Wire.h>
#include <Adafruit_BNO08x.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define button 23

Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

unsigned long time_var = 0;
unsigned long delta_time = 0;

bool lastState = 1;

void setReports() {
  Serial.println("Setting desired reports");
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    Serial.println("Could not enable rotation vector");
  }
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 10000)) {
    Serial.println("Could not enable linear acceleration");
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(button, INPUT_PULLUP);

  Serial.println("BNO08x test");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);  // 100kHz for stability
  delay(100);

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }

  Serial.println("BNO08x Found!");
  setReports();
  Serial.println("Reading events...");
}

// Global variables to store latest sensor data
float quat_w = 0, quat_i = 0, quat_j = 0, quat_k = 0;
float accel_x = 0, accel_y = 0, accel_z = 0;

void loop() {
  if (bno08x.wasReset()) {
    Serial.println("sensor was reset!");
    setReports();
  }

  bool reading = digitalRead(button);
  if (reading == LOW && lastState == HIGH) {
    Serial.println("BUTTON:1");
  }
  lastState = reading;

  // Read sensor data and store latest values
  if (bno08x.getSensorEvent(&sensorValue)) {

    switch (sensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        quat_w = sensorValue.un.rotationVector.real;
        quat_i = sensorValue.un.rotationVector.i;
        quat_j = sensorValue.un.rotationVector.j;
        quat_k = sensorValue.un.rotationVector.k;
        break;

      case SH2_LINEAR_ACCELERATION:
        accel_x = sensorValue.un.linearAcceleration.x;
        accel_y = sensorValue.un.linearAcceleration.y;
        accel_z = sensorValue.un.linearAcceleration.z;
        break;
    }
  }

  // Print both together every 10ms
  if (millis() - delta_time > 10) {
    delta_time = millis();

    Serial.print(quat_w, 6);
    Serial.print(",");
    Serial.print(quat_i, 6);
    Serial.print(",");
    Serial.print(quat_j, 6);
    Serial.print(",");
    Serial.print(quat_k, 6);
    Serial.print(",");
    Serial.print(accel_x, 2);
    Serial.print(",");
    Serial.print(accel_y, 2);
    Serial.print(",");
    Serial.println(accel_z, 2);
  }
}