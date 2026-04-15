#include <PDM.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// -----------------------------
// Microphone globals
// -----------------------------
short sampleBuffer[256];
volatile int samplesRead = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

// -----------------------------
// Thresholds
// Tune these on your board if needed
// -----------------------------
const int MIC_THRESHOLD = 1200;        // sound threshold
const int CLEAR_THRESHOLD = 120;       // lower = dark, higher = bright
const float MOTION_THRESHOLD = 0.25;   // motion metric threshold
const int PROX_THRESHOLD = 180;        // higher = near for APDS9960 on many boards

// -----------------------------
// State values
// -----------------------------
int micLevel = 0;
int clearValue = 0;
float motionMetric = 0.0;
int proxValue = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);

  // Microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM microphone.");
    while (1);
  }

  // IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  // Light/proximity sensor
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  Serial.println("Task 10 smart workspace classifier started");
}

void loop() {
  // -----------------------------
  // 1. Read microphone activity
  // -----------------------------
  if (samplesRead > 0) {
    long sum = 0;
    for (int i = 0; i < samplesRead; i++) {
      sum += abs(sampleBuffer[i]);
    }
    micLevel = sum / samplesRead;
    samplesRead = 0;
  }

  // -----------------------------
  // 2. Read ambient light
  // -----------------------------
  int r, g, b, c;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, c);
    clearValue = c;
  }

  // -----------------------------
  // 3. Read proximity
  // -----------------------------
  if (APDS.proximityAvailable()) {
    proxValue = APDS.readProximity();
  }

  // -----------------------------
  // 4. Read motion from IMU
  // Using acceleration magnitude difference from 1 g
  // -----------------------------
  float ax, ay, az;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    float mag = sqrt(ax * ax + ay * ay + az * az);
    motionMetric = abs(mag - 1.0);
  }

  // -----------------------------
  // 5. Threshold each modality
  // -----------------------------
  bool sound = (micLevel > MIC_THRESHOLD);
  bool dark = (clearValue < CLEAR_THRESHOLD);
  bool moving = (motionMetric > MOTION_THRESHOLD);
  bool near = (proxValue > PROX_THRESHOLD);

  // -----------------------------
  // 6. Rule-based final label
  // Must classify into one of the 4 required labels
  // -----------------------------
  const char* finalLabel = "QUIET_BRIGHT_STEADY_FAR";

  if (!sound && !dark && !moving && !near) {
    finalLabel = "QUIET_BRIGHT_STEADY_FAR";
  } 
  else if (sound && !dark && !moving && !near) {
    finalLabel = "NOISY_BRIGHT_STEADY_FAR";
  } 
  else if (!sound && dark && !moving && near) {
    finalLabel = "QUIET_DARK_STEADY_NEAR";
  } 
  else if (sound && !dark && moving && near) {
    finalLabel = "NOISY_BRIGHT_MOVING_NEAR";
  } 
  else {
    // Fallback: choose nearest intended state by priority
    if (sound && moving && near && !dark) {
      finalLabel = "NOISY_BRIGHT_MOVING_NEAR";
    } else if (!sound && dark && near) {
      finalLabel = "QUIET_DARK_STEADY_NEAR";
    } else if (sound && !dark) {
      finalLabel = "NOISY_BRIGHT_STEADY_FAR";
    } else {
      finalLabel = "QUIET_BRIGHT_STEADY_FAR";
    }
  }

  // -----------------------------
  // 7. Print required output
  // -----------------------------
  Serial.print("mic=");
  Serial.print(micLevel);
  Serial.print(" clear=");
  Serial.print(clearValue);
  Serial.print(" motion=");
  Serial.print(motionMetric, 3);
  Serial.print(" prox=");
  Serial.println(proxValue);

  Serial.print("sound=");
  Serial.print(sound ? 1 : 0);
  Serial.print(" dark=");
  Serial.print(dark ? 1 : 0);
  Serial.print(" moving=");
  Serial.print(moving ? 1 : 0);
  Serial.print(" near=");
  Serial.println(near ? 1 : 0);

  Serial.print("FINAL_LABEL=");
  Serial.println(finalLabel);
  Serial.println();

  delay(250);
}