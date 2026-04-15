#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// -----------------------------
// Thresholds
// Tune these based on your room and board
// -----------------------------
const float HUMIDITY_JUMP_THRESHOLD = 3.0;     // %RH increase from baseline
const float TEMP_RISE_THRESHOLD = 1.0;         // deg C increase from baseline
const float MAG_SHIFT_THRESHOLD = 15.0;        // magnetic field change threshold
const int CLEAR_CHANGE_THRESHOLD = 80;         // clear channel change
const int COLOR_CHANGE_THRESHOLD = 60;         // summed RGB channel change

const unsigned long COOLDOWN_MS = 2000;

// -----------------------------
// Sensor values
// -----------------------------
float rh = 0.0;
float temp = 0.0;
float magMetric = 0.0;
int r = 0, g = 0, b = 0, clearValue = 0;

// -----------------------------
// Baseline values
// -----------------------------
float baselineRH = 0.0;
float baselineTemp = 0.0;
float baselineMag = 0.0;
int baselineR = 0, baselineG = 0, baselineB = 0, baselineClear = 0;

bool baselineSet = false;

// -----------------------------
// Event cooldown state
// -----------------------------
unsigned long lastEventTime = 0;
const char* latchedLabel = "BASELINE_NORMAL";

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (!HS300x.begin()) {
    Serial.println("Failed to initialize humidity/temperature sensor.");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  Serial.println("Task 11 event detector started");
}

void loop() {
  // -----------------------------
  // Read humidity and temperature
  // -----------------------------
  rh = HS300x.readHumidity();
  temp = HS300x.readTemperature();

  // -----------------------------
  // Read magnetic field
  // -----------------------------
  float mx, my, mz;
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    magMetric = sqrt(mx * mx + my * my + mz * mz);
  }

  // -----------------------------
  // Read color / ambient light
  // -----------------------------
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clearValue);
  }

  // -----------------------------
  // Set initial baseline once
  // -----------------------------
  if (!baselineSet) {
    baselineRH = rh;
    baselineTemp = temp;
    baselineMag = magMetric;
    baselineR = r;
    baselineG = g;
    baselineB = b;
    baselineClear = clearValue;
    baselineSet = true;
    delay(500);
    return;
  }

  // -----------------------------
  // Compute thresholded indicators
  // -----------------------------
  bool humid_jump = (rh - baselineRH) > HUMIDITY_JUMP_THRESHOLD;
  bool temp_rise = (temp - baselineTemp) > TEMP_RISE_THRESHOLD;
  bool mag_shift = abs(magMetric - baselineMag) > MAG_SHIFT_THRESHOLD;

  int clearDiff = abs(clearValue - baselineClear);
  int colorDiff = abs(r - baselineR) + abs(g - baselineG) + abs(b - baselineB);
  bool light_or_color_change = (clearDiff > CLEAR_CHANGE_THRESHOLD) || (colorDiff > COLOR_CHANGE_THRESHOLD);

  // -----------------------------
  // Rule-based event detection
  // Priority matters
  // -----------------------------
  const char* detectedLabel = "BASELINE_NORMAL";

  if (mag_shift) {
    detectedLabel = "MAGNETIC_DISTURBANCE_EVENT";
  } else if (humid_jump || temp_rise) {
    detectedLabel = "BREATH_OR_WARM_AIR_EVENT";
  } else if (light_or_color_change) {
    detectedLabel = "LIGHT_OR_COLOR_CHANGE_EVENT";
  } else {
    detectedLabel = "BASELINE_NORMAL";
  }

  // -----------------------------
  // Cooldown / debounce logic
  // -----------------------------
  unsigned long now = millis();

  if (strcmp(detectedLabel, "BASELINE_NORMAL") == 0) {
    latchedLabel = "BASELINE_NORMAL";
  } else {
    if ((now - lastEventTime) > COOLDOWN_MS || strcmp(latchedLabel, detectedLabel) != 0) {
      latchedLabel = detectedLabel;
      lastEventTime = now;
    }
  }

  // -----------------------------
  // Serial output
  // -----------------------------
  Serial.print("rh=");
  Serial.print(rh, 2);
  Serial.print(" temp=");
  Serial.print(temp, 2);
  Serial.print(" mag=");
  Serial.print(magMetric, 2);
  Serial.print(" r=");
  Serial.print(r);
  Serial.print(" g=");
  Serial.print(g);
  Serial.print(" b=");
  Serial.print(b);
  Serial.print(" clear=");
  Serial.println(clearValue);

  Serial.print("humid_jump=");
  Serial.print(humid_jump ? 1 : 0);
  Serial.print(" temp_rise=");
  Serial.print(temp_rise ? 1 : 0);
  Serial.print(" mag_shift=");
  Serial.print(mag_shift ? 1 : 0);
  Serial.print(" light_or_color_change=");
  Serial.println(light_or_color_change ? 1 : 0);

  Serial.print("FINAL_LABEL=");
  Serial.println(latchedLabel);
  Serial.println();

  delay(250);
}