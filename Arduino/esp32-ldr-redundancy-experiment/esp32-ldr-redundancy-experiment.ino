/*
  AURA ESP32 LDR redundancy experiment and serial bridge

  This sketch supports two demonstration modes:

  1. Standalone two-sensor AURA mode:
     - The ESP32 reads the two LDRs and computes the same pairwise AURA
       redundancy score used by the backend:
           aura = sin(pi * r1 / (r1 + r2))^2
     - Repeated redundancy evidence accumulates deactivation energy, matching
       AURA's temporal sleep-pressure design.
     - When the accumulated energy crosses the sleep threshold, sensor 2 is
       treated as redundant. LED1 stays ON and LED2 turns OFF.
     - If one LDR is covered or the readings become different, both sensors
       remain active. LED1 and LED2 are ON.

  2. AURA dashboard serial mode:
     - The backend sends comma-separated command bits over USB serial.
     - 0 means active/on.
     - 1 means sleep/off.
     - The ESP32 applies the first two bits to LED1 and LED2.

  Default wiring:
    LDR1 voltage divider output -> GPIO 34
    LDR2 voltage divider output -> GPIO 35
    LED1 anode through 220/330 ohm resistor -> GPIO 25, cathode -> GND
    LED2 anode through 220/330 ohm resistor -> GPIO 26, cathode -> GND

  LDR divider for each LDR:
    3.3V -> LDR -> analog pin -> 10k resistor -> GND

  Arduino IDE:
    Board: ESP32 Dev Module
    Port: the COM port shown when the ESP32 is connected
    Baud: 115200
*/

const long BAUD_RATE = 115200;

const int LDR1_PIN = 34;
const int LDR2_PIN = 35;
const int LED1_PIN = 25;
const int LED2_PIN = 26;

const float AURA_SCORE_THRESHOLD = 0.92f;
const float NORMALIZED_DIFF_THRESHOLD = 0.12f;
const float ENERGY_RISE = 0.08f;
const float ENERGY_DECAY = 0.16f;
const float SLEEP_ENERGY_THRESHOLD = 0.76f;
const float WAKE_ENERGY_THRESHOLD = 0.34f;
const float ADC_MAX_VALUE = 4095.0f;
const float SMOOTHING = 0.18f;
const unsigned long SAMPLE_INTERVAL_MS = 120;
const unsigned long SERIAL_COMMAND_HOLD_MS = 15000;
const int MAX_COMMAND_CHARS = 80;

char inputBuffer[MAX_COMMAND_CHARS];
int inputIndex = 0;
float ldr1Smooth = 0.0f;
float ldr2Smooth = 0.0f;
float deactivationEnergy = 0.0f;
unsigned long lastSampleMs = 0;
unsigned long lastSerialCommandMs = 0;
int lastActiveCount = 2;
int lastSleepingCount = 0;
float lastAuraScore = 0.0f;
float lastNormalizedDiff = 0.0f;

void setLedStates(bool node1Active, bool node2Active) {
  digitalWrite(LED1_PIN, node1Active ? HIGH : LOW);
  digitalWrite(LED2_PIN, node2Active ? HIGH : LOW);
  lastActiveCount = (node1Active ? 1 : 0) + (node2Active ? 1 : 0);
  lastSleepingCount = 2 - lastActiveCount;
}

void sendAck(const char* status) {
  Serial.print("ACK AURA_ESP32 ");
  Serial.print(status);
  Serial.print(" active=");
  Serial.print(lastActiveCount);
  Serial.print(" sleeping=");
  Serial.print(lastSleepingCount);
  Serial.print(" ldr1=");
  Serial.print((int)ldr1Smooth);
  Serial.print(" ldr2=");
  Serial.print((int)ldr2Smooth);
  Serial.print(" aura=");
  Serial.print(lastAuraScore, 3);
  Serial.print(" diff=");
  Serial.print(lastNormalizedDiff, 3);
  Serial.print(" energy=");
  Serial.println(deactivationEnergy, 3);
}

bool isCommandSeparator(char c) {
  return c == ',' || c == ' ' || c == ';' || c == '\t';
}

char upperAscii(char c) {
  if (c >= 'a' && c <= 'z') {
    return c - 32;
  }
  return c;
}

bool commandEquals(const char* left, const char* right) {
  int i = 0;
  while (left[i] != '\0' && right[i] != '\0') {
    if (upperAscii(left[i]) != upperAscii(right[i])) {
      return false;
    }
    i++;
  }
  return left[i] == '\0' && right[i] == '\0';
}

void sampleLdrs() {
  int raw1 = analogRead(LDR1_PIN);
  int raw2 = analogRead(LDR2_PIN);

  ldr1Smooth = (SMOOTHING * raw1) + ((1.0f - SMOOTHING) * ldr1Smooth);
  ldr2Smooth = (SMOOTHING * raw2) + ((1.0f - SMOOTHING) * ldr2Smooth);
}

void applyStandaloneLdrDecision() {
  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  sampleLdrs();
  float r1 = constrain(ldr1Smooth / ADC_MAX_VALUE, 0.0f, 1.0f);
  float r2 = constrain(ldr2Smooth / ADC_MAX_VALUE, 0.0f, 1.0f);
  float total = max(r1 + r2, 0.0001f);

  lastAuraScore = powf(sinf(PI * r1 / total), 2.0f);
  lastNormalizedDiff = fabsf(r1 - r2);
  bool redundantEvidence = lastAuraScore >= AURA_SCORE_THRESHOLD && lastNormalizedDiff <= NORMALIZED_DIFF_THRESHOLD;

  if (redundantEvidence) {
    deactivationEnergy += ENERGY_RISE;
  } else {
    deactivationEnergy -= ENERGY_DECAY;
  }
  deactivationEnergy = constrain(deactivationEnergy, 0.0f, 1.0f);

  bool node2CurrentlySleeping = lastSleepingCount > 0;
  bool node2ShouldSleep = node2CurrentlySleeping;
  if (deactivationEnergy >= SLEEP_ENERGY_THRESHOLD) {
    node2ShouldSleep = true;
  } else if (deactivationEnergy <= WAKE_ENERGY_THRESHOLD) {
    node2ShouldSleep = false;
  }

  // Node 1 is the representative active sensor. Node 2 sleeps only when the
  // accumulated two-sensor AURA energy says the readings are redundant.
  setLedStates(true, !node2ShouldSleep);
}

void applySerialBitCommand(char* command) {
  bool states[2] = {true, true};
  int node = 0;
  bool sawBit = false;

  for (int i = 0; command[i] != '\0' && node < 2; i++) {
    char c = command[i];
    if (c == '0' || c == '1') {
      states[node] = c == '0';
      node++;
      sawBit = true;
    } else if (!isCommandSeparator(c)) {
      sendAck("ERR_BAD_CHAR");
      return;
    }
  }

  if (!sawBit) {
    sendAck("ERR_EMPTY");
    return;
  }

  if (node == 1) {
    states[1] = false;
  }

  setLedStates(states[0], states[1]);
  lastSerialCommandMs = millis();
  sendAck("OK");
}

void handleCommand(char* command) {
  while (*command == ' ' || *command == '\t') {
    command++;
  }

  sampleLdrs();

  if (commandEquals(command, "PING")) {
    sendAck("PONG");
    return;
  }
  if (commandEquals(command, "AUTO")) {
    lastSerialCommandMs = 0;
    deactivationEnergy = 0.0f;
    sendAck("AUTO_AURA_MODE");
    return;
  }
  if (commandEquals(command, "BOTH_ON") || commandEquals(command, "ALL_ON")) {
    setLedStates(true, true);
    lastSerialCommandMs = millis();
    sendAck("BOTH_ON");
    return;
  }
  if (commandEquals(command, "LED2_SLEEP") || commandEquals(command, "0,1")) {
    setLedStates(true, false);
    lastSerialCommandMs = millis();
    sendAck("LED2_SLEEP");
    return;
  }

  applySerialBitCommand(command);
}

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);

  analogReadResolution(12);
  analogSetPinAttenuation(LDR1_PIN, ADC_11db);
  analogSetPinAttenuation(LDR2_PIN, ADC_11db);

  ldr1Smooth = analogRead(LDR1_PIN);
  ldr2Smooth = analogRead(LDR2_PIN);

  setLedStates(true, true);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputIndex > 0) {
        inputBuffer[inputIndex] = '\0';
        handleCommand(inputBuffer);
        inputIndex = 0;
      }
      continue;
    }

    if (inputIndex < MAX_COMMAND_CHARS - 1) {
      inputBuffer[inputIndex++] = c;
    } else {
      inputIndex = 0;
      sendAck("ERR_TOO_LONG");
    }
  }

  bool serialOverrideActive =
      lastSerialCommandMs != 0 && (millis() - lastSerialCommandMs) < SERIAL_COMMAND_HOLD_MS;

  if (!serialOverrideActive) {
    applyStandaloneLdrDecision();
  }
}
