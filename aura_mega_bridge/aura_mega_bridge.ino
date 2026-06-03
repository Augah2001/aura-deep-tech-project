/*
  AURA Arduino Mega LED sleep/wake bridge

  Purpose:
    Receives AURA serial command bits from the dashboard backend and maps them
    to LEDs. A command bit of 0 means the sensor/node is active, so the LED is
    ON. A command bit of 1 means the sensor/node is sleeping, so the LED is OFF.

  Expected serial command format:
    0,1,0,0,1,1,0

  Default wiring:
    Node 1  -> pin 22 -> 220/330 ohm resistor -> LED anode, LED cathode -> GND
    Node 2  -> pin 23 -> resistor -> LED -> GND
    ...
    Node 10 -> pin 31 -> resistor -> yellow LED -> GND
    Node 28 -> pin 49 -> resistor -> LED -> GND

  Arduino IDE:
    Board: Arduino Mega or Mega 2560
    Port: the COM port shown when the Mega is connected
    Baud: 115200
*/

const long BAUD_RATE = 115200;
const int START_PIN = 22;
const int NODE_COUNT = 28;
const int MAX_COMMAND_CHARS = 160;

char inputBuffer[MAX_COMMAND_CHARS];
int inputIndex = 0;
int lastActiveCount = NODE_COUNT;
int lastSleepingCount = 0;

void applyAllActive() {
  for (int i = 0; i < NODE_COUNT; i++) {
    digitalWrite(START_PIN + i, HIGH);
  }
  lastActiveCount = NODE_COUNT;
  lastSleepingCount = 0;
}

void applyAllSleep() {
  for (int i = 0; i < NODE_COUNT; i++) {
    digitalWrite(START_PIN + i, LOW);
  }
  lastActiveCount = 0;
  lastSleepingCount = NODE_COUNT;
}

void blinkReadyPattern() {
  applyAllSleep();
  delay(150);
  for (int i = 0; i < min(NODE_COUNT, 6); i++) {
    digitalWrite(START_PIN + i, HIGH);
    delay(50);
  }
  delay(150);
  applyAllActive();
}

void sendAck(const char* status) {
  Serial.print("ACK AURA_MEGA ");
  Serial.print(status);
  Serial.print(" active=");
  Serial.print(lastActiveCount);
  Serial.print(" sleeping=");
  Serial.println(lastSleepingCount);
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

void applyBitCommand(char* command) {
  int node = 0;
  int activeCount = 0;
  int sleepingCount = 0;
  bool sawBit = false;

  for (int i = 0; command[i] != '\0' && node < NODE_COUNT; i++) {
    char c = command[i];
    if (c == '0' || c == '1') {
      bool active = c == '0';
      digitalWrite(START_PIN + node, active ? HIGH : LOW);
      activeCount += active ? 1 : 0;
      sleepingCount += active ? 0 : 1;
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

  // Any LEDs beyond the received command are switched off so stale state does
  // not look like an active AURA decision.
  for (int i = node; i < NODE_COUNT; i++) {
    digitalWrite(START_PIN + i, LOW);
  }

  lastActiveCount = activeCount;
  lastSleepingCount = sleepingCount + (NODE_COUNT - node);
  sendAck("OK");
}

void handleCommand(char* command) {
  while (*command == ' ' || *command == '\t') {
    command++;
  }

  if (commandEquals(command, "PING")) {
    sendAck("PONG");
    return;
  }
  if (commandEquals(command, "ALL_ON") || commandEquals(command, "ACTIVE")) {
    applyAllActive();
    sendAck("ALL_ON");
    return;
  }
  if (commandEquals(command, "ALL_OFF") || commandEquals(command, "SLEEP")) {
    applyAllSleep();
    sendAck("ALL_OFF");
    return;
  }

  applyBitCommand(command);
}

void setup() {
  Serial.begin(BAUD_RATE);

  for (int i = 0; i < NODE_COUNT; i++) {
    pinMode(START_PIN + i, OUTPUT);
  }

  blinkReadyPattern();
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
}
