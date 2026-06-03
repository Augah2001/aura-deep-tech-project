/*
  AURA Arduino Mega bridge

  The FastAPI backend sends a comma-separated command string such as:
      0,1,0,1,1,0

  Convention:
      0 = active/on
      1 = sleeping/off

  This sketch maps the first 28 visible AURA nodes to Arduino Mega pins 22-49.
  In the demonstration board layout, node 10 maps to pin 31 and uses the
  yellow LED shown in the hardware illustration.
  Attach one LED per pin through a resistor to GND. An LED ON means the node is
  active. An LED OFF means AURA has suppressed that node.
*/

const unsigned long BAUD_RATE = 115200;
const byte NODE_COUNT = 28;

const byte NODE_PINS[NODE_COUNT] = {
  22, 23, 24, 25, 26, 27, 28,
  29, 30, 31, 32, 33, 34, 35,
  36, 37, 38, 39, 40, 41, 42,
  43, 44, 45, 46, 47, 48, 49
};

String inputLine = "";

void setAllActive() {
  for (byte i = 0; i < NODE_COUNT; i++) {
    digitalWrite(NODE_PINS[i], HIGH);
  }
}

void applyCommand(const String &command) {
  byte nodeIndex = 0;
  int activeCount = 0;
  int sleepCount = 0;

  for (unsigned int i = 0; i < command.length() && nodeIndex < NODE_COUNT; i++) {
    char value = command.charAt(i);

    if (value == '0') {
      digitalWrite(NODE_PINS[nodeIndex], HIGH);
      activeCount++;
      nodeIndex++;
    } else if (value == '1') {
      digitalWrite(NODE_PINS[nodeIndex], LOW);
      sleepCount++;
      nodeIndex++;
    }
  }

  // Leave unused visible nodes off if a shorter command is sent.
  while (nodeIndex < NODE_COUNT) {
    digitalWrite(NODE_PINS[nodeIndex], LOW);
    nodeIndex++;
  }

  Serial.print("ACK AURA active=");
  Serial.print(activeCount);
  Serial.print(" sleeping=");
  Serial.println(sleepCount);
}

void setup() {
  Serial.begin(BAUD_RATE);

  for (byte i = 0; i < NODE_COUNT; i++) {
    pinMode(NODE_PINS[i], OUTPUT);
  }

  setAllActive();
  Serial.println("AURA Mega bridge ready");
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n' || incoming == '\r') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        applyCommand(inputLine);
      }
      inputLine = "";
    } else {
      inputLine += incoming;
    }
  }
}
