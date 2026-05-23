// Arduino Mega 2560 all-digital-pin LED ON test
//
// Connect one LED to any digital pin from 2 to 53:
//   Arduino pin -> 220 ohm or 330 ohm resistor -> LED long leg
//   LED short leg -> GND
//
// The sketch sets every digital pin from 2 to 53 HIGH.
// If your LED is connected to any tested pin, it should stay on.

const int FIRST_PIN = 2;
const int LAST_PIN = 53;

void setup() {
  for (int pin = FIRST_PIN; pin <= LAST_PIN; pin++) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, HIGH);
  }
}

void loop() {
  // Keep all tested pins on.
}
