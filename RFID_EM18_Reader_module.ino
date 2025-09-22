#include <SoftwareSerial.h>
SoftwareSerial rfid(2, 3); // RX=2

void setup() {
  Serial.begin(9600);
  rfid.begin(9600); // try 2400 if nothing appears
  Serial.println("Place RFID tag...");
}

void loop() {
  if (rfid.available()) {
    char c = rfid.read();
    Serial.write(c);  // print raw data
  }
}
