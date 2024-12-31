int ledPins[] = {8, 9, 10, 11, 12};  

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < 5; i++) {
    pinMode(ledPins[i], OUTPUT);
  }
}

void loop() {
  if (Serial.available() >= 5) {  
    for (int i = 0; i < 5; i++) {
      int fingerState = Serial.read(); 
      digitalWrite(ledPins[i], fingerState == 1 ? HIGH : LOW);  
    }
  }
}
