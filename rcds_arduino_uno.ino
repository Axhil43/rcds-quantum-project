/* Arduino Uno RCDS sensor stream.
 * Libraries: Adafruit TCS34725, OneWire, DallasTemperature.
 * CSV (no header): R,G,B,blank_R,blank_G,blank_B,pH,temp_C
 * Calibrate blank channels and pH conversion using the actual assay and buffers.
 */
#include <Wire.h>
#include <Adafruit_TCS34725.h>
#include <OneWire.h>
#include <DallasTemperature.h>
const uint8_t PH_PIN = A0;
const uint8_t ONE_WIRE_PIN = 2;
const float ADC_REFERENCE_V = 5.0;
const float PH_SLOPE = -5.70;   // illustrative; calibrate
const float PH_OFFSET = 21.34;  // illustrative; calibrate
const uint16_t BLANK_R = 65535; // replace with measured reagent-blank RGB
const uint16_t BLANK_G = 65535;
const uint16_t BLANK_B = 65535;
Adafruit_TCS34725 rgbSensor(TCS34725_INTEGRATIONTIME_154MS, TCS34725_GAIN_4X);
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature temperatureSensor(&oneWire);
void setup() {
  Serial.begin(9600);
  Wire.begin();
  if (!rgbSensor.begin()) { while (true) delay(1000); }
  temperatureSensor.begin();
}
void loop() {
  uint16_t red, green, blue, clear;
  rgbSensor.getRawData(&red, &green, &blue, &clear);
  float voltage = analogRead(PH_PIN) * ADC_REFERENCE_V / 1023.0;
  float pH = PH_SLOPE * voltage + PH_OFFSET;
  temperatureSensor.requestTemperatures();
  float tempC = temperatureSensor.getTempCByIndex(0);
  Serial.print(red); Serial.print(',');
  Serial.print(green); Serial.print(',');
  Serial.print(blue); Serial.print(',');
  Serial.print(BLANK_R); Serial.print(',');
  Serial.print(BLANK_G); Serial.print(',');
  Serial.print(BLANK_B); Serial.print(',');
  Serial.print(pH,2); Serial.print(',');
  Serial.println(tempC,2);
  delay(1000);
}
