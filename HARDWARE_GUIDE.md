# RCDS Arduino Uno Colorimeter, pH, and Temperature Hardware Guide

This is a benchtop research prototype. The Griess reaction reports nitrite under the selected assay conditions; it does **not** directly identify or quantify nitrosamines, carcinogens, DNA adducts, or personal exposure. The dashboard's biological and risk outputs are illustrative, not clinical measurements. Use the datasheets for the exact breakouts in hand: TCS34725 boards vary in 5 V tolerance and I2C level shifting, and pH boards have different output ranges and calibration steps.

## Bill of materials

| Qty | Item | Use / specification |
|---:|---|---|
| 1 | Arduino Uno R3 or compatible 5 V board | Controller and USB serial |
| 1 | TCS34725 RGB/clear sensor breakout | Prefer board with regulator and I2C level shifting |
| 1 | Green LED, nominal 525–545 nm | Fixed illumination near Griess visible response |
| 1 | 220 Ω resistor | LED current limit |
| 1 | Analog pH interface board | Compatible with pH electrode; analog output must fit Uno ADC range |
| 1 | Combination pH electrode | Matching BNC electrode, cap, and manufacturer-recommended storage solution |
| 1 | Waterproof DS18B20 probe | Liquid temperature |
| 1 | 4.7 kΩ resistor | DS18B20 1-Wire pull-up |
| 1 | 10 mm-path cuvette | Same cuvette type/path length for all reads |
| 1 | Opaque matte-black enclosure | LED, cuvette, and detector housing |
| 1 | Diffuser | Opal acrylic or diffuser film |
| Several | Jumper leads, breadboard/prototype board, USB cable | Initial assembly; solder stable final connections |
| Optional | 3D-printed holder, baffles, black flocking | Alignment and stray-light control |
| Calibration supplies | Fresh pH 4.01, 7.00, optionally 10.01 buffers; calibrated thermometer | Calibration and verification |
| Assay supplies | Commercial Griess reagent and known nitrite standards | Follow reagent instructions and SDS; matrix-match standards |

The TCS34725 is a broad RGB/clear detector, not a narrow 540 nm spectrophotometer. Adafruit describes the voltage protection on its own breakout; don't assume a generic module matches it. [Adafruit TCS34725 guide](https://learn.adafruit.com/adafruit-color-sensors). [DS18B20 data sheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ds18b20.pdf).

## Pin-by-pin wiring

Disconnect USB before wiring. Tie grounds together. The pH glass electrode connects to the pH board's BNC only, never directly to A0. Keep its analog lead away from LED current and I2C wiring.

| Uno pin / rail | Destination | Detail |
|---|---|---|
| 5V | TCS34725 VIN/VCC | Only if that breakout supports 5 V |
| GND | TCS34725 GND | Common ground |
| A4 / SDA | TCS34725 SDA | Uno R3 I2C data |
| A5 / SCL | TCS34725 SCL | Uno R3 I2C clock |
| 5V | pH interface VCC | Check board supply |
| GND | pH interface GND | Common ground |
| A0 | pH interface AO | Analog output; verify below ADC reference |
| 5V | DS18B20 VDD (often red) | External-powered mode |
| GND | DS18B20 GND (often black) | Common ground |
| D2 | DS18B20 DQ/DATA (often yellow) | 1-Wire data |
| 4.7 kΩ resistor | Between D2/DATA and 5V | Bus pull-up |
| 5V through 220 Ω | LED anode | LED cathode to GND |

Text wiring map:

    Uno 5V -> TCS34725 VIN/VCC       Uno A4/SDA -> TCS34725 SDA
    Uno GND -> TCS34725 GND           Uno A5/SCL -> TCS34725 SCL
    Uno 5V -> pH interface VCC        Uno A0 <- pH interface AO
    Uno GND -> pH interface GND       pH electrode -> interface BNC
    Uno 5V -> DS18B20 VDD
    Uno GND -> DS18B20 GND
    Uno D2 -> DS18B20 DATA
    D2/DATA -> 4.7 kΩ resistor -> Uno 5V
    Uno 5V -> 220 Ω resistor -> LED anode; LED cathode -> Uno GND

The 4.7 kΩ pull-up is specified for the 1-Wire bus. [Analog Devices 1-Wire interface note](https://www.analog.com/en/resources/technical-articles/interfacing-the-ds18x20ds1822-1wire-temperature-sensor-in-a-microcontroller-environment.html).

## Optical cuvette enclosure

1. Make a rigid matte-black box, approximately 100 × 50 × 50 mm or larger as needed. Close it fully during readings. Blacken reflective interior surfaces and keep loose fibers away from samples.
2. Mount the LED and TCS34725 opposite each other, with the cuvette between them on the 10 mm optical path. Align LED, cuvette, and detector apertures; keep the sensor board dry and outside the sample area.
3. Put a diffuser between LED and cuvette. Fix LED current, sensor gain/integration time, diffuser, and geometry after calibration.
4. Add small apertures and baffles so the sensor sees transmitted light through the cuvette, not stray light around it or wall reflections.
5. Add a hard stop or keyed holder. Mark one cuvette face and insert it the same way each time. Keep fill height fixed, remove bubbles, and wipe the outside.
6. Measure pH and temperature separately, then remove probes before inserting the cuvette. Do not press the pH bulb against a vessel wall.
7. Check the closed enclosure blocks room light. A green LED near 525–545 nm is useful for a Griess color response, but the broad RGB detector still needs empirical standards.

Griess procedures use a nitrite standard curve and commonly read near 540–548 nm; wavelength, reagent order, and incubation time depend on the kit. [Promega Griess protocol](https://www.promega.com/-/media/files/resources/protocols/technical-bulletins/0/griess-reagent-system-protocol.pdf), [Thermo Fisher protocol](https://documents.thermofisher.com/TFS-Assets/LSG/manuals/mp07921.pdf).

## Calibration procedure

### 1. Lock down hardware and firmware

1. Verify each board's supply voltage, pinout, and analog output limits. Upload rcds_arduino_uno.ino and confirm the 9600-baud serial line contains eight comma-separated fields: R,G,B,blank_R,blank_G,blank_B,pH,temp_C.
2. Fix optical geometry and sensor settings. Warm the LED and electronics for five minutes.
3. Replace the sketch's blank RGB placeholders (currently 65535) with an actual reagent-blank reading at the selected sensor settings. Ensure blank and sample readings do not saturate.
4. Close the enclosure and compare the reading with the lid open/closed; a large change means stray light remains.

### 2. Calibrate color response using nitrite

1. Use commercial Griess reagent per the supplier's protocol and SDS. Make a zero-nitrite reagent blank and known nitrite standards in the same buffer/matrix as samples. An initial study series may be 0, 5, 10, 20, 50, 100, and 200 µM; narrow it if sensor/reagent responses saturate.
2. Follow the kit's exact reagent volumes, order, incubation time, and temperature for every standard and sample. Record at least three cuvette readings at each level.
3. Clean the outside, use the same fill line and orientation, remove bubbles, close the enclosure, and save raw RGB/clear counts. Re-run the blank between groups to watch drift.
4. If dark intensity D is available, let I0 be blank intensity and I sample intensity. Calculate corrected absorbance A = -log10((I-D)/(I0-D)). Reject clipped measurements and any case with I-D ≤ 0.
5. Fit A = mC + b against known nitrite concentration C over the verified linear region; convert by C = (A-b)/m. Inspect residuals and triplicate spread; do not force a linear fit through saturation.
6. Save curve, date, reagent lot, matrix, sensor settings, blank, and temperature. Recheck blank and mid-range standard each session; repeat calibration after changing reagent, matrix, LED, cuvette, geometry, or sensor gain/integration time.
7. Do not extrapolate beyond standards. Dilute and re-assay or flag out-of-range samples.

A multi-level nitrite curve around 540 nm is described in published assay work. [Example study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8348835/). This assay is nitrite-specific under its validated chemistry, not a direct nitrosamine test.

### 3. Calibrate pH

1. Use fresh, traceable pH 7.00 and 4.01 buffers; include 10.01 if alkaline samples are expected. Let buffers and electrode reach stable temperature.
2. Rinse with deionized water between buffers and gently blot; do not wipe the bulb. Keep the electrode hydrated and store in the maker's storage solution, not DI water.
3. Read pH-board output voltage through the Uno with the same supply, wiring, and ADC reference used in operation. Record repeated voltage readings after stabilization.
4. Fit pH = aV + b from buffer pH values and measured voltages. Use pH 4 and 7 for a two-point fit; check fit linearity with a third buffer. Update PH_SLOPE and PH_OFFSET in the sketch (current entries are placeholders).
5. Verify against a buffer not used in the fit and record the error. Follow any board-specific trimmer or software calibration procedure.
6. Calibrate and measure at similar temperature or use an interface with automatic temperature compensation. A DS18B20 reading alone does not compensate the pH electrode.

### 4. Verify temperature and complete chain

1. Compare the DS18B20 in a stirred ice-water bath and at room temperature against a calibrated thermometer; wait for equilibrium and keep electrical splices dry.
2. Run a blank and mid-range nitrite control at the beginning and end of each session. Log raw channels, absorbance, curve-derived concentration, pH, temperature, replicate spread, reagent lot, and any flags.
3. Check matrix-matched spike recovery. A provisional engineering target such as 80–120% recovery and <5% replicate CV may help identify problems, but validate acceptance criteria for the actual assay; these are not guaranteed specifications.
4. Don't report values outside the validated range. Dilute and repeat or mark them out of range.

## Firmware and interpretation limitations

The sketch needs Arduino libraries Adafruit TCS34725, OneWire, and DallasTemperature. Edit its blank RGB values and pH slope/offset. Keep serial numeric ranges consistent with the sensor's raw channel output.

The dashboard currently uses an illustrative conversion with epsilon_M_cm = 3.8e3; it does not yet apply your measured calibration slope/intercept. Implement the measured curve in host software before quantitative use. The hardware measures optical assay color, pH, and temperature; it does not chemically identify carcinogens or measure DNA adducts.

## References

- [Adafruit TCS34725 Color Sensors](https://learn.adafruit.com/adafruit-color-sensors)
- [Analog Devices DS18B20 data sheet](https://www.analog.com/media/en/technical-documentation/data-sheets/ds18b20.pdf)
- [Analog Devices 1-Wire interface note](https://www.analog.com/en/resources/technical-articles/interfacing-the-ds18x20ds1822-1wire-temperature-sensor-in-a-microcontroller-environment.html)
- [Promega Griess Reagent System protocol](https://www.promega.com/-/media/files/resources/protocols/technical-bulletins/0/griess-reagent-system-protocol.pdf)
- [Thermo Fisher Griess protocol](https://documents.thermofisher.com/TFS-Assets/LSG/manuals/mp07921.pdf)
- [Nitrite assay calibration study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8348835/)
