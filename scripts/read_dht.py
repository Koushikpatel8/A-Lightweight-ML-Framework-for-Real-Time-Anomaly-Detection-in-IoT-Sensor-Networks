# scripts/read_dht.py
# This script is used to test and continuously read data from the DHT11 sensor.
# It collects temperature (in Celsius) and humidity values from the sensor 
# connected to the Raspberry Pi (using GPIO4, BCM numbering).
# Occasional read errors are expected because the DHT sensor is known to be noisy,
# so those errors are caught and ignored during execution.

import time
import adafruit_dht
import board

# Initialize the DHT11 sensor on GPIO pin 4 (BCM numbering)
dht = adafruit_dht.DHT11(board.D4)

# Continuously read temperature and humidity
while True:
    try:
        # Attempt to read temperature and humidity
        t = dht.temperature  # Temperature in °C
        h = dht.humidity     # Humidity in %
        print(f"Temp: {t} °C, Humidity: {h} %")
    except Exception as e:
        # The sensor sometimes fails to read values correctly, 
        # so errors are printed but ignored to keep the loop running.
        print("Read error:", e)
    # Wait for 2 seconds before the next reading
    time.sleep(2)
