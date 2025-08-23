import time
import adafruit_dht
import board

# DHT11 on GPIO4 (BCM numbering)
dht = adafruit_dht.DHT11(board.D4)

while True:
    try:
        t = dht.temperature  # °C
        h = dht.humidity     # %
        print(f"Temp: {t} °C, Humidity: {h} %")
    except Exception as e:
        # DHTs are noisy; ignore occasional read errors
        print("Read error:", e)
    time.sleep(2)
