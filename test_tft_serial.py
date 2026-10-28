
import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

try:
    esp32 = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # ESP32 may reset when serial opens

    print("Connected to ESP32!")
    print("Type a message to display on the TFT.")
    print("Type 'exit' to quit.")

    while True:
        text = input("Text: ")

        if text.lower() == "exit":
            break

        esp32.write((text + "\n").encode("utf-8"))
        print("Sent:", text)

except serial.SerialException as error:
    print("Serial connection error:", error)

finally:
    if "esp32" in locals() and esp32.is_open:
        esp32.close()
