# Sign2Text — Real-Time Sign Language Recognition with ESP32 TFT

A real-time sign language recognition project that uses a laptop camera to recognize hand signs, displays recognition results on the laptop screen, and transmits the recognized text to an ESP32 over USB serial for display on an ILI9341 TFT LCD.

The project combines computer vision, machine learning, Python-based inference, serial communication, and embedded display control.

## Features

- **Live camera input:** Captures video from the laptop camera.
- **Sign recognition:** Processes camera frames using the available trained recognition models.
- **Laptop display:** Shows the live camera feed and recognition results.
- **ESP32 integration:** Sends recognized text from the Python application to the ESP32 through USB serial communication.
- **TFT output:** Displays recognized text on an ILI9341-based 2.4-inch SPI TFT LCD (240 × 320).
- **Multiple recognition modes:** Includes gloss-based recognition, word recognition, and alphabet/fingerspelling recognition launchers.
- **Reusable firmware:** The ESP32 firmware remains flashed between sessions, so it does not need to be uploaded every time the application is launched.

## System Architecture

```text
                ┌──────────────────────┐
                │   Laptop Webcam      │
                └──────────┬───────────┘
                           │ Video Frames
                           ▼
                ┌──────────────────────┐
                │  Python Application  │
                │                      │
                │  Image Processing    │
                │  Landmark Extraction │
                │  Model Inference     │
                │  Text Prediction     │
                └──────────┬───────────┘
                           │
                ┌──────────┴───────────┐
                │                      │
                ▼                      ▼
      ┌──────────────────┐  ┌────────────────────┐
      │  Laptop Display  │  │ USB Serial Output  │
      │                  │  │                    │
      │ Live Camera Feed │  │ Recognized Text    │
      │ Predictions      │  │ Newline-Terminated │
      └──────────────────┘  └─────────┬──────────┘
                                      │ USB
                                      ▼
                            ┌────────────────────┐
                            │       ESP32        │
                            │ Serial Reception   │
                            │ TFT_eSPI Rendering │
                            └─────────┬──────────┘
                                      │ SPI
                                      ▼
                            ┌────────────────────┐
                            │  ILI9341 TFT LCD   │
                            │                    │
                            │ Recognized Text    │
                            └────────────────────┘
```

## Hardware Requirements

- Laptop or computer running Linux
- Webcam or compatible USB camera
- ESP32 development board
- 2.4-inch ILI9341 SPI TFT LCD, 240 × 320 resolution
- USB cable for connecting the ESP32 to the laptop
- Jumper wires for connecting the TFT to the ESP32

## Software and Libraries

The project uses the following technologies across its recognition and display pipeline:

- Python
- OpenCV for camera access and image processing
- MediaPipe for hand/body landmark extraction where required by the selected recognition pipeline
- TensorFlow/Keras for applicable trained models
- PyTorch for applicable model pipelines
- ONNX Runtime for the alphabet recognition pipeline
- pyserial for communication between Python and the ESP32
- TFT_eSPI for ESP32 TFT rendering
- Arduino-compatible ESP32 firmware

The dependencies required for each mode may differ.

## Project Structure

The repository contains the following launchers and project components:

```text
live-asl-translation/
├── models/
├── sign2text/
├── tests/
├── .venv/
├── main_livefeed.ipynb
├── test_tft_serial.py
├── run_sign2text.sh
├── run_sign2text_integrated.sh
├── run_sign2text_py310.sh
├── run_sign2text_words.sh
├── run_sign2text_alphabet.sh
└── README.md
```

Additional model files, configuration files, and supporting scripts may be present in the repository.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/luvk1412/sign-language-translator.git
```

Alternatively, if you already have your project repository cloned, navigate directly to your local project directory.

> **Note:** Use the GitHub URL of your actual project repository when cloning. The command above is an example repository URL from the related project exploration, not necessarily the remote for your current working directory.

### 2. Navigate to the Project Directory

```bash
cd ~/live-asl-translation
```

### 3. Activate the Python Virtual Environment

```bash
source .venv/bin/activate
```

The virtual environment keeps the project's Python dependencies separate from system-installed packages.

If the environment has not been created or its dependencies have not been installed, follow the dependency instructions associated with the relevant recognition pipeline.

### 4. Connect the Hardware

1. Connect the TFT display to the ESP32 using the configured SPI pins.
2. Connect the ESP32 to the laptop through USB.
3. Confirm that the board is detected as a serial device, typically `/dev/ttyUSB0` in the current setup.
4. Ensure the ESP32 has the serial receiver firmware uploaded.
5. Close Arduino IDE Serial Monitor or any other program currently using the same serial port.

## Running the Integrated Sign2Text Application

This is the main workflow for running live recognition and displaying the output on both the laptop and TFT.

Open a terminal and execute:

```bash
cd ~/live-asl-translation
source .venv/bin/activate
./run_sign2text_integrated.sh
```

### What Happens When You Run It?

1. The Python application starts.
2. The laptop camera captures live video.
3. The recognition pipeline processes the input.
4. The recognized output is displayed on the laptop.
5. Recognized text is sent over USB serial to the ESP32.
6. The ESP32 receives the text and renders it on the TFT.

**Important:** The ESP32 firmware normally only needs to be uploaded when it is first installed or when the firmware is changed. It does not need to be reflashed for every application run.

## Available Recognition Modes

The repository contains launchers for multiple recognition pipelines.

| Launcher | Purpose |
|---|---|
| `run_sign2text_integrated.sh` | Integrated live recognition with laptop and ESP32 TFT output |
| `run_sign2text_py310.sh` | GRU-based gloss recognition pipeline |
| `run_sign2text_words.sh` | LSTM-based word recognition pipeline |
| `run_sign2text_alphabet.sh` | ONNX-based alphabet/fingerspelling recognition pipeline |
| `run_sign2text.sh` | General Sign2Text launcher; behavior depends on its configuration |

To run a standalone mode, navigate to the project directory, activate the appropriate environment, and execute its launcher. Some pipelines may require a different Python environment or dependency configuration.

For the complete laptop-to-TFT experience, use `run_sign2text_integrated.sh`.

## ESP32 and TFT Configuration

The TFT uses the TFT_eSPI library. The working display configuration established for the current hardware is:

```cpp
#define ILI9341_DRIVER

#define TFT_MISO 19
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_CS   27
#define TFT_DC   22
#define TFT_RST  4
```

These settings belong in the TFT_eSPI user setup configuration used by the ESP32 firmware.

The ESP32 firmware initializes the display, waits for serial input, and updates the display when it receives recognized text.

The current setup uses:

- **Display controller:** ILI9341
- **Display resolution:** 240 × 320
- **Communication:** SPI between ESP32 and TFT
- **Laptop-to-ESP32 connection:** USB serial
- **Serial baud rate:** 115200

Do not change the pin configuration unless the physical wiring or hardware configuration is also changed.

## Serial Communication

The Python application communicates with the ESP32 through a serial connection.

The current hardware setup uses:

```text
Device: /dev/ttyUSB0
Baud rate: 115200
Message format: UTF-8 text terminated by a newline
```

The ESP32 reads incoming text, removes surrounding whitespace, and updates the TFT when a non-empty message arrives.

The file `test_tft_serial.py` is available as a serial communication test utility.

## Troubleshooting

### 1. Serial Port Is Busy

**Symptom:** The Python application reports a serial connection error such as `Errno 16: Device or resource busy`.

**Solution:**

Close Arduino IDE Serial Monitor or any other application using `/dev/ttyUSB0`.

To identify processes using the port:

```bash
sudo fuser -v /dev/ttyUSB0
```

Close the application that owns the port and launch the integrated application again.

### 2. TFT Does Not Display Text

Check the following:

- The ESP32 is powered and connected.
- The TFT wiring matches the configured pin definitions.
- The TFT_eSPI setup uses the ILI9341 driver.
- The ESP32 firmware is installed.
- The Python application is sending newline-terminated messages to the correct serial device.

### 3. Camera Does Not Open

Check that the webcam is connected and available to the operating system.

On Linux, camera devices commonly appear under `/dev/video*`. Close other applications that may be using the camera and try again.

### 4. Model or Dependency Errors

Confirm that the correct virtual environment is active and that the required model files exist.

Some recognition pipelines use different frameworks and may require different Python versions or dependencies. Use the launcher appropriate for the model you intend to run.

### 5. Application Stops or Closes

Check the terminal output for Python errors, model-loading failures, camera initialization errors, or serial connection problems.

Use the integrated launcher for the complete laptop-to-TFT workflow.

## Development and Testing

The project has been tested across several parts of the recognition pipeline, including model loading, prediction distribution generation, camera capture, landmark extraction, and serial communication.

The ESP32/TFT display was also tested independently before being integrated with live recognition.

The following files may be useful during development:

- `test_tft_serial.py` — tests sending text from Python to the TFT through the ESP32.
- `tests/` — contains project tests.
- `main_livefeed.ipynb` — notebook for live-feed experimentation.

## Future Improvements

Potential extensions include:

- Better formatting of longer recognized sentences on the TFT.
- Automatic line wrapping and scrolling.
- Improved recognition stability and confidence handling.
- Better handling of mirrored camera input.
- Integration with additional embedded hardware.
- Further optimization of model inference and camera performance.

## License

Add the appropriate license for this repository if you intend to distribute or reuse the project publicly.

## Acknowledgements

This project combines computer vision, machine learning, embedded systems, and hardware integration to explore real-time sign language recognition and text display.


