"""
Sign2Text Configuration Settings

Centralized configuration for all components. Modify these values
to adjust behavior without changing code.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "sign_classifier.pkl"

# Dataset path - configurable, defaults to the ASL alphabet training dataset
DATASET_DIR = Path("/home/akshay-sv/SignDB/asl_alphabet_train/asl_alphabet_train")

# Default paths for model, metadata, and logs (can be overridden via CLI)
DEFAULT_MODEL_DIR = MODELS_DIR
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_WARMUP_FRAMES = 5
CAMERA_MAX_READ_RETRIES = 3

# =============================================================================
# MEDIAPIPE HAND DETECTION SETTINGS
# =============================================================================
MAX_NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# =============================================================================
# FEATURE EXTRACTION SETTINGS
# =============================================================================
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
MAX_HANDS = 2
FEATURE_VECTOR_SIZE = FEATURES_PER_HAND * MAX_HANDS  # 126

WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9
EPSILON = 1e-6

# =============================================================================
# ML CLASSIFIER SETTINGS
# =============================================================================
CLASSIFIER_TYPE = "random_forest"  # "random_forest" or "svm"
RANDOM_FOREST_N_ESTIMATORS = 200
RANDOM_FOREST_MAX_DEPTH = None
RANDOM_FOREST_MIN_SAMPLES_SPLIT = 2
RANDOM_FOREST_MIN_SAMPLES_LEAF = 1
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"

TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# =============================================================================
# PREDICTION SETTINGS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.6
STABLE_FRAMES = 5  # Number of consecutive frames with same prediction for debouncing
DEBOUNCE_FRAMES = 5  # Alias for STABLE_FRAMES
MAX_SAME_SIGN_REPEAT = 1  # Only output sign once per stable detection
PREDICTION_COOLDOWN = 200  # Minimum ms between outputs of same sign

# =============================================================================
# TEXT BUFFER SETTINGS
# =============================================================================
SPACE_KEY = ord(" ")
BACKSPACE_KEY = 8
CLEAR_KEY = ord("c")
ENTER_KEY = 13

# =============================================================================
# SERIAL/UART COMMUNICATION SETTINGS
# =============================================================================
SERIAL_PORT = "/dev/ttyUSB0"  # Linux default; Windows: "COM3", macOS: "/dev/tty.usbserial-*"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1.0
SERIAL_WRITE_TIMEOUT = 1.0
LINE_ENDING = "\n"

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
WINDOW_NAME = "Sign2Text - Real-time Sign Recognition"
SHOW_FPS = True
SHOW_CONFIDENCE = True
SHOW_PREDICTION = True
SHOW_TEXT_BUFFER = True
MIRROR_FEED = False
SHOW_LANDMARKS = True  # Show hand landmark skeleton overlay

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================
CNN_INFERENCE_INTERVAL = 3  # Run CNN inference every N frames (1 = every frame)
ENABLE_TIMING_MEASUREMENTS = True  # Enable detailed timing measurements  # Show hand landmark skeleton overlay

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================
CNN_INFERENCE_INTERVAL = 3  # Run CNN inference every N frames (1 = every frame)
ENABLE_TIMING_MEASUREMENTS = True  # Enable detailed timing measurements

# =============================================================================
# LPC2148 FIRMWARE SETTINGS (PLACEHOLDERS - UPDATE WITH ACTUAL PINOUT)
# =============================================================================
# UART Configuration for LPC2148
LPC2148_UART_PORT = 0  # UART0 or UART1
LPC2148_UART_BAUD = 9600
LPC2148_UART_PIN_TX = 0  # P0.0 for UART0 TX, P0.8 for UART1 TX
LPC2148_UART_PIN_RX = 1  # P0.1 for UART0 RX, P0.9 for UART1 RX

# LCD Configuration for LPC2148 (update based on your board)
LPC2148_LCD_TYPE = "HD44780"  # Common LCD controller
LPC2148_LCD_RS_PIN = 16      # Register Select
LPC2148_LCD_EN_PIN = 17      # Enable
LPC2148_LCD_D4_PIN = 18      # Data 4
LPC2148_LCD_D5_PIN = 19      # Data 5
LPC2148_LCD_D6_PIN = 20      # Data 6
LPC2148_LCD_D7_PIN = 21      # Data 7
LPC2148_LCD_COLS = 16        # Columns (16 or 20)
LPC2148_LCD_ROWS = 2         # Rows (2 or 4)

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "sign2text.log"
