import os
from pathlib import Path

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
DATA_FILE = Path('detections.json')

# Detection configuration
MOTION_THRESHOLD = 600
DETECTION_COOLDOWN = 120  # 2 min cooldown once we see a train
STATE_CHANGE_COOLDOWN = 5  # Minimum seconds between state changes

# Camera configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"
