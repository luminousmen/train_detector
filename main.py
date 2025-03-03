import logging
import threading

from src.config import CHAT_ID, DATA_FILE, TELEGRAM_TOKEN
from src.tg import TelegramBot
from src.train_detector import TrainDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    # Create logger for main script
    logger = logging.getLogger(__name__)

    try:
        # Initialize Telegram Bot and Train Detector
        telegram_bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID, DATA_FILE)
        detector = TrainDetector(DATA_FILE, telegram_bot)

        # Start detector in a separate thread
        logger.info("Starting train detector thread...")
        bot_thread = threading.Thread(target=detector.run, daemon=True)
        bot_thread.start()

        logger.info("Starting Telegram bot...")
        telegram_bot.run()

    except Exception as e:
        logger.error(f"Critical error in train detector: {e}", exc_info=True)
