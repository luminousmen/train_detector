import json
import logging
from pathlib import Path

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes

from src.stats import _format_stats_message, generate_train_stats


class TelegramBot:
    def __init__(self, token: str, chat_id: str, data_file: Path) -> None:
        self.token = token
        self.chat_id = chat_id
        self.data_file = data_file
        self.telegram_url = f"https://api.telegram.org/bot{token}"
        self.app = ApplicationBuilder().token(token).build()

        # Add callback handler for button presses
        self.app.add_handler(CommandHandler("stats", self.handle_stats))
        self.app.add_handler(CallbackQueryHandler(self.handle_button))
        self.app.add_error_handler(self.error_handler)  # Add error handler

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle and log errors occurring during Telegram bot operations.

        :param update: Incoming update that caused the error
        :type update: Update
        :param context: Context for the current error
        :type context: ContextTypes.DEFAULT_TYPE
        """
        logging.error(f"Exception while handling an update: {context.error}")

        # Log the error
        if update:
            logging.error(f"Update {update} caused error {context.error}")
        else:
            logging.error(f"Error occurred: {context.error}")

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Process inline button interactions from Telegram messages.

        Handles 'False Detection' button press by removing the specified detection.

        :param update: Incoming update containing button interaction
        :type update: Update
        :param context: Context for the current interaction
        :type context: ContextTypes.DEFAULT_TYPE
        """
        query = update.callback_query
        await query.answer()

        # Extract timestamp from callback data
        timestamp = query.data.replace("delete_", "")

        if self.delete_detection(timestamp):
            self.send_message(f"❌ Marked as false detection: {timestamp}")
        else:
            self.send_message(f"Error: Could not delete detection: {timestamp}")

    def delete_detection(self, timestamp: str) -> bool:
        """
        Remove a specific train detection from the data file.

        :param timestamp: Timestamp of the detection to be deleted
        :type timestamp: str
        :return: True if detection was successfully deleted, False otherwise
        :rtype: bool
        """
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
            if len([d for d in data["detections"] if d["timestamp"] == timestamp]) > 0:
                logging.info(f"Detection found {timestamp}!")
                data["detections"] = [d for d in data["detections"] if d["timestamp"] != timestamp]
                with open(self.data_file, "w") as f:
                    json.dump(data, f, indent=2)
                return True
            return False
        except Exception as e:
            logging.error(f"Error deleting detection: {e}")
            return False

    def send_message(self, message: str) -> bool:
        """
        Send a text message to the configured Telegram chat.

        :param message: Message text to be sent
        :type message: str
        :return: True if message was sent successfully, False otherwise
        :rtype: bool
        """
        try:
            url = f"{self.telegram_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            response = requests.post(url, data=data, timeout=(30, 60))
            return response.json()["ok"]
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            return False

    def send_photo(self, photo_path: str, caption: str, timestamp: str) -> bool:
        """
        Send a photo to the Telegram chat with an inline 'False Detection' button.

        :param photo_path: File path of the photo to be sent
        :type photo_path: str
        :param caption: Caption for the photo
        :type caption: str
        :param timestamp: Timestamp associated with the detection
        :type timestamp: str
        :return: True if photo was sent successfully, False otherwise
        :rtype: bool
        """
        try:
            url = f"{self.telegram_url}/sendPhoto"

            keyboard = {
                "inline_keyboard": [
                    [{"text": "❌ False Detection", "callback_data": f"delete_{timestamp}"}]
                ]
            }

            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "reply_markup": json.dumps(keyboard),
                }
                response = requests.post(url, data=data, files=files, timeout=(60, 120))
            return response.json()["ok"]
        except Exception as e:
            logging.error(f"Error sending Telegram photo: {e}")
            return False

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle the /stats command to generate and send train detection statistics.

        :param update: Incoming update containing the stats command
        :type update: Update
        :param context: Context for the current interaction
        :type context: ContextTypes.DEFAULT_TYPE
        """
        try:
            stats = generate_train_stats(data_file=self.data_file)
            message = _format_stats_message(stats)
            await update.message.reply_text(message)
        except Exception as e:
            logging.error(f"Error generating stats: {e}", exc_info=True)
            await update.message.reply_text("Error generating statistics.")

    def run(self) -> None:
        """
        Start the Telegram bot and begin polling for updates.
        """
        self.app.run_polling()
