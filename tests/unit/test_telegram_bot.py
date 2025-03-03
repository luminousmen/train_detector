import json
from pathlib import Path
from unittest.mock import patch, mock_open, AsyncMock

import pytest

from src.tg import TelegramBot


@pytest.fixture
def sample_data():
    return {
        "detections": [
            {
                "timestamp": "2025-02-27 10:00:00",
                "left_motion": 1000,
                "right_motion": 200,
                "direction": "Left to Right",
                "left_occupied": True,
                "right_occupied": False,
            },
            {
                "timestamp": "2025-02-27 15:00:00",
                "left_motion": 300,
                "right_motion": 1200,
                "direction": "Right to Left",
                "left_occupied": False,
                "right_occupied": True,
            },
        ]
    }


@pytest.fixture
def telegram_bot():
    return TelegramBot("test_token", "test_chat", Path("test_detections.json"))


@pytest.mark.asyncio
async def test_send_message(telegram_bot):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"ok": True}
        success = telegram_bot.send_message("Test message")
        assert success is True
        mock_post.assert_called_once()


def test_delete_detection(telegram_bot, sample_data):
    json_data = json.dumps(sample_data)

    # Mock file operations with side_effect to handle read-then-write
    def open_side_effect(*args, **kwargs):
        if args[1] == "r":
            return mock_open(read_data=json_data)()
        elif args[1] == "w":
            return mock_open()()
        raise ValueError("Unexpected file mode")

    with patch("builtins.open", side_effect=open_side_effect) as mock_file:
        mock_file.return_value.read.return_value = json_data
        success = telegram_bot.delete_detection("2025-02-27 10:00:00")
        assert success is True, f"Delete detection failed: expected True, got {success}"


@pytest.mark.asyncio
async def test_handle_stats(telegram_bot, sample_data):
    update = AsyncMock()
    context = AsyncMock()
    update.message.reply_text = AsyncMock()
    with patch("builtins.open", mock_open(read_data=json.dumps(sample_data))):
        await telegram_bot.handle_stats(update, context)
        update.message.reply_text.assert_awaited_once()
        args, _ = update.message.reply_text.call_args
        assert "Today" in args[0]
        assert "Last 7 Days" in args[0]


@pytest.mark.asyncio
async def test_handle_button(telegram_bot, sample_data):
    update = AsyncMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "delete_2025-02-27 10:00:00"
    context = AsyncMock()

    # Define the JSON data as a string
    json_data = json.dumps(sample_data)

    # Mock file operations with side_effect
    def open_side_effect(*args, **kwargs):
        if args[1] == "r":
            return mock_open(read_data=json_data)()
        elif args[1] == "w":
            return mock_open()()
        raise ValueError("Unexpected file mode")

    with patch("builtins.open", side_effect=open_side_effect) as mock_file:
        mock_file.return_value.read.return_value = json_data
        with patch.object(telegram_bot, "send_message") as mock_send:
            await telegram_bot.handle_button(update, context)
            update.callback_query.answer.assert_awaited_once()
            mock_send.assert_called_with("❌ Marked as false detection: 2025-02-27 10:00:00")


@pytest.mark.asyncio
async def test_handle_stats_empty_data(telegram_bot):
    update = AsyncMock()
    context = AsyncMock()
    update.message.reply_text = AsyncMock()

    empty_data = {"detections": []}

    with patch("builtins.open", mock_open(read_data=json.dumps(empty_data))):
        await telegram_bot.handle_stats(update, context)
        update.message.reply_text.assert_awaited_once()
        args, _ = update.message.reply_text.call_args
        assert "Today" in args[0]
        assert "0 trains" in args[0]


def test_send_photo(telegram_bot):
    photo_path = "test_photo.jpg"
    caption = "Test caption"
    timestamp = "2025-02-28 12:00:00"

    # Mock file operations and HTTP request
    mock_file = mock_open(read_data=b"test image data")

    with patch("builtins.open", mock_file), patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"ok": True}
        success = telegram_bot.send_photo(photo_path, caption, timestamp)

        assert success is True
        mock_post.assert_called_once()

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["chat_id"] == telegram_bot.chat_id
        assert kwargs["data"]["caption"] == caption
        assert "delete_" in kwargs["data"]["reply_markup"]


def test_send_photo_failure(telegram_bot):
    with (
        patch("builtins.open", mock_open(read_data=b"test image data")),
        patch("requests.post") as mock_post
    ):
        # Configure mock to raise an exception
        mock_post.side_effect = Exception("Connection error")

        success = telegram_bot.send_photo("test.jpg", "caption", "timestamp")

        assert success is False


def test_delete_detection_missing_timestamp(telegram_bot, sample_data):
    with patch("builtins.open", mock_open(read_data=json.dumps(sample_data))):
        success = telegram_bot.delete_detection("2025-02-28 12:34:56")  # Timestamp not in data
        assert success is False


def test_delete_detection_with_exception(telegram_bot):
    with patch("builtins.open", side_effect=Exception("File error")):
        success = telegram_bot.delete_detection("2025-02-27 10:00:00")
        assert success is False


@pytest.mark.asyncio
async def test_error_handler(telegram_bot):
    update = AsyncMock()
    context = AsyncMock()
    context.error = Exception("Test error")

    # No assertion needed, we just want to ensure it doesn't raise an exception
    await telegram_bot.error_handler(update, context)

    # Test with None update
    await telegram_bot.error_handler(None, context)
