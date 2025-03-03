import json
import time

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, AsyncMock

import numpy as np

from src.train_detector import TrainDetector
from src.tg import TelegramBot


@pytest.fixture
def integrated_system():
    """Setup the full system with all components for integration testing."""
    data_file = Path("test_integration_detections.json")

    # Create a real TelegramBot instance
    telegram_bot = TelegramBot("test_token", "test_chat_id", data_file)

    # Create the TrainDetector with the real bot
    with patch("cv2.VideoCapture") as mock_capture:
        mock_capture.return_value.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        mock_capture.return_value.set.return_value = True

        detector = TrainDetector(data_file, telegram_bot)

        # Mock file operations for both components
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("builtins.open", mock_open(read_data=json.dumps({"detections": []}))),
        ):
            # Return the integrated components
            yield detector, telegram_bot


def test_detection_workflow(integrated_system):
    """Test the complete workflow from detection to notification."""
    detector, telegram_bot = integrated_system

    # Mock the methods we want to verify
    telegram_bot.send_message = Mock(return_value=True)
    telegram_bot.send_photo = Mock(return_value=True)
    detector.load_data = Mock()
    detector.save_data = Mock()
    detector.data = {"detections": []}

    # Mock image saving and cleanup
    with patch("cv2.imwrite") as mock_imwrite, patch("os.remove") as mock_remove:
        # Test the detection workflow
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.handle_detection(
            detected_count=1,
            left_motion=1000,
            right_motion=200,
            direction="Left to Right",
            left_occupied=True,
            right_occupied=False,
            frame=frame,
        )

        # Verify integration points
        detector.load_data.assert_called_once()
        detector.save_data.assert_called_once()
        assert len(detector.data["detections"]) == 1

        # Verify Telegram notifications
        telegram_bot.send_message.assert_called_once()
        telegram_bot.send_photo.assert_called_once()

        # Verify image handling
        mock_imwrite.assert_called_once()
        mock_remove.assert_called_once()


@pytest.mark.asyncio
async def test_stats_generation_with_telegram(integrated_system):
    """Test the integration between stats and telegram bot."""
    detector, telegram_bot = integrated_system

    # Create sample data
    sample_data = {
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

    # Setup mocks
    update = AsyncMock()
    context = AsyncMock()
    update.message.reply_text = AsyncMock()

    # Mock file operations
    with patch("builtins.open", mock_open(read_data=json.dumps(sample_data))):
        # Test the stats generation via Telegram command (using await instead of asyncio.run)
        await telegram_bot.handle_stats(update, context)

        # Verify the stats were sent via Telegram
        update.message.reply_text.assert_awaited_once()

        # Check content of the message
        args, _ = update.message.reply_text.call_args
        message = args[0]

        # Verify key statistics are included
        assert "Train Statistics Report" in message
        assert "Today" in message
        assert "Last 7 Days" in message
        assert "Weekday Averages" in message


@pytest.mark.asyncio
async def test_detection_deletion_workflow(integrated_system):
    """Test the integration between detector data and telegram deletion."""
    detector, telegram_bot = integrated_system

    # Create sample data
    sample_data = {
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

    # Mock button press
    update = AsyncMock()
    update.callback_query = AsyncMock()
    update.callback_query.data = "delete_2025-02-27 10:00:00"
    context = AsyncMock()

    # Mock message sending
    telegram_bot.send_message = Mock(return_value=True)

    # Mock file operations with side_effect to handle reading then writing
    read_mock = mock_open(read_data=json.dumps(sample_data))
    read_mock.return_value.read.return_value = json.dumps(sample_data)
    write_mock = mock_open()

    # Setup mocks for file operations
    with patch("builtins.open", read_mock), patch("json.dump") as mock_json_dump:
        # Test the button handler (using await instead of asyncio.run)
        await telegram_bot.handle_button(update, context)

        # Verify the callback was answered
        update.callback_query.answer.assert_awaited_once()

        # Verify message was sent
        telegram_bot.send_message.assert_called_with(
            "❌ Marked as false detection: 2025-02-27 10:00:00"
        )

        # Verify data was modified and saved
        mock_json_dump.assert_called_once()
        args, _ = mock_json_dump.call_args
        modified_data = args[0]

        # Check that the detection was removed
        assert len(modified_data["detections"]) == 1
        assert modified_data["detections"][0]["timestamp"] == "2025-02-27 15:00:00"


def test_train_detector_detection_logic(integrated_system):
    """Test the train detection logic without running the infinite loop."""
    detector, telegram_bot = integrated_system

    # Setup detector for testing
    detector.motion_threshold = 500
    detector.detection_cooldown = 0  # No cooldown for testing
    detector.state_change_cooldown = 0  # No cooldown for testing
    detector.handle_detection = Mock()

    # Create test frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Setup initial state
    detector.train_passing = False
    detector.last_detection_time = 0

    # Create ROI analyzers for testing
    left_gray_empty = np.zeros((100, 100), dtype=np.uint8)
    right_gray_empty = np.zeros((100, 100), dtype=np.uint8)
    left_gray_motion = np.zeros((100, 100), dtype=np.uint8)
    # Add a white square to simulate more motion
    left_gray_motion[20:80, 20:80] = 255
    right_gray_motion = np.zeros((100, 100), dtype=np.uint8)
    # Add a white square to simulate more motion
    right_gray_motion[20:80, 20:80] = 255

    # Step 1: Test with no motion
    # Since there's no motion, left_occupied and right_occupied should remain False
    left_motion, right_motion = detector.process_motion(
        left_gray_empty, right_gray_empty, left_gray_empty, right_gray_empty
    )

    assert left_motion < detector.motion_threshold
    assert right_motion < detector.motion_threshold

    # Step 2: Test with left motion (train coming from left)
    # This should set left_occupied to True and leave right_occupied as False
    prev_left = left_gray_empty
    prev_right = right_gray_empty

    left_motion, right_motion = detector.process_motion(
        left_gray_motion, right_gray_empty, prev_left, prev_right
    )

    assert left_motion > detector.motion_threshold
    assert right_motion < detector.motion_threshold

    # Manually test the detection logic similar to what happens in the run loop
    current_time = time.time()
    detector.last_state_change = current_time - 10  # Ensure we're past the cooldown

    # Set the state based on motion (what would happen in the run loop)
    left_occupied = left_motion > detector.motion_threshold
    right_occupied = right_motion > detector.motion_threshold

    assert left_occupied == True
    assert right_occupied == False

    # Simulate a train detection
    if left_occupied != right_occupied:  # One ROI occupied, other empty
        direction = "Left to Right" if left_occupied else "Right to Left"

        # Only detect if enough time has passed
        if current_time - detector.last_detection_time > detector.detection_cooldown:
            detector.last_detection_time = current_time
            detector.handle_detection(
                1, left_motion, right_motion, direction, left_occupied, right_occupied, frame
            )

    # Check that detection happened with correct parameters
    detector.handle_detection.assert_called_once()
    args, _ = detector.handle_detection.call_args
    assert args[3] == "Left to Right"  # Direction argument
    assert args[4] == True  # left_occupied
    assert args[5] == False  # right_occupied

    # Reset for the next test
    detector.handle_detection.reset_mock()

    # Step 3: Test with right motion (train coming from right)
    left_motion, right_motion = detector.process_motion(
        left_gray_empty, right_gray_motion, prev_left, prev_right
    )

    assert left_motion < detector.motion_threshold
    assert right_motion > detector.motion_threshold

    # Update the state
    left_occupied = left_motion > detector.motion_threshold
    right_occupied = right_motion > detector.motion_threshold

    assert left_occupied is False
    assert right_occupied is True

    # Simulate another train detection
    if left_occupied != right_occupied:  # One ROI occupied, other empty
        direction = "Left to Right" if left_occupied else "Right to Left"

        # Always detect in test since we reset the time
        detector.last_detection_time = 0
        current_time = time.time()
        if current_time - detector.last_detection_time > detector.detection_cooldown:
            detector.last_detection_time = current_time
            detector.handle_detection(
                2, left_motion, right_motion, direction, left_occupied, right_occupied, frame
            )

    # Check that detection happened with correct parameters
    detector.handle_detection.assert_called_once()
    args, _ = detector.handle_detection.call_args
    assert args[3] == "Right to Left"  # Direction should now be Right to Left
    assert args[4] is False  # left_occupied
    assert args[5] is True  # right_occupied
