import os
import time
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, call

import pytest
import numpy as np

from src.train_detector import TrainDetector


@pytest.fixture
def test_data_file():
    data_file = Path("test_detections.json")
    if data_file.exists():
        os.remove(data_file)

    yield data_file

    if data_file.exists():
        os.remove(data_file)


@pytest.fixture
def telegram_bot_mock():
    return Mock(spec=["send_message", "send_photo"])


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
def train_detector(telegram_bot_mock, test_data_file):
    with patch("cv2.VideoCapture") as mock_capture:
        mock_capture.return_value.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        detector = TrainDetector(
            data_file=Path("test_detections.json"),
            telegram_bot=telegram_bot_mock,
            motion_threshold=500,
            detection_cooldown=180,
            state_change_cooldown=5,
        )
        yield detector
        if hasattr(detector, "cap"):
            detector.cap.release()


def test_train_detector_init(train_detector, telegram_bot_mock):
    assert train_detector.data_file == Path("test_detections.json")
    assert train_detector.telegram_bot == telegram_bot_mock
    assert train_detector.motion_threshold == 500
    assert train_detector.detection_cooldown == 180
    assert train_detector.state_change_cooldown == 5
    assert hasattr(train_detector, "cap")
    assert train_detector.roi_left is not None
    assert train_detector.roi_right is not None

    telegram_bot_mock.send_message.assert_called_once_with("Train detector starting up...")


def test_load_data_existing(train_detector, sample_data):
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(sample_data))),
        patch("pathlib.Path.exists", return_value=True),
    ):
        train_detector.load_data()
        assert len(train_detector.data["detections"]) == 2


def test_load_data_new(train_detector):
    with patch("builtins.open", mock_open()), patch("pathlib.Path.exists", return_value=False):
        train_detector.load_data()
        assert train_detector.data == {"detections": []}


def test_analyze_roi(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi = [(100, 100), (200, 200)]
    result = train_detector.analyze_roi(frame, roi)
    assert result.shape == (100, 100)
    assert result.dtype == np.uint8


def test_process_motion(train_detector):
    prev_left = np.zeros((100, 100), dtype=np.uint8)
    prev_right = np.zeros((100, 100), dtype=np.uint8)

    # Test 1: No motion
    left_gray = np.zeros((100, 100), dtype=np.uint8)
    right_gray = np.zeros((100, 100), dtype=np.uint8)
    left_motion, right_motion = train_detector.process_motion(
        left_gray, right_gray, prev_left, prev_right
    )
    assert left_motion == 0
    assert right_motion == 0

    # Test 2: Motion in left ROI
    left_gray_motion = np.zeros((100, 100), dtype=np.uint8)
    left_gray_motion[20:80, 20:80] = 255  # Large white square
    left_motion, right_motion = train_detector.process_motion(
        left_gray_motion, right_gray, prev_left, prev_right
    )
    assert left_motion > 0
    assert right_motion == 0

    # Test 3: Motion in right ROI
    right_gray_motion = np.zeros((100, 100), dtype=np.uint8)
    right_gray_motion[20:80, 20:80] = 255  # Large white square
    left_motion, right_motion = train_detector.process_motion(
        left_gray, right_gray_motion, prev_left, prev_right
    )
    assert left_motion == 0
    assert right_motion > 0

    # Test 4: Motion in both ROIs
    left_motion, right_motion = train_detector.process_motion(
        left_gray_motion, right_gray_motion, prev_left, prev_right
    )
    assert left_motion > 0
    assert right_motion > 0


def test_draw_status(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    left_motion = 500
    right_motion = 700

    # Call the method
    train_detector.draw_status(frame, left_motion, right_motion)

    # Check that drawing was done (some pixels should be non-zero)
    assert np.count_nonzero(frame) > 0


def test_detect_train_left_to_right(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = time.time()
    train_detector.last_detection_time = 0  # Ensure cooldown is satisfied

    detection_made, new_count, detection_time = train_detector.detect_train(
        current_time=current_time,
        left_motion=600,  # Above threshold
        right_motion=100,  # Below threshold
        prev_left_state=False,  # State change
        prev_right_state=False,
        frame=frame,
        detected_count=5,
    )

    assert detection_made is True
    assert new_count == 6
    assert detection_time == current_time
    train_detector.telegram_bot.send_message.assert_called()
    assert train_detector.telegram_bot.send_message.call_count == 2


def test_detect_train_right_to_left(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = time.time()
    train_detector.last_detection_time = 0  # Ensure cooldown is satisfied

    detection_made, new_count, detection_time = train_detector.detect_train(
        current_time=current_time,
        left_motion=100,  # Below threshold
        right_motion=600,  # Above threshold
        prev_left_state=False,
        prev_right_state=False,  # State change
        frame=frame,
        detected_count=5,
    )

    assert detection_made is True
    assert new_count == 6
    assert detection_time == current_time


def test_detect_train_cooldown(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = time.time()
    train_detector.last_detection_time = current_time - 5  # Recent detection
    train_detector.detection_cooldown = 10  # 10 second cooldown

    detection_made, new_count, detection_time = train_detector.detect_train(
        current_time=current_time,
        left_motion=600,  # Above threshold
        right_motion=100,  # Below threshold
        prev_left_state=False,  # State change
        prev_right_state=False,
        frame=frame,
        detected_count=5,
    )

    assert detection_made is False
    assert new_count == 5  # Unchanged
    assert detection_time == current_time - 5  # Unchanged


def test_detect_train_no_state_change(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = time.time()
    train_detector.last_detection_time = 0  # Ensure cooldown is satisfied

    detection_made, new_count, detection_time = train_detector.detect_train(
        current_time=current_time,
        left_motion=600,  # Above threshold
        right_motion=100,  # Below threshold
        prev_left_state=True,  # No state change (already occupied)
        prev_right_state=False,
        frame=frame,
        detected_count=5,
    )

    # Verify no detection was made due to lack of state change
    assert detection_made is False
    assert new_count == 5  # Unchanged
    assert detection_time == 0  # Unchanged


def test_detect_train_both_occupied(train_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = time.time()
    train_detector.last_detection_time = 0  # Ensure cooldown is satisfied

    detection_made, new_count, detection_time = train_detector.detect_train(
        current_time=current_time,
        left_motion=600,  # Above threshold
        right_motion=600,  # Above threshold
        prev_left_state=False,  # State change
        prev_right_state=False,  # State change
        frame=frame,
        detected_count=5,
    )

    # Verify no detection was made due to both sides being occupied
    assert detection_made is False
    assert new_count == 5  # Unchanged
    assert detection_time == 0  # Unchanged


@patch("os.remove")
@patch("cv2.imwrite")
def test_handle_detection(mock_imwrite, mock_remove, train_detector):
    detected_count = 5
    left_motion = 1000
    right_motion = 200
    direction = "Left to Right"
    left_occupied = True
    right_occupied = False
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Mock methods
    train_detector.load_data = Mock()
    train_detector.save_data = Mock()
    train_detector.data = {"detections": []}

    # Call the method
    train_detector.handle_detection(
        detected_count, left_motion, right_motion, direction, left_occupied, right_occupied, frame
    )

    # Verify method calls
    train_detector.load_data.assert_called_once()
    train_detector.save_data.assert_called_once()
    assert len(train_detector.data["detections"]) == 1
    mock_imwrite.assert_called_once()
    train_detector.telegram_bot.send_message.assert_called()  # Just check it was called
    assert (
        train_detector.telegram_bot.send_message.call_count == 2
    )  # Check it was called exactly twice
    train_detector.telegram_bot.send_photo.assert_called_once()
    mock_remove.assert_called_once()

    detection = train_detector.data["detections"][0]
    assert detection["timestamp"] == current_time
    assert detection["left_motion"] == 1000
    assert detection["right_motion"] == 200
    assert detection["direction"] == "Left to Right"
    assert detection["left_occupied"] is True
    assert detection["right_occupied"] is False


def test_get_frame(train_detector):
    with patch.object(train_detector.cap, "read") as mock_read:
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_read.return_value = (True, test_frame)
        frame = train_detector.get_frame()
        assert frame is not None
        assert np.array_equal(frame, test_frame)

    # Test failed frame retrieval
    with patch.object(train_detector.cap, "read") as mock_read:
        mock_read.return_value = (False, None)
        frame = train_detector.get_frame()
        assert frame is None


def test_initialize_detector_state(train_detector):
    state = train_detector._initialize_detector_state()

    assert state["prev_left"] is None
    assert state["prev_right"] is None
    assert state["detected_count"] == 0
    assert state["left_occupied"] is False
    assert state["right_occupied"] is False
    assert isinstance(state["last_state_change"], float)
    # The last_state_change should be approximately now
    assert abs(state["last_state_change"] - time.time()) < 1


def test_check_state_changes_cooldown_active(train_detector):
    current_time = time.time()
    initial_state = {
        "prev_left": np.zeros((20, 20)),
        "prev_right": np.zeros((20, 20)),
        "detected_count": 3,
        "left_occupied": True,
        "right_occupied": False,
        "last_state_change": current_time - 0.5,  # Less than cooldown (1s)
    }
    frame = np.zeros((100, 100, 3))
    left_motion = 600
    right_motion = 100
    new_left = np.ones((20, 20))
    new_right = np.ones((20, 20))

    with patch.object(
        train_detector, "detect_train", wraps=train_detector.detect_train
    ) as spy_method:
        result_state = train_detector._check_state_changes(
            initial_state, frame, left_motion, right_motion, new_left, new_right
        )

        assert result_state["prev_left"] is new_left
        assert result_state["prev_right"] is new_right
        assert result_state["detected_count"] == 3
        assert result_state["left_occupied"] is True
        assert result_state["right_occupied"] is False
        assert result_state["last_state_change"] == initial_state["last_state_change"]

        spy_method.assert_not_called()


def test_process_frame_no_frame(train_detector):
    initial_state = {
        "prev_left": np.zeros((10, 10)),
        "prev_right": np.zeros((10, 10)),
        "detected_count": 5,
        "left_occupied": True,
        "right_occupied": False,
        "last_state_change": time.time() - 10
    }

    # Configure mock to return None for get_frame
    with patch.object(train_detector, "get_frame") as mock_get_frame:
        mock_get_frame.return_value = None

        result_state = train_detector._process_frame(initial_state)

        assert result_state is initial_state  # Same object reference
        mock_get_frame.assert_called_once()


def test_process_frame_initialize_prev_frames(train_detector):
    initial_state = {
        "prev_left": None,
        "prev_right": None,
        "detected_count": 0,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": time.time()
    }

    # Test frame and ROIs
    test_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    left_roi = np.zeros((100, 100), dtype=np.uint8)
    right_roi = np.ones((100, 100), dtype=np.uint8)

    # Configure mocks
    with patch.object(train_detector, "get_frame") as mock_get_frame, \
            patch.object(train_detector, "analyze_roi") as mock_analyze_roi, \
            patch.object(train_detector, "_check_state_changes") as mock_check_changes:
        mock_get_frame.return_value = test_frame
        mock_analyze_roi.side_effect = [left_roi, right_roi]

        # Call the method
        result_state = train_detector._process_frame(initial_state)

        # Verify
        assert result_state is initial_state  # Same object, modified
        assert result_state["prev_left"] is left_roi
        assert result_state["prev_right"] is right_roi
        assert result_state["detected_count"] == 0  # Unchanged

        mock_get_frame.assert_called_once()
        mock_analyze_roi.assert_has_calls([
            call(test_frame, train_detector.roi_left),
            call(test_frame, train_detector.roi_right)
        ])


def test_process_frame_normal_operation(train_detector):
    prev_left = np.zeros((20, 20), dtype=np.uint8)
    prev_right = np.zeros((20, 20), dtype=np.uint8)
    initial_state = {
        "prev_left": prev_left,
        "prev_right": prev_right,
        "detected_count": 1,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": time.time() - 10
    }
    test_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    left_roi = np.ones((20, 20), dtype=np.uint8)
    right_roi = np.ones((20, 20), dtype=np.uint8)

    # Expected updated state
    updated_state = {"detected_count": 2, "some_new_key": True}

    # Configure mocks
    with patch.object(train_detector, "get_frame") as mock_get_frame, \
            patch.object(train_detector, "analyze_roi") as mock_analyze_roi, \
            patch.object(train_detector, "process_motion") as mock_process_motion, \
            patch.object(train_detector, "draw_status") as mock_draw_status, \
            patch.object(train_detector, "_check_state_changes") as mock_check_changes:
        mock_get_frame.return_value = test_frame
        mock_analyze_roi.side_effect = [left_roi, right_roi]
        mock_process_motion.return_value = (600, 100)  # Motion values
        mock_check_changes.return_value = updated_state

        # Call the method
        result_state = train_detector._process_frame(initial_state)

        assert result_state is updated_state

        mock_get_frame.assert_called_once()
        mock_analyze_roi.assert_has_calls([
            call(test_frame, train_detector.roi_left),
            call(test_frame, train_detector.roi_right)
        ])
        mock_process_motion.assert_called_once_with(
            left_roi, right_roi, prev_left, prev_right
        )
        mock_draw_status.assert_called_once_with(test_frame, 600, 100)
        mock_check_changes.assert_called_once_with(
            initial_state, test_frame, 600, 100, left_roi, right_roi
        )


def test_process_frame_with_modified_state(train_detector):
    prev_left = np.zeros((20, 20), dtype=np.uint8)
    prev_right = np.zeros((20, 20), dtype=np.uint8)
    initial_state = {
        "prev_left": prev_left,
        "prev_right": prev_right,
        "detected_count": 1,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": time.time() - 10,
        "custom_field": "test value",  # Additional field
        "statistics": {"total": 42}  # Nested field
    }

    # Test frame and ROIs
    test_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    left_roi = np.ones((20, 20), dtype=np.uint8)
    right_roi = np.ones((20, 20), dtype=np.uint8)

    # Configure mocks - _check_state_changes will preserve custom fields
    def mock_check_fn(state, frame, l_motion, r_motion, l_gray, r_gray):
        state["detected_count"] = 2  # Update a field
        return state  # Return the same modified state

    with patch.object(train_detector, "get_frame") as mock_get_frame, \
            patch.object(train_detector, "analyze_roi") as mock_analyze_roi, \
            patch.object(train_detector, "process_motion") as mock_process_motion, \
            patch.object(train_detector, "draw_status") as mock_draw_status, \
            patch.object(train_detector, "_check_state_changes", side_effect=mock_check_fn):
        mock_get_frame.return_value = test_frame
        mock_analyze_roi.side_effect = [left_roi, right_roi]
        mock_process_motion.return_value = (600, 100)

        result_state = train_detector._process_frame(initial_state)
        assert result_state is initial_state
        assert result_state["custom_field"] == "test value"
        assert result_state["statistics"]["total"] == 42
        assert result_state["detected_count"] == 2


def test_check_state_changes_updates_prev_frames(train_detector):
    current_time = time.time()
    initial_state = {
        "prev_left": np.zeros((20, 20)),
        "prev_right": np.zeros((20, 20)),
        "detected_count": 3,
        "left_occupied": True,
        "right_occupied": False,
        "last_state_change": current_time - 0.5,  # Recent change (within cooldown)
    }
    frame = np.zeros((100, 100, 3))
    left_motion = 600
    right_motion = 100
    new_left = np.ones((20, 20))
    new_right = np.ones((20, 20))

    with patch("time.time", return_value=current_time):
        result_state = train_detector._check_state_changes(
            initial_state, frame, left_motion, right_motion, new_left, new_right
        )

    assert result_state is initial_state  # Same object
    assert result_state["prev_left"] is new_left
    assert result_state["prev_right"] is new_right
    assert result_state["detected_count"] == 3
    assert result_state["left_occupied"] is True
    assert result_state["right_occupied"] is False


def test_check_state_changes_cooldown_period(train_detector):
    current_time = time.time()
    train_detector.state_change_cooldown = 1.0

    initial_state = {
        "prev_left": np.zeros((20, 20)),
        "prev_right": np.zeros((20, 20)),
        "detected_count": 3,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": current_time - 0.5,  # 0.5 seconds ago (within cooldown)
    }

    frame = np.zeros((100, 100, 3))
    left_motion = 600  # Above threshold
    right_motion = 100  # Below threshold
    new_left = np.ones((20, 20))
    new_right = np.ones((20, 20))

    with patch("time.time", return_value=current_time), \
            patch.object(train_detector, "detect_train") as mock_detect_train:
        result_state = train_detector._check_state_changes(
            initial_state, frame, left_motion, right_motion, new_left, new_right
        )
        mock_detect_train.assert_not_called()
        assert result_state["detected_count"] == 3
        assert result_state["last_state_change"] == current_time - 0.5
        assert result_state["left_occupied"] is False
        assert result_state["right_occupied"] is False


def test_check_state_changes_after_cooldown_no_detection(train_detector):
    current_time = time.time()
    train_detector.motion_threshold = 500
    train_detector.state_change_cooldown = 1.0  # 1 second cooldown

    initial_state = {
        "prev_left": np.zeros((20, 20)),
        "prev_right": np.zeros((20, 20)),
        "detected_count": 3,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": current_time - 2.0,  # 2 seconds ago (past cooldown)
    }
    frame = np.zeros((100, 100, 3))
    left_motion = 600  # Above threshold
    right_motion = 100  # Below threshold
    new_left = np.ones((20, 20))
    new_right = np.ones((20, 20))

    with patch("time.time", return_value=current_time), \
            patch.object(train_detector, "detect_train") as mock_detect_train:
        # Configure detect_train to return no detection
        mock_detect_train.return_value = (False, 3, 0)

        result_state = train_detector._check_state_changes(
            initial_state, frame, left_motion, right_motion, new_left, new_right
        )
        mock_detect_train.assert_called_once_with(
            current_time, left_motion, right_motion, False, False, frame, 3
        )

        assert result_state["detected_count"] == 3  # Unchanged
        assert result_state["last_state_change"] == current_time - 2.0  # Unchanged
        assert result_state["left_occupied"] is True  # Updated (left_motion > threshold)
        assert result_state["right_occupied"] is False  # Updated (right_motion < threshold)


def test_check_state_changes_with_detection(train_detector):
    current_time = time.time()
    train_detector.motion_threshold = 500
    train_detector.state_change_cooldown = 1.0

    initial_state = {
        "prev_left": np.zeros((20, 20)),
        "prev_right": np.zeros((20, 20)),
        "detected_count": 3,
        "left_occupied": False,
        "right_occupied": False,
        "last_state_change": current_time - 2.0,  # Past cooldown
    }

    frame = np.zeros((100, 100, 3))
    left_motion = 600  # Above threshold
    right_motion = 100  # Below threshold
    new_left = np.ones((20, 20))
    new_right = np.ones((20, 20))

    with patch("time.time", return_value=current_time), \
            patch.object(train_detector, "detect_train") as mock_detect_train:
        # Configure detect_train to return a detection
        mock_detect_train.return_value = (True, 4, current_time)

        result_state = train_detector._check_state_changes(
            initial_state, frame, left_motion, right_motion, new_left, new_right
        )

        mock_detect_train.assert_called_once()
        assert result_state["detected_count"] == 4  # Updated
        assert result_state["last_state_change"] == current_time  # Updated
        assert result_state["left_occupied"] is True  # Updated based on motion
        assert result_state["right_occupied"] is False  # Updated based on motion
        assert train_detector.last_detection_time == current_time

