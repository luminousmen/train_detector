import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.config import CAMERA_HEIGHT, CAMERA_WIDTH, DATETIME_FORMAT


class TrainDetector:
    """
    Train detector using motion detection with dual region of interest (ROI) analysis.
    Detects trains passing by and determines their direction based on which ROI shows motion first.
    """

    def __init__(
        self,
        data_file: Path,
        telegram_bot: Any,
        motion_threshold: int = 600,
        detection_cooldown: int = 180,
        state_change_cooldown: int = 5,
    ):
        self.telegram_bot = telegram_bot
        self.data_file = data_file

        # Configuration
        self.motion_threshold = motion_threshold
        self.detection_cooldown = detection_cooldown
        self.state_change_cooldown = state_change_cooldown

        # State
        self.last_detection_time = 0
        self.train_passing = False
        self.data = {"detections": []}
        self.load_data()

        # Camera setup
        self._initialize_camera()

        # Notify start
        telegram_bot.send_message("Train detector starting up...")

    def _initialize_camera(self) -> None:
        """
        Sets up the camera capture device, configures resolution, and
        defines rectangular regions for motion detection.

        :raises RuntimeError: If camera initialization fails
        """
        logging.info("Initializing camera...")
        self.cap = cv2.VideoCapture(0)

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Get frame dimensions and set up ROIs
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not initialize camera")

        height, width = frame.shape[:2]
        square_size = min(width, height) // 4
        center_y = height // 2

        # Define regions of interest for left and right sides
        self.roi_left = [
            (width // 6 - square_size // 2, center_y - square_size // 2),  # Top-left
            (width // 6 + square_size // 2, center_y + square_size // 2),  # Bottom-right
        ]
        self.roi_right = [
            (5 * width // 6 - square_size // 2, center_y - square_size // 2),  # Top-left
            (5 * width // 6 + square_size // 2, center_y + square_size // 2),  # Bottom-right
        ]

        logging.info(f"Frame size: {width}x{height}")
        logging.info(f"Left ROI: {self.roi_left}")
        logging.info(f"Right ROI: {self.roi_right}")

    @staticmethod
    def _initialize_detector_state() -> dict:
        """
        Create initial state dictionary for the detector.

        :return: Initial detector state with default values
        :rtype: dict
        """
        return {
            "prev_left": None,
            "prev_right": None,
            "detected_count": 0,
            "left_occupied": False,
            "right_occupied": False,
            "last_state_change": time.time(),
        }

    def load_data(self) -> None:
        """
        Load previous detection data from JSON file.
        """
        if self.data_file.exists():
            with open(self.data_file) as f:
                logging.debug(f"Data file exists, loading {self.data_file}")
                self.data = json.load(f)

    def save_data(self) -> None:
        """
        Writes the current detection history to the specified data file.
        """
        with open(self.data_file, "w") as f:
            json.dump(self.data, f)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        :return: Captured video frame or None if capture fails
        :rtype: Optional[np.ndarray]
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def analyze_roi(self, frame: np.ndarray, roi: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extract and preprocess a Region of Interest (ROI) from the frame.

        :param frame: Input video frame
        :type frame: np.ndarray
        :param roi: Region of Interest coordinates [(top_left), (bottom_right)]
        :type roi: List[Tuple[int, int]]
        :return: Processed grayscale ROI
        :rtype: np.ndarray
        """
        x1, y1 = roi[0]
        x2, y2 = roi[1]
        roi_frame = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        return gray

    def process_motion(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        prev_left: np.ndarray,
        prev_right: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Calculate motion intensity for left and right ROIs.
        Computes pixel-level differences between current and previous frames.

        :param left_gray: Current grayscale left ROI frame
        :type left_gray: np.ndarray
        :param right_gray: Current grayscale right ROI frame
        :type right_gray: np.ndarray
        :param prev_left: Previous grayscale left ROI frame
        :type prev_left: np.ndarray
        :param prev_right: Previous grayscale right ROI frame
        :type prev_right: np.ndarray
        :return: Tuple of motion intensities for left and right ROIs
        :rtype: Tuple[int, int]
        """
        left_delta = cv2.absdiff(prev_left, left_gray)
        right_delta = cv2.absdiff(prev_right, right_gray)

        # Threshold
        left_thresh = cv2.threshold(left_delta, 25, 255, cv2.THRESH_BINARY)[1]
        right_thresh = cv2.threshold(right_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Count non-zero pixels
        return cv2.countNonZero(left_thresh), cv2.countNonZero(right_thresh)

    def draw_status(self, frame: np.ndarray, left_motion: int, right_motion: int) -> None:
        """
        Draws rectangles for ROIs and displays motion intensity values.

        :param frame: Input video frame to annotate
        :type frame: np.ndarray
        :param left_motion: Motion intensity for left ROI
        :type left_motion: int
        :param right_motion: Motion intensity for right ROI
        :type right_motion: int
        """
        cv2.rectangle(frame, self.roi_left[0], self.roi_left[1], (0, 255, 0), 2)
        cv2.rectangle(frame, self.roi_right[0], self.roi_right[1], (0, 255, 0), 2)
        cv2.putText(
            frame, f"L: {left_motion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"R: {right_motion}",
            (self.roi_right[0][0], 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def handle_detection(
        self,
        detected_count: int,
        left_motion: int,
        right_motion: int,
        direction: str,
        left_occupied: bool,
        right_occupied: bool,
        frame: np.ndarray,
    ) -> None:
        """
        Handles saving detection data, sending notifications,
        and logging detection details.

        :param detected_count: Current number of train detections
        :type detected_count: int
        :param left_motion: Motion intensity in left ROI
        :type left_motion: int
        :param right_motion: Motion intensity in right ROI
        :type right_motion: int
        :param direction: Train movement direction
        :type direction: str
        :param left_occupied: Whether left ROI is occupied
        :type left_occupied: bool
        :param right_occupied: Whether right ROI is occupied
        :type right_occupied: bool
        :param frame: Video frame at detection moment
        :type frame: np.ndarray
        """
        timestamp = datetime.now().strftime(DATETIME_FORMAT)

        # Load, update and save detection data
        self.load_data()
        self.data["detections"].append(
            {
                "timestamp": timestamp,
                "left_motion": left_motion,
                "right_motion": right_motion,
                "direction": direction,
                "left_occupied": left_occupied,
                "right_occupied": right_occupied,
            }
        )
        self.save_data()

        # Save detection image
        filename = f"detection_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(filename, frame)

        # Send Telegram notification
        message = (
            f"🚆 Train detected moving {direction}!\n"
            f"Time: {timestamp}\n"
            f"Left: {'Occupied' if left_occupied else 'Empty'}\n"
            f"Right: {'Occupied' if right_occupied else 'Empty'}\n"
        )

        capture = f"Train {direction} - {timestamp}"
        self.telegram_bot.send_message(message)
        self.telegram_bot.send_photo(filename, capture, timestamp)

        # Clean up image file
        os.remove(filename)

        # Log detection
        logging.info(f"Train detected! Count: {detected_count} at {timestamp}")
        logging.info(f"Direction: {direction}")
        logging.debug(
            f"Left: {'Occupied' if left_occupied else 'Empty'}, "
            f"Right: {'Occupied' if right_occupied else 'Empty'}"
        )

    def detect_train(
        self,
        current_time: float,
        left_motion: int,
        right_motion: int,
        prev_left_state: bool,
        prev_right_state: bool,
        frame: np.ndarray,
        detected_count: int,
    ) -> Tuple[bool, int, float]:
        """
        Determine if a train has been detected based on motion and state changes.

        :param current_time: Current timestamp
        :type current_time: float
        :param left_motion: Motion intensity in left ROI
        :type left_motion: int
        :param right_motion: Motion intensity in right ROI
        :type right_motion: int
        :param prev_left_state: Previous left ROI occupancy state
        :type prev_left_state: bool
        :param prev_right_state: Previous right ROI occupancy state
        :type prev_right_state: bool
        :param frame: Current video frame
        :type frame: np.ndarray
        :param detected_count: Current number of train detections
        :type detected_count: int
        :return: Tuple of (detection_made, new_detection_count, last_detection_time)
        :rtype: Tuple[bool, int, float]
        """
        # Determine current states based on motion
        left_occupied = left_motion > self.motion_threshold
        right_occupied = right_motion > self.motion_threshold

        # Check for state transitions
        state_changed = (prev_left_state != left_occupied) or (prev_right_state != right_occupied)

        # Only one side should be occupied for a valid train detection
        valid_train_condition = left_occupied != right_occupied

        # Check cooldown period
        cooldown_satisfied = current_time - self.last_detection_time > self.detection_cooldown

        if state_changed and valid_train_condition and cooldown_satisfied:
            # Determine direction based on which side is occupied
            direction = "Left to Right" if left_occupied else "Right to Left"

            # Increment counter
            new_count = detected_count + 1

            # Process detection
            self.handle_detection(
                new_count,
                left_motion,
                right_motion,
                direction,
                left_occupied,
                right_occupied,
                frame,
            )

            return True, new_count, current_time
        return False, detected_count, self.last_detection_time

    def run(self) -> None:
        """
        Main detection loop for continuous train monitoring.

        Runs an infinite loop capturing and processing video frames
        to detect train movements. Handles graceful interruption
        and provides final detection statistics.
        """
        logging.info("Dual ROI motion detection started!")
        logging.info(f"Motion threshold: {self.motion_threshold}")

        detector_state = self._initialize_detector_state()

        try:
            while True:
                time.sleep(0.1)
                detector_state = self._process_frame(detector_state)

        except KeyboardInterrupt:
            logging.info("\nStopping detection...")
        finally:
            self.cap.release()
            logging.info(f"\nTotal detections: {detector_state['detected_count']}")
            logging.info("Detection stopped")

    def _process_frame(self, state: dict) -> dict:
        """
        Process a single video frame and update detector state.

        :param state: Current detector state dictionary
        :type state: dict
        :return: Updated detector state dictionary
        :rtype: dict
        """
        frame = self.get_frame()
        if frame is None:
            return state

        # Analyze both ROIs
        left_gray = self.analyze_roi(frame, self.roi_left)
        right_gray = self.analyze_roi(frame, self.roi_right)

        # Initialize previous frames if needed
        if state["prev_left"] is None or state["prev_right"] is None:
            state["prev_left"] = left_gray
            state["prev_right"] = right_gray
            return state

        # Process motion
        left_motion, right_motion = self.process_motion(
            left_gray, right_gray, state["prev_left"], state["prev_right"]
        )

        # Update display
        self.draw_status(frame, left_motion, right_motion)

        # Check for state changes
        return self._check_state_changes(
            state, frame, left_motion, right_motion, left_gray, right_gray
        )

    def _check_state_changes(
        self,
        state: dict,
        frame: np.ndarray,
        left_motion: int,
        right_motion: int,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
    ) -> dict:
        """
        Evaluate and process potential state changes in train detection.

        :param state: Current detector state dictionary
        :type state: dict
        :param frame: Current video frame
        :type frame: np.ndarray
        :param left_motion: Motion intensity in left ROI
        :type left_motion: int
        :param right_motion: Motion intensity in right ROI
        :type right_motion: int
        :param left_gray: Processed grayscale left ROI
        :type left_gray: np.ndarray
        :param right_gray: Processed grayscale right ROI
        :type right_gray: np.ndarray
        :return: Updated detector state dictionary
        :rtype: dict
        """
        current_time = time.time()

        # Always update previous frames for next iteration
        state["prev_left"] = left_gray
        state["prev_right"] = right_gray

        # Check if enough time has passed since last state change
        if current_time - state["last_state_change"] <= self.state_change_cooldown:
            return state

        # Store previous states
        prev_left_state = state["left_occupied"]
        prev_right_state = state["right_occupied"]

        # Check for train detection
        detection_made, detected_count, detection_time = self.detect_train(
            current_time,
            left_motion,
            right_motion,
            prev_left_state,
            prev_right_state,
            frame,
            state["detected_count"],
        )

        if detection_made:
            self.last_detection_time = detection_time
            state["last_state_change"] = current_time
            state["detected_count"] = detected_count

        # Update occupancy states for next iteration
        state["left_occupied"] = left_motion > self.motion_threshold
        state["right_occupied"] = right_motion > self.motion_threshold

        return state
