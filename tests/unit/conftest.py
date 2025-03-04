from pathlib import Path

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
