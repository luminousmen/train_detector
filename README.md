[![Run Tests](https://github.com/luminousmen/train_detector/actions/workflows/test.yml/badge.svg)](https://github.com/luminousmen/train_detector/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/luminousmen/train_detector/graph/badge.svg?token=BMD9B5VCPZ)](https://codecov.io/gh/luminousmen/train_detector)

# Train Detector

It's a small Python-based motion detection system that identifies train movements using computer vision and sends notifications via Telegram. 

I run it on Raspberry Pi with an attached camera. For me, it solves the problem of gathering statistics on the passing trains—hence the name—but I believe it can be used in other cases as well. I tried to find something like this before, so I hope this will be a good starting point for someone with the same problem.
## Installation

To set up the project, ensure you have `uv` installed:

```sh
pip install uv
```

Then, sync dependencies:

```sh
uv sync
```

## Usage

Run the train detection system with:

```sh
uv run python main.py
```

### Telegram Bot Integration
Set up a Telegram bot and provide your bot token and chat ID as environment variables:

```sh
export TELEGRAM_TOKEN="your-bot-token"
export CHAT_ID="your-chat-id"
```

## Example Output

When a train is detected, the bot sends a notification:

```
🚆 Train detected moving Left to Right!
Time: 2025-02-27 10:00:00
Left: Occupied
Right: Empty
```

Examples are in the [docs/examples.md](docs/examples.md)

## Algorithm Workflow

The algorithm is simple yet powerful. It minimizes false negatives but may generate some false positives. 

### 1. Camera Initialization
- Capture video stream from a fixed camera
- Set predefined resolution (640x480 pixels)
- Define two rectangular Regions of Interest (ROIs)
  - Left ROI: Left side of the frame
  - Right ROI: Right side of the frame

### 2. Motion Detection Process
1. Capture video frame
2. Convert frame to grayscale
3. Apply Gaussian blur to reduce noise
4. Compare current frame with previous frame
5. Calculate pixel-level differences

#### Detection Logic
- Compute motion intensity by counting changed pixels
- Detect train when:
  - Motion detected in only one ROI
  - State change occurs
  - Cooldown period elapsed

To reduce false positives, consider:

- Adjusting motion detection sensitivity
- Ignoring small, temporary motion artifacts
- Implementing background subtraction for better accuracy
