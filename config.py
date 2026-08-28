from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent
MODELS_FOLDER = PROJECT_FOLDER / "models"
SOUNDS_FOLDER = PROJECT_FOLDER / "sounds"
LOGS_FOLDER = PROJECT_FOLDER / "logs"

CAMERA_INDEX = 0
WINDOW_NAME = "Smart Eye Monitor - Press Q to quit"
INITIAL_WINDOW_WIDTH = 1100
INITIAL_WINDOW_HEIGHT = 620
STATUS_PANEL_WIDTH = 340

FACE_OVERLAY_DURATION_SECONDS = 2.0
# Both eyes must be more strongly closed than a single eye.
BOTH_EYES_CLOSED_THRESHOLD = 0.14
CALIBRATED_BOTH_EYES_CLOSED_RATIO = 0.55

EYE_CLOSED_THRESHOLD = 0.17
EYE_SMOOTHING_FRAMES = 3
EYE_STATE_CONFIRMATION_FRAMES = 3
EYE_CLOSED_DURATION_SECONDS = 1.5

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

FACE_WIDTH_MOVING_AWAY_THRESHOLD = 0.28
FACE_WIDTH_TOO_FAR_THRESHOLD = 0.18
DISTANCE_SMOOTHING_FRAMES = 5
DISTANCE_STATE_CONFIRMATION_FRAMES = 4

NO_FACE_GRACE_PERIOD_SECONDS = 1.0

EYE_COVER_DURATION_SECONDS = 1.0
EYE_COVER_MIN_OVERLAP = 0.30
HAND_DETECTION_INTERVAL = 2

# Likely full-face cover: face disappears while a hand remains over its last
# known position. This is also useful when an object is held across the face.
FACE_COVER_DURATION_SECONDS = 0.75
FACE_COVER_MIN_OVERLAP = 0.45
LAST_FACE_POSITION_TIMEOUT_SECONDS = 1.5

# Upper-body detection separates a covered face from a person who left frame.
POSE_DETECTION_INTERVAL = 3
UPPER_BODY_GRACE_PERIOD_SECONDS = 0.8
MIN_SHOULDER_VISIBILITY = 0.50

CALIBRATION_DURATION_SECONDS = 5.0
CALIBRATION_MIN_SAMPLES = 30
CALIBRATED_EYE_CLOSED_RATIO = 0.65
CALIBRATED_MOVING_AWAY_RATIO = 0.75
CALIBRATED_TOO_FAR_RATIO = 0.55

MODEL_PATH = MODELS_FOLDER / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_FOLDER / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_FOLDER / "pose_landmarker_lite.task"

EYE_CLOSED_SOUND_PATH = SOUNDS_FOLDER / "eye_closed.wav"
MOVED_AWAY_SOUND_PATH = SOUNDS_FOLDER / "moved_away.wav"
EYE_COVERED_SOUND_PATH = SOUNDS_FOLDER / "eye_covered.wav"
FACE_COVERED_SOUND_PATH = SOUNDS_FOLDER / "face_covered.wav"

# Privacy-friendly event log. The program records text events only.
EVENT_LOG_PATH = LOGS_FOLDER / "events.log"

MESSAGES = {
    "eye_closed": "Please open your eyes.",
    "moved_away": "Please move closer to the camera.",
    "no_face": "Please stay in front of the camera.",
    "eye_covered": "Please remove your hand from your eyes.",
    "face_covered": "Please remove the object covering your face.",
}
