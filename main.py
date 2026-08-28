import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import config
from alerts.sound_manager import SoundManager, update_away_sound
from ui.display import (
    build_status_panel,
    draw_alert_panel,
    get_distance_state_color,
    get_eye_state_color,
)
from utils.logger import EventLogger


def get_face_area(landmarks):
    x_values = [point.x for point in landmarks]
    y_values = [point.y for point in landmarks]
    return (max(x_values) - min(x_values)) * (max(y_values) - min(y_values))


def get_face_width_ratio(landmarks):
    x_values = [point.x for point in landmarks]
    return max(x_values) - min(x_values)


def get_landmark_points(landmarks, indices, frame_width, frame_height):
    return np.array(
        [
            (
                int(landmarks[index].x * frame_width),
                int(landmarks[index].y * frame_height),
            )
            for index in indices
        ],
        dtype=np.int32,
    )


def draw_primary_face(frame, landmarks):
    frame_height, frame_width = frame.shape[:2]

    points = [
        (
            int(point.x * frame_width),
            int(point.y * frame_height),
        )
        for point in landmarks
    ]

    x, y, width, height = cv2.boundingRect(np.array(points, dtype=np.int32))

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    for point_x, point_y in points:
        cv2.circle(frame, (point_x, point_y), 1, (0, 255, 255), -1)

    cv2.putText(
        frame,
        "FACE DETECTED",
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def calculate_eye_aspect_ratio(
    landmarks,
    eye_indices,
    frame_width,
    frame_height,
):
    points = get_landmark_points(
        landmarks,
        eye_indices,
        frame_width,
        frame_height,
    )

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def classify_eye_state(left_ear, right_ear):
    left_closed = left_ear < config.EYE_CLOSED_THRESHOLD
    right_closed = right_ear < config.EYE_CLOSED_THRESHOLD

    # Use a stricter check before saying both eyes are closed.
    both_eyes_closed = (
        left_ear < config.BOTH_EYES_CLOSED_THRESHOLD
        and right_ear < config.BOTH_EYES_CLOSED_THRESHOLD
    )

    if both_eyes_closed:
        return "BOTH EYES CLOSED"
    if left_closed:
        return "LEFT EYE CLOSED"
    if right_closed:
        return "RIGHT EYE CLOSED"

    return "BOTH EYES OPEN"


def classify_distance_state(face_width_ratio):
    if face_width_ratio < config.FACE_WIDTH_TOO_FAR_THRESHOLD:
        return "TOO FAR"

    if face_width_ratio < config.FACE_WIDTH_MOVING_AWAY_THRESHOLD:
        return "MOVING AWAY"

    return "NORMAL"


def get_normalized_box(landmarks):
    x_values = [point.x for point in landmarks]
    y_values = [point.y for point in landmarks]

    return min(x_values), min(y_values), max(x_values), max(y_values)


def get_eye_region_box(face_landmarks, eye_indices):
    eye_landmarks = [face_landmarks[index] for index in eye_indices]
    left, top, right, bottom = get_normalized_box(eye_landmarks)

    padding_x = max((right - left) * 0.70, 0.01)
    padding_y = max((bottom - top) * 2.00, 0.015)

    return (
        max(0.0, left - padding_x),
        max(0.0, top - padding_y),
        min(1.0, right + padding_x),
        min(1.0, bottom + padding_y),
    )


def get_overlap_ratio(hand_box, eye_box):
    hand_left, hand_top, hand_right, hand_bottom = hand_box
    eye_left, eye_top, eye_right, eye_bottom = eye_box

    overlap_left = max(hand_left, eye_left)
    overlap_top = max(hand_top, eye_top)
    overlap_right = min(hand_right, eye_right)
    overlap_bottom = min(hand_bottom, eye_bottom)

    overlap_width = max(0.0, overlap_right - overlap_left)
    overlap_height = max(0.0, overlap_bottom - overlap_top)

    overlap_area = overlap_width * overlap_height
    eye_area = (eye_right - eye_left) * (eye_bottom - eye_top)

    if eye_area == 0:
        return 0.0

    return overlap_area / eye_area


def get_eye_cover_state(face_landmarks, hand_landmarks_list):
    """Return which eye regions are likely covered by a detected hand."""
    if not hand_landmarks_list:
        return "NO"

    left_eye_box = get_eye_region_box(
        face_landmarks,
        config.LEFT_EYE_INDICES,
    )
    right_eye_box = get_eye_region_box(
        face_landmarks,
        config.RIGHT_EYE_INDICES,
    )

    hand_boxes = [
        get_normalized_box(hand_landmarks)
        for hand_landmarks in hand_landmarks_list
    ]

    left_covered = any(
        get_overlap_ratio(hand_box, left_eye_box)
        >= config.EYE_COVER_MIN_OVERLAP
        for hand_box in hand_boxes
    )

    right_covered = any(
        get_overlap_ratio(hand_box, right_eye_box)
        >= config.EYE_COVER_MIN_OVERLAP
        for hand_box in hand_boxes
    )

    if left_covered and right_covered:
        return "BOTH EYES COVERED"
    if left_covered:
        return "LEFT EYE COVERED"
    if right_covered:
        return "RIGHT EYE COVERED"

    return "NO"


def is_face_likely_covered(last_face_box, hand_landmarks_list):
    """Check whether a hand overlaps a large part of the last face position."""
    if last_face_box is None or not hand_landmarks_list:
        return False

    hand_boxes = [
        get_normalized_box(hand_landmarks)
        for hand_landmarks in hand_landmarks_list
    ]

    return any(
        get_overlap_ratio(hand_box, last_face_box)
        >= config.FACE_COVER_MIN_OVERLAP
        for hand_box in hand_boxes
    )


def is_upper_body_visible(pose_landmarks_list):
    """Return True when both shoulders are confidently visible."""
    left_shoulder_index = 11
    right_shoulder_index = 12

    for pose_landmarks in pose_landmarks_list:
        if len(pose_landmarks) <= right_shoulder_index:
            continue

        left_shoulder = pose_landmarks[left_shoulder_index]
        right_shoulder = pose_landmarks[right_shoulder_index]

        left_visibility = left_shoulder.visibility or 0.0
        right_visibility = right_shoulder.visibility or 0.0

        if (
            left_visibility >= config.MIN_SHOULDER_VISIBILITY
            and right_visibility >= config.MIN_SHOULDER_VISIBILITY
        ):
            return True

    return False


class EyeStateTracker:
    def __init__(self):
        self.left_history = deque(maxlen=config.EYE_SMOOTHING_FRAMES)
        self.right_history = deque(maxlen=config.EYE_SMOOTHING_FRAMES)
        self.confirmed_state = None
        self.candidate_state = None
        self.candidate_frames = 0

    def reset(self):
        self.left_history.clear()
        self.right_history.clear()
        self.confirmed_state = None
        self.candidate_state = None
        self.candidate_frames = 0

    def update(self, left_ear, right_ear):
        self.left_history.append(left_ear)
        self.right_history.append(right_ear)

        if len(self.left_history) < config.EYE_SMOOTHING_FRAMES:
            return "ANALYZING"

        left_average = sum(self.left_history) / len(self.left_history)
        right_average = sum(self.right_history) / len(self.right_history)

        candidate = classify_eye_state(left_average, right_average)

        if candidate == self.candidate_state:
            self.candidate_frames += 1
        else:
            self.candidate_state = candidate
            self.candidate_frames = 1

        if self.candidate_frames >= config.EYE_STATE_CONFIRMATION_FRAMES:
            self.confirmed_state = self.candidate_state

        return self.confirmed_state or "ANALYZING"


class DistanceTracker:
    def __init__(self):
        self.history = deque(maxlen=config.DISTANCE_SMOOTHING_FRAMES)
        self.confirmed_state = None
        self.candidate_state = None
        self.candidate_frames = 0

    def reset(self):
        self.history.clear()
        self.confirmed_state = None
        self.candidate_state = None
        self.candidate_frames = 0

    def update(self, face_width_ratio):
        self.history.append(face_width_ratio)

        if len(self.history) < config.DISTANCE_SMOOTHING_FRAMES:
            return "ANALYZING"

        average_width = sum(self.history) / len(self.history)
        candidate = classify_distance_state(average_width)

        if candidate == self.candidate_state:
            self.candidate_frames += 1
        else:
            self.candidate_state = candidate
            self.candidate_frames = 1

        if self.candidate_frames >= config.DISTANCE_STATE_CONFIRMATION_FRAMES:
            self.confirmed_state = self.candidate_state

        return self.confirmed_state or "ANALYZING"


class TimedAlertTracker:
    def __init__(self, duration_seconds):
        self.duration_seconds = duration_seconds
        self.started_at = None
        self.alert_is_active = False

    def reset(self):
        self.started_at = None
        self.alert_is_active = False

    def update(self, condition, current_time):
        if not condition:
            self.reset()
            return False, False

        if self.started_at is None:
            self.started_at = current_time

        if (
            current_time - self.started_at >= self.duration_seconds
            and not self.alert_is_active
        ):
            self.alert_is_active = True
            return True, True

        return self.alert_is_active, False


class NoFaceTracker(TimedAlertTracker):
    @property
    def is_waiting(self):
        return self.started_at is not None and not self.alert_is_active

    def update(self, current_time):
        return super().update(True, current_time)


class CalibrationManager:
    def __init__(self):
        self.active = False
        self.started_at = None
        self.eye_values = []
        self.face_width_values = []
        self.completed_at = None

    def start(self):
        self.active = True
        self.started_at = time.perf_counter()
        self.eye_values.clear()
        self.face_width_values.clear()
        self.completed_at = None
        print("Calibration started. Keep both eyes open and sit normally.")

    def update(self, left_ear, right_ear, face_width_ratio, current_time):
        if not self.active:
            return False

        if (
            left_ear > config.EYE_CLOSED_THRESHOLD
            and right_ear > config.EYE_CLOSED_THRESHOLD
        ):
            self.eye_values.append((left_ear + right_ear) / 2)
            self.face_width_values.append(face_width_ratio)

        if current_time - self.started_at < config.CALIBRATION_DURATION_SECONDS:
            return False

        if len(self.eye_values) < config.CALIBRATION_MIN_SAMPLES:
            print("Calibration needs more clear samples. Restarting.")
            self.started_at = current_time
            self.eye_values.clear()
            self.face_width_values.clear()
            return False

        normal_eye_openness = sum(self.eye_values) / len(self.eye_values)
        normal_face_width = (
            sum(self.face_width_values) / len(self.face_width_values)
        )

        config.EYE_CLOSED_THRESHOLD = (
            normal_eye_openness * config.CALIBRATED_EYE_CLOSED_RATIO
        )
        config.BOTH_EYES_CLOSED_THRESHOLD = (
            normal_eye_openness * config.CALIBRATED_BOTH_EYES_CLOSED_RATIO
        )
        config.FACE_WIDTH_MOVING_AWAY_THRESHOLD = (
            normal_face_width * config.CALIBRATED_MOVING_AWAY_RATIO
        )
        config.FACE_WIDTH_TOO_FAR_THRESHOLD = (
            normal_face_width * config.CALIBRATED_TOO_FAR_RATIO
        )

        self.active = False
        self.completed_at = current_time

        print("Calibration complete.")
        print(f"Eye threshold: {config.EYE_CLOSED_THRESHOLD:.3f}")
        print(f"Normal face width: {normal_face_width:.3f}")

        return True

    def get_status(self, current_time):
        if self.active:
            seconds_left = max(
                0,
                config.CALIBRATION_DURATION_SECONDS
                - (current_time - self.started_at),
            )
            return f"RUNNING {seconds_left:.1f}s", (0, 165, 255)

        if (
            self.completed_at is not None
            and current_time - self.completed_at < 3.0
        ):
            return "COMPLETE", (0, 255, 0)

        return "READY (C)", (185, 190, 195)


def create_face_landmarker():
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(config.MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=3,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(config.HAND_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def create_pose_landmarker():
    """Create a lightweight Pose Landmarker for upper-body detection."""
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(config.POSE_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def main():
    if not config.MODEL_PATH.is_file():
        print(f"Face model file not found: {config.MODEL_PATH}")
        return

    if not config.HAND_MODEL_PATH.is_file():
        print(f"Hand model file not found: {config.HAND_MODEL_PATH}")
        return

    if not config.POSE_MODEL_PATH.is_file():
        print(f"Pose model file not found: {config.POSE_MODEL_PATH}")
        print("Download pose_landmarker_lite.task into the models folder.")
        return

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"Could not open camera {config.CAMERA_INDEX}.")
        return

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.WINDOW_NAME,
        config.INITIAL_WINDOW_WIDTH,
        config.INITIAL_WINDOW_HEIGHT,
    )

    sound_manager = SoundManager(
        {
            "eye_closed": config.EYE_CLOSED_SOUND_PATH,
            "moved_away": config.MOVED_AWAY_SOUND_PATH,
            "eye_covered": config.EYE_COVERED_SOUND_PATH,
            "face_covered": config.FACE_COVERED_SOUND_PATH,
        }
    )

    previous_time = time.perf_counter()
    start_time = time.perf_counter()
    last_timestamp_ms = 0

    face_was_detected = False
    face_overlay_started_at = None
    away_alert_is_active = False
    too_far_event_is_active = False

    hand_frame_counter = 0
    last_hand_landmarks = []
    last_face_box = None
    last_face_seen_at = None
    pose_frame_counter = 0
    last_upper_body_seen_at = None

    eye_tracker = EyeStateTracker()
    distance_tracker = DistanceTracker()
    closed_eye_alert = TimedAlertTracker(
        config.EYE_CLOSED_DURATION_SECONDS
    )
    no_face_tracker = NoFaceTracker(
        config.NO_FACE_GRACE_PERIOD_SECONDS
    )
    eye_cover_alert = TimedAlertTracker(
        config.EYE_COVER_DURATION_SECONDS
    )
    face_cover_alert = TimedAlertTracker(
        config.FACE_COVER_DURATION_SECONDS
    )
    calibration = CalibrationManager()
    event_logger = EventLogger(config.EVENT_LOG_PATH)
    event_logger.log("Smart Eye Monitor started")

    try:
        with (
            create_face_landmarker() as face_landmarker,
            create_hand_landmarker() as hand_landmarker,
            create_pose_landmarker() as pose_landmarker,
        ):
            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    print("Could not read a frame from the camera.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                face_result = face_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                # Keep hand tracking active even when the face is hidden.
                hand_frame_counter += 1
                if (
                    hand_frame_counter == 1
                    or hand_frame_counter % config.HAND_DETECTION_INTERVAL == 0
                ):
                    hand_result = hand_landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms,
                    )
                    last_hand_landmarks = hand_result.hand_landmarks

                if face_result.face_landmarks:
                    primary_face = max(
                        face_result.face_landmarks,
                        key=get_face_area,
                    )

                    last_face_box = get_normalized_box(primary_face)
                    last_face_seen_at = time.perf_counter()
                    pose_frame_counter = 0
                    last_upper_body_seen_at = None

                    if face_cover_alert.alert_is_active:
                        sound_manager.stop("face_covered")
                        event_logger.log("Face-cover alert cleared")

                    face_cover_alert.reset()

                    was_no_face_active = no_face_tracker.alert_is_active
                    no_face_tracker.reset()

                    if was_no_face_active:
                        event_logger.log("Person returned to camera")

                    if not face_was_detected or face_overlay_started_at is None:
                        face_overlay_started_at = time.perf_counter()

                    face_was_detected = True

                    if (
                        time.perf_counter() - face_overlay_started_at
                        < config.FACE_OVERLAY_DURATION_SECONDS
                    ):
                        draw_primary_face(frame, primary_face)

                    frame_height, frame_width = frame.shape[:2]

                    left_ear = calculate_eye_aspect_ratio(
                        primary_face,
                        config.LEFT_EYE_INDICES,
                        frame_width,
                        frame_height,
                    )

                    right_ear = calculate_eye_aspect_ratio(
                        primary_face,
                        config.RIGHT_EYE_INDICES,
                        frame_width,
                        frame_height,
                    )

                    eye_state = eye_tracker.update(left_ear, right_ear)

                    eye_cover_state = get_eye_cover_state(
                        primary_face,
                        last_hand_landmarks,
                    )
                    eyes_likely_covered = (
                        eye_cover_state == "BOTH EYES COVERED"
                    )
                    any_eye_likely_covered = eye_cover_state != "NO"

                    was_eye_cover_active = eye_cover_alert.alert_is_active

                    eye_cover_active, eye_cover_started = eye_cover_alert.update(
                        eyes_likely_covered,
                        time.perf_counter(),
                    )

                    if was_eye_cover_active and not eye_cover_active:
                        sound_manager.stop("eye_covered")
                        event_logger.log("Eye-cover alert cleared")

                    if any_eye_likely_covered:
                        if closed_eye_alert.alert_is_active:
                            sound_manager.stop("eye_closed")
                            event_logger.log("Eyes closed alert cleared")

                        closed_eye_alert.reset()
                        closed_eye_active = False
                        closed_eye_started = False

                    else:
                        was_closed_eye_active = closed_eye_alert.alert_is_active

                        closed_eye_active, closed_eye_started = (
                            closed_eye_alert.update(
                                eye_state == "BOTH EYES CLOSED",
                                time.perf_counter(),
                            )
                        )

                        if was_closed_eye_active and not closed_eye_active:
                            sound_manager.stop("eye_closed")
                            event_logger.log("Eyes returned to normal")

                    if eye_cover_started:
                        sound_manager.stop("eye_closed")
                        sound_manager.stop("moved_away")
                        sound_manager.play("eye_covered")
                        event_logger.log("Eyes covered alert triggered")

                    if closed_eye_started:
                        sound_manager.stop("moved_away")
                        sound_manager.play("eye_closed")
                        event_logger.log("Eyes closed alert triggered")

                    face_width_ratio = get_face_width_ratio(primary_face)

                    calibration_completed = calibration.update(
                        left_ear,
                        right_ear,
                        face_width_ratio,
                        time.perf_counter(),
                    )

                    if calibration_completed:
                        eye_tracker.reset()
                        distance_tracker.reset()
                        event_logger.log("Calibration completed")

                    distance_state = distance_tracker.update(face_width_ratio)

                    was_too_far_event_active = too_far_event_is_active
                    too_far_event_is_active = distance_state == "TOO FAR"

                    if (
                        too_far_event_is_active
                        and not was_too_far_event_active
                    ):
                        event_logger.log("Person moved too far from camera")
                    elif (
                        was_too_far_event_active
                        and not too_far_event_is_active
                    ):
                        event_logger.log("Person returned to normal distance")

                    was_away_alert_active = away_alert_is_active
                    away_alert_is_active = distance_state == "TOO FAR"

                    priority_alert_active = (
                        closed_eye_active or eye_cover_active
                    )

                    update_away_sound(
                        sound_manager,
                        was_away_alert_active,
                        away_alert_is_active,
                        priority_alert_active,
                    )

                    visible_eye_state = (
                        eye_cover_state
                        if any_eye_likely_covered
                        else eye_state
                    )

                    if eye_cover_active:
                        eye_cover_status = "YES"
                        eye_cover_color = (0, 0, 255)
                    elif eyes_likely_covered:
                        eye_cover_status = "CHECKING"
                        eye_cover_color = (0, 165, 255)
                    elif eye_cover_state == "LEFT EYE COVERED":
                        eye_cover_status = "LEFT EYE"
                        eye_cover_color = (0, 165, 255)
                    elif eye_cover_state == "RIGHT EYE COVERED":
                        eye_cover_status = "RIGHT EYE"
                        eye_cover_color = (0, 165, 255)
                    else:
                        eye_cover_status = "NO"
                        eye_cover_color = (0, 255, 0)

                    panel_face_value = "DETECTED"
                    panel_face_color = (0, 255, 0)

                    panel_eye_value = visible_eye_state
                    panel_eye_color = get_eye_state_color(visible_eye_state)

                    panel_distance_value = distance_state
                    panel_distance_color = get_distance_state_color(
                        distance_state
                    )

                    panel_eye_cover_value = eye_cover_status
                    panel_eye_cover_color = eye_cover_color

                    if eye_cover_active:
                        panel_alert_value = "EYES COVERED"
                        panel_alert_color = (0, 0, 255)

                        draw_alert_panel(
                            frame,
                            "WARNING: EYES COVERED",
                            config.MESSAGES["eye_covered"],
                            (0, 0, 180),
                        )

                    elif closed_eye_active:
                        panel_alert_value = "EYES CLOSED"
                        panel_alert_color = (0, 0, 255)

                        draw_alert_panel(
                            frame,
                            "WARNING: EYES CLOSED",
                            config.MESSAGES["eye_closed"],
                            (0, 0, 180),
                        )

                    elif away_alert_is_active:
                        panel_alert_value = "MOVE CLOSER"
                        panel_alert_color = (0, 165, 255)

                        draw_alert_panel(
                            frame,
                            "WARNING: YOU ARE TOO FAR",
                            config.MESSAGES["moved_away"],
                            (0, 100, 200),
                        )

                    else:
                        panel_alert_value = "NONE"
                        panel_alert_color = (0, 255, 0)

                else:
                    face_was_detected = False
                    face_overlay_started_at = None

                    eye_tracker.reset()
                    distance_tracker.reset()
                    too_far_event_is_active = False

                    if closed_eye_alert.alert_is_active:
                        sound_manager.stop("eye_closed")

                    if eye_cover_alert.alert_is_active:
                        sound_manager.stop("eye_covered")

                    closed_eye_alert.reset()
                    eye_cover_alert.reset()

                    current_time = time.perf_counter()
                    pose_frame_counter += 1

                    if (
                        pose_frame_counter == 1
                        or pose_frame_counter % config.POSE_DETECTION_INTERVAL
                        == 0
                    ):
                        pose_result = pose_landmarker.detect_for_video(
                            mp_image,
                            timestamp_ms,
                        )

                        if is_upper_body_visible(
                            pose_result.pose_landmarks
                        ):
                            last_upper_body_seen_at = current_time

                    upper_body_is_visible = (
                        last_upper_body_seen_at is not None
                        and current_time - last_upper_body_seen_at
                        <= config.UPPER_BODY_GRACE_PERIOD_SECONDS
                    )

                    last_face_position_is_recent = (
                        last_face_box is not None
                        and last_face_seen_at is not None
                        and current_time - last_face_seen_at
                        <= config.LAST_FACE_POSITION_TIMEOUT_SECONDS
                    )
                    face_likely_covered = (
                        upper_body_is_visible
                        or (
                            last_face_position_is_recent
                            and is_face_likely_covered(
                                last_face_box,
                                last_hand_landmarks,
                            )
                        )
                    )

                    was_face_cover_active = face_cover_alert.alert_is_active
                    face_cover_active, face_cover_started = (
                        face_cover_alert.update(
                            face_likely_covered,
                            current_time,
                        )
                    )

                    if was_face_cover_active and not face_cover_active:
                        sound_manager.stop("face_covered")
                        event_logger.log("Face-cover alert cleared")

                    if face_cover_started:
                        sound_manager.stop("moved_away")
                        sound_manager.play("face_covered")
                        event_logger.log("Face-cover alert triggered")

                    if face_cover_active:
                        # A cover is different from leaving the camera.
                        no_face_tracker.reset()
                        no_face_active = False
                    else:
                        no_face_active, no_face_started = no_face_tracker.update(
                            current_time
                        )

                        if no_face_started:
                            event_logger.log(
                                "No person detected alert triggered"
                            )

                    was_away_alert_active = away_alert_is_active

                    away_alert_is_active = (
                        not face_cover_active
                        and (
                            no_face_active
                            or (
                                away_alert_is_active
                                and no_face_tracker.is_waiting
                            )
                        )
                    )

                    update_away_sound(
                        sound_manager,
                        was_away_alert_active,
                        away_alert_is_active,
                        face_cover_active,
                    )

                    if face_cover_active:
                        panel_face_value = "COVERED"
                        panel_face_color = (0, 0, 255)
                    else:
                        panel_face_value = (
                            "NO PERSON" if no_face_active else "CHECKING"
                        )
                        panel_face_color = (
                            (0, 0, 255)
                            if no_face_active
                            else (0, 165, 255)
                        )

                    panel_eye_value = "NOT AVAILABLE"
                    panel_eye_color = (220, 220, 220)

                    panel_distance_value = "NOT AVAILABLE"
                    panel_distance_color = (220, 220, 220)

                    panel_eye_cover_value = (
                        "FACE COVERED"
                        if face_cover_active
                        else "NOT AVAILABLE"
                    )
                    panel_eye_cover_color = (
                        (0, 0, 255)
                        if face_cover_active
                        else (220, 220, 220)
                    )

                    if face_cover_active:
                        panel_alert_value = "FACE COVERED"
                        panel_alert_color = (0, 0, 255)

                        draw_alert_panel(
                            frame,
                            "WARNING: FACE COVERED",
                            config.MESSAGES["face_covered"],
                            (0, 0, 180),
                        )

                    elif no_face_active:
                        panel_alert_value = "NO PERSON"
                        panel_alert_color = (0, 0, 255)

                        draw_alert_panel(
                            frame,
                            "WARNING: NO PERSON DETECTED",
                            config.MESSAGES["no_face"],
                            (0, 0, 180),
                        )

                    else:
                        panel_alert_value = "NONE"
                        panel_alert_color = (0, 255, 0)

                current_time = time.perf_counter()
                elapsed_time = current_time - previous_time
                fps = 1 / elapsed_time if elapsed_time > 0 else 0
                previous_time = current_time

                calibration_value, calibration_color = calibration.get_status(
                    time.perf_counter()
                )

                status_panel = build_status_panel(
                    frame.shape[0],
                    panel_face_value,
                    panel_face_color,
                    panel_eye_value,
                    panel_eye_color,
                    panel_distance_value,
                    panel_distance_color,
                    panel_eye_cover_value,
                    panel_eye_cover_color,
                    panel_alert_value,
                    panel_alert_color,
                    calibration_value,
                    calibration_color,
                    fps,
                )

                display_frame = np.hstack((frame, status_panel))

                cv2.imshow(config.WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("c"):
                    calibration.start()
                    event_logger.log("Calibration started")

    except RuntimeError as error:
        print(f"MediaPipe could not run: {error}")

    finally:
        camera.release()
        sound_manager.close()
        cv2.destroyAllWindows()
        event_logger.log("Smart Eye Monitor stopped")
        print("Camera, sound, and MediaPipe resources released.")


if __name__ == "__main__":
    main()
