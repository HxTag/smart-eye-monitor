"""Functions that draw the alert and status interface."""

import cv2
import numpy as np

import config


def get_eye_state_color(state):
    if "COVERED" in state:
        return (0, 0, 255)
    if state == "BOTH EYES OPEN":
        return (0, 255, 0)
    if state in {"LEFT EYE CLOSED", "RIGHT EYE CLOSED"}:
        return (0, 165, 255)
    if state == "BOTH EYES CLOSED":
        return (0, 0, 255)

    return (220, 220, 220)


def get_distance_state_color(state):
    if state == "NORMAL":
        return (0, 255, 0)
    if state == "MOVING AWAY":
        return (0, 165, 255)
    if state == "TOO FAR":
        return (0, 0, 255)

    return (220, 220, 220)


def draw_alert_panel(frame, title, message, color):
    """Draw the temporary warning panel over the lower camera area."""
    frame_height, frame_width = frame.shape[:2]
    panel_top = max(frame_height - 115, 0)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, panel_top),
        (frame_width, frame_height),
        color,
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(
        frame,
        title,
        (20, panel_top + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        message,
        (20, panel_top + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def build_status_panel(
    frame_height,
    face_value,
    face_color,
    eye_value,
    eye_color,
    distance_value,
    distance_color,
    eye_cover_value,
    eye_cover_color,
    alert_value,
    alert_color,
    calibration_value,
    calibration_color,
    fps,
):
    """Build the separate right-side status panel."""
    panel_width = config.STATUS_PANEL_WIDTH

    panel = np.full(
        (frame_height, panel_width, 3),
        (28, 32, 38),
        dtype=np.uint8,
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel_width - 1, frame_height - 1),
        (0, 200, 255),
        1,
    )

    cv2.putText(
        panel,
        "SMART EYE MONITOR",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
    )

    cv2.line(
        panel,
        (16, 55),
        (panel_width - 16, 55),
        (90, 95, 100),
        1,
    )

    rows = [
        ("FACE", face_value, face_color),
        ("EYES", eye_value, eye_color),
        ("DISTANCE", distance_value, distance_color),
        ("EYE COVER", eye_cover_value, eye_cover_color),
        ("ALERT", alert_value, alert_color),
        ("CALIBRATION", calibration_value, calibration_color),
        ("FPS", f"{fps:.1f}", (255, 255, 255)),
    ]

    row_y = 90

    for label, value, color in rows:
        cv2.putText(
            panel,
            f"{label}:",
            (18, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (185, 190, 195),
            1,
        )

        cv2.putText(
            panel,
            value,
            (132, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )

        row_y += 32

    return panel
