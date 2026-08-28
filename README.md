# Smart Eye Monitor

Smart Eye Monitor is a beginner-friendly Python application that uses a webcam to monitor face visibility, eye state, and distance from the camera. It provides on-screen status information and plays configurable voice alerts when it detects situations that need attention.

> This is an educational computer-vision project. It is not a medical device or a certified safety system.

## Features

- Detects whether a face is visible in the webcam feed.
- Detects open and closed eyes.
- Identifies left-eye, right-eye, or both-eye coverage.
- Detects when the face is covered by an object.
- Detects when a person moves too far from the camera or leaves the frame.
- Supports calibration for normal eye openness and face distance.
- Shows a separate status panel without covering the webcam image.
- Plays customizable voice alerts for monitored events.
- Saves activity events to a local log file.

## Technologies Used

- **Python**
- **OpenCV** for webcam capture and display
- **MediaPipe** for face, hand, and pose landmark detection
- **NumPy** for numerical calculations
- **Pygame** for alert audio playback

## Project Structure

```text
smart_eye_monitor/
|-- main.py                 # Starts the application
|-- config.py               # Settings, thresholds, and file paths
|-- requirements.txt        # Python dependencies
|-- alerts/                 # Sound-alert handling
|-- ui/                     # Webcam and status-panel display code
|-- utils/                  # Logging helpers
|-- models/                 # MediaPipe .task model files
|-- sounds/                 # Customizable WAV alert sounds
|-- logs/                   # Generated event logs
`-- README.md
```

## Installation (Windows PowerShell)

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR-USERNAME/smart-eye-monitor.git
Set-Location .\smart-eye-monitor
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following command for the current PowerShell window and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

With the virtual environment activated, run:

```powershell
python main.py
```

Allow camera access if Windows asks for permission. The webcam window and its status panel will open.

## Controls

| Key | Action |
| --- | --- |
| `C` | Start calibration. Sit normally and keep both eyes open. |
| `Q` | Quit the application and release the camera. |

## Alert Sounds

The project uses custom voice alerts to make monitoring more engaging. They are implemented as configurable WAV files, so you can choose professional, humorous, or personalized recordings for each alert.

The sound files are stored in the `sounds/` folder:

| File | Used for |
| --- | --- |
| `eye_closed.wav` | Eyes-closed alert |
| `eye_covered.wav` | Eye-coverage alert |
| `face_covered.wav` | Face-covered alert |
| `moved_away.wav` | Person moved away or is no longer visible |

### Replace an alert sound

1. Record or download a WAV audio file.
2. Rename it to the alert filename you want to replace, such as `eye_closed.wav`.
3. Place it in the `sounds/` folder and replace the existing file.
4. Restart the application.

Keep the filename the same unless you also update the relevant path in `config.py`.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Activate the virtual environment and reinstall the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### The camera does not open

- Check that no other application, such as Teams, Zoom, or a browser tab, is using the webcam.
- Check Windows camera privacy settings and allow desktop apps to use the camera.
- Confirm that the selected camera index in `config.py` matches your webcam.

### MediaPipe model file is missing

Ensure the following files exist in the `models/` folder:

```text
face_landmarker.task
hand_landmarker.task
pose_landmarker_lite.task
```

The filenames must match the paths configured in `config.py`. If you cloned the repository, pull the latest changes or restore the missing model file from the project source.

### Alert sound does not play

- Confirm the required WAV file exists in the `sounds/` folder.
- Check that your system volume is on.
- Confirm that no other application has exclusive control of the audio device.

## Limitations

- Detection quality depends on lighting, camera angle, camera quality, and system performance.
- Hands, masks, glasses, or other objects may occasionally affect face and eye detection.
- The application is intended for learning and demonstration purposes only. Do not rely on it for medical decisions, emergency response, or safety-critical monitoring.

## Future Improvements

- Add a graphical settings screen for camera and alert thresholds.
- Allow users to select alert sounds from inside the application.
- Add an event-history viewer or exportable report.
- Improve performance for lower-powered computers.
- Add support for multiple camera devices.

## License

No license has been selected yet. Add a license file before distributing or reusing this project publicly.
