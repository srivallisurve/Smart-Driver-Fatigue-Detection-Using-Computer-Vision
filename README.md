# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **MediaPipe Face Mesh** and **OpenCV**. It monitors the driver's eyes and mouth to detect signs of drowsiness and triggers an alarm to alert them.

## ✨ Features

- **Real-time face landmark detection** using MediaPipe Face Mesh (468 landmarks)
- **Eye Aspect Ratio (EAR)** monitoring to detect eye closure
- **Mouth Aspect Ratio (MAR)** monitoring to detect yawning
- **Drowsiness scoring system** with 3 states: `SAFE`, `WARNING`, `DROWSY`
- **Audio alarm** that loops continuously when drowsiness is detected
- **Startup delay** to prevent false detections during initialization

## 📁 Project Structure

```
drivers_drowsiness/
├── main.py              # Main application with webcam + detection loop
├── utils.py             # EAR and MAR calculation functions
├── config.py            # Threshold values and configuration
├── requirements.txt     # Python dependencies
├── alarm.wav            # Alarm sound file
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.10 – 3.12
- Webcam

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drivers_drowsiness.git
   cd drivers_drowsiness
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add an alarm sound**
   Place an `alarm.wav` file in the project root directory.

## 🚀 Usage

```bash
python main.py
```

- The webcam will open with a live feed
- EAR, MAR, Score, and Status are displayed on screen
- Press **`q`** to exit

## ⚙️ Configuration

You can adjust detection sensitivity in `config.py`:

| Parameter          | Default | Description                               |
|--------------------|---------|-------------------------------------------|
| `EAR_THRESHOLD`    | 0.25    | Eye closure threshold (lower = more closed) |
| `MAR_THRESHOLD`    | 0.65    | Yawn detection threshold                  |
| `FRAME_THRESHOLD`  | 15      | Consecutive closed-eye frames to trigger  |
| `DROWSY_SCORE_LIMIT` | 15   | Score limit for DROWSY state              |
| `WARNING_SCORE_LIMIT` | 7    | Score limit for WARNING state             |

## 🧠 How It Works

1. **Face Mesh Detection** — MediaPipe detects 468 facial landmarks in real-time
2. **EAR Calculation** — Measures eye openness using 6 landmarks per eye
3. **MAR Calculation** — Measures mouth openness using 8 landmarks
4. **Scoring** — Closed eyes and yawning increment a drowsiness score
5. **Alert** — When the score exceeds thresholds, status changes and alarm plays

## 🔧 Tech Stack

- **OpenCV** — Video capture and display
- **MediaPipe** — Face mesh landmark detection
- **Pygame** — Audio alarm playback
- **NumPy** — Mathematical calculations
