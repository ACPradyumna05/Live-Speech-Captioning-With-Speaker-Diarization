# 🎙️ Live Speaker-Diarized Captioning App

> **Real-time transcription & speaker identification from microphone input using Whisper, Wav2Vec2, and Resemblyzer**

![demo](https://img.shields.io/badge/Status-Under%20Active%20Development-orange) ![Python](https://img.shields.io/badge/Made%20with-Python-blue) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

This application captures live microphone input, transcribes speech using state-of-the-art ASR models (Whisper, Wav2Vec2), and performs **speaker diarization** using Resemblyzer. Captions are emitted in real-time to a web UI via WebSockets.

### ✨ Features

* 🎧 Live microphone audio streaming
* 🗣️ Real-time transcription (Whisper / Wav2Vec2)
* 👥 Speaker diarization with clustering
* 🖥️ Web interface with live captions
* 🔄 Dynamic model switching at runtime
* 🔌 WebSocket communication for instant updates

---

## 📸 Preview


> ![ezgif-4-6a5ec28529](https://github.com/user-attachments/assets/47cd5d90-37b8-401e-8c35-1afb3739fa14)
> ![image](https://github.com/user-attachments/assets/40553152-616f-49b3-8541-2b3e8774f4ff)



---

## 🛠️ Tech Stack

| Purpose               | Technology Used                 |
| --------------------- | ------------------------------- |
| Backend Server        | Flask + Flask-SocketIO          |
| Speech-to-Text Models | OpenAI Whisper / Wav2Vec2       |
| Speaker Diarization   | Resemblyzer + KMeans Clustering |
| Audio Input           | SoundDevice (PySoundDevice)     |
| Web Frontend          | HTML + JS (via `front.html`)    |

---

## 🧰 Installation

### 🔗 Prerequisites

* Python 3.8+
* Git, pip

### 📦 Setup

```bash
git clone https://github.com/yourusername/live-captioning-diarization.git
cd live-captioning-diarization

# Install dependencies
pip install -r requirements.txt
```

> **Note:** You may need additional dependencies for `sounddevice` and `torch` depending on your system.

---

## 🧪 Run the App

```bash
python app.py
```

Now open your browser and go to [http://localhost:5000](http://localhost:5000)
You’ll see the live caption interface ready to use!

---

## ⚙️ Available Models

| Model Key       | Description                                |
| --------------- | ------------------------------------------ |
| `whisper_base`  | Fast, lightweight ASR model (Whisper base) |
| `whisper_large` | High-accuracy Whisper large-v2 model       |
| `wav2vec2`      | Facebook’s Wav2Vec2 large-960h CTC model   |

You can switch models live via the web UI. Model switching is thread-safe and updates all clients in real time.

---

## 🧠 How It Works

1. **Audio Recording**: Audio chunks of 10 seconds are captured.
2. **Transcription**: Selected model transcribes the audio.
3. **Diarization**: Resemblyzer embeds speaker segments → KMeans clusters them.
4. **Captioning**: Captions are segmented by speaker and timestamped.
5. **Streaming**: Captions are pushed to the client via WebSocket (`/caption` event).

---

## 📁 Project Structure

```
.
├── app.py                # Main Flask server
├── templates/
│   └── front.html        # Web interface
├── static/               # (Optional: static files for frontend)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🔒 Security Notes

* This demo uses `SECRET_KEY = 'your-secret-key-here'`. Replace it for production use.
* Live audio processing may use significant system resources and model downloads.

---

## 📈 Roadmap

* [ ] Add speaker labels with memory across sessions
* [ ] Add speaker color tags on frontend
* [ ] Support live audio via browser mic (WebRTC)
* [ ] Transcription export (TXT/SRT)

---

## 🧑‍💻 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.

---

## 📄 License

[MIT](LICENSE)

---

## 💬 Acknowledgements

* [OpenAI Whisper](https://github.com/openai/whisper)
* [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [Flask-SocketIO](https://flask-socketio.readthedocs.io/)

---

Let me know if you'd like a **custom badge**, **frontend UI preview**, or **deployment instructions** (e.g., Docker, Heroku, etc.) added!
