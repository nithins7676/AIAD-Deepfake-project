# 🚀 Deepfake Detection Extension + Bot 🎯  
> **"Reality is your best armor."**

---

## 🌌 About

**Deepfake Detection** is a blazing-fast Chrome Extension + Telegram Bot that detects AI-generated fake images and videos in real-time.

✅ No installations  
✅ No complex setup  
✅ Download → Load into Chrome or start the bot → Detect Deepfakes Instantly

---

## ⚡ Quick Access

| 👉 | **Chrome Extension** |
|----|----------------------|
| 📥 | [Download Here (MediaFire)](https://www.mediafire.com/file/wro2f0ybid2kz66/deepfake_extension.crx/file) |

| 👉 | **Visit Our Website** |
|----|-----------------------|
| 🌐 | [v0dev - Official Site](https://v0-deepfake-video-detector-9klunz.vercel.app/) |

---

## 🧩 How to Install the Chrome Extension

1. **Download** the `.crx` or `.zip` file from the link above  
2. Open Chrome → Go to `chrome://extensions/`  
3. Turn ON **Developer Mode** (top-right)  
4. Click **Load unpacked** → Select the **extracted folder**  
5. 🎯 Done! Start detecting deepfakes in real time

> 📖 More instructions + demo videos on [v0dev.site](https://v0-deepfake-video-detector-9klunz.vercel.app/)

---

## 🌐 Backend API (for Extension)

> Powered by FastAPI + Uvicorn

### 📄 File: `main.py`

This is the REST API backend that powers the Chrome extension.

### ▶️ How to Run the Extension Server
```bash
pip install -r requirements.txt

# Then run the server:
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 🔁 What It Does

- Accepts image or video uploads from the extension  
- Processes files using trained deepfake models  
- Returns detection results in real-time

> ⚠️ Make sure model files are present and properly loaded within `main.py`.

---

## 🤖 Telegram Bot Backend

> **"Because deepfakes don't announce themselves."**

### 📁 Folder: `backend/`

Includes:
- `deepfake_bot.py` — Telegram bot logic  
- `requirements.txt` — Python dependencies  
- `models.txt` — Direct download links for pretrained models  
- `.env` — Required for API keys (not included)

---

### 🛠️ Setup Guide

#### 1. Environment Setup
```bash
cd backend
python -m venv .venv

# Activate:
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Create `.env` File
Inside `backend/`, create a `.env` file:
```env
TELEGRAM_TOKEN=your_telegram_bot_token
GOOGLE_VISION_API_KEY=your_google_vision_api_key
SERP_API_KEY=your_serp_api_key
```

#### 4. Download Model Files

Model download links are listed in [`models.txt`](backend/models.txt).  
Download and place them in `backend/`.

---

### ▶️ Run the Bot

```bash
python deepfake_bot.py
```

Start chatting with the bot on Telegram:  
Send `/start`, then upload an image → Get instant deepfake results!

---

## 🌟 Features

- 🧠 AI-Powered Deepfake Detection
- 🖼️ Image & Video File Scanning
- ⚡ Instant Results (Few Seconds)
- 🔒 100% Local Processing (Privacy First)
- 🌐 Works on WhatsApp Web, Instagram, Telegram, Facebook

---

## 🛠️ Tech Stack

| Layer      | Tools Used                          |
|------------|-------------------------------------|
| Frontend   | React.js, Tailwind CSS              |
| Extension  | JavaScript + API (FastAPI)          |
| Backend    | TensorFlow, FastAPI, Uvicorn |
| Models     | ViT, MViTv2   |

---

## 📈 Performance

| Metric              | Value         |
|---------------------|---------------|
| 🎯 Accuracy         | >90%          |
| ⚡ Detection Speed  | <5s (Images)  |
| 🎥 Video Support    | Short Clips   |
| 🔒 Privacy          | 100% Local    |

---

## 🔮 Future Plans

- 📱 Mobile App Version  
- 🎙️ Audio Deepfake Detection  
- 🚨 Enhanced Threat Alerts & Reports

---

## 🛡️ License

Licensed under the **MIT License**.  
You are free to fork, remix, and contribute.

---

## 🕹️ Stay Real. Stay Sharp.

**Developed by People Who Can’t Trust Their Own Eyes 👀 | Stay Real 🌌**
