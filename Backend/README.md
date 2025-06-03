# DeepFake Detection Telegram Bot

A powerful Telegram bot that can analyze images for deepfakes, perform reverse image searches to find real sources, and provide detailed content analysis using Google Vision API.

## Features

- **DeepFake Detection**: Analyzes images using MobileViTv2 model to identify manipulated media
- **Real Source Detection**: Performs reverse image search to find actual websites where the image appears (including news sources)
- **Face Analysis**: Detects and analyzes faces in the image
- **Content Recognition**: Identifies objects, scenes, and concepts in the image
- **Safety Check**: Flags potentially inappropriate content

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with the following:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token_here
   GOOGLE_VISION_API_KEY=your_google_vision_api_key_here
   SERP_API_KEY=your_serpapi_key_here
   ```

3. **Get API Keys**:
   - Telegram Bot Token: Talk to [@BotFather](https://t.me/botfather) on Telegram
   - Google Vision API: [Google Cloud Console](https://console.cloud.google.com/apis/library/vision.googleapis.com)
   - SerpAPI Key: [SerpAPI Website](https://serpapi.com/) (for real reverse image search)

4. **Ensure Model Files**:
   Make sure your model files are in the correct location:
   - `best_vit_model.pth` and/or `mvitv2.pkl` in the project root

## Usage

### Running the Bot

```bash
# Run just the Telegram bot
python run.py bot

# Run both the API server and the Telegram bot
python run.py all
```

### Talking to the Bot

1. Start a chat with your bot on Telegram
2. Send `/start` to initiate the conversation
3. Send any image to analyze it for:
   - DeepFake detection
   - Real websites where the image appears
   - Face analysis
   - Content labels and categories

## How It Works

1. **Image Upload**: User sends an image to the bot via Telegram
2. **DeepFake Analysis**: Image is analyzed using the MobileViTv2 model
3. **Vision Analysis**: Google Vision API extracts faces, content labels, and safety information
4. **Source Detection**: SerpAPI (or fallback methods) perform reverse image search to find real sources
5. **Response**: The bot returns a comprehensive analysis including deepfake probability, source websites, and content details

## Troubleshooting

- **"No module named 'serpapi'"**: Run `pip install serpapi`
- **API Key Errors**: Ensure all API keys in the `.env` file are correct
- **"Error performing reverse image search"**: Check your internet connection and SerpAPI key

## Credits

This bot uses:
- PyTorch with MobileViTv2 for deepfake detection
- Google Cloud Vision API for image analysis
- SerpAPI for reverse image search
- python-telegram-bot for the Telegram interface 