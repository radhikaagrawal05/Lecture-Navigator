# Chrome Extension for LectureFind

This folder contains a simple Chrome extension that calls the local backend at `http://localhost:8000/analyze`.

## Install
1. Open `chrome://extensions`
2. Enable `Developer mode`
3. Click `Load unpacked`
4. Select this folder: `chrome-extension`

## Usage
- Open a YouTube video page.
- Click the extension icon.
- Enter a concept and search.
- Click a clip to jump the video to that timestamp.

## Notes
- The backend must be running on `http://localhost:8000`.
- The extension uses the same transcript search API as the web frontend.
