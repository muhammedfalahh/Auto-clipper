# Auto-Clipper: Podcast Video Clip Generator

## Project Purpose
This application automatically clips podcasts into short, engaging vertical videos optimized for social media. It detects exciting moments in conversations (interesting topics, laughter, mindset discussions, fun facts) and creates clips under 60 seconds in vertical format (9:16 ratio).

## Core Technologies
- **FFmpeg + Whisper**: For speech/text-based clipping and video processing
- **LLaVA-7B**: For visual analysis and moment detection
- **Face Detection**: To intelligently crop horizontal (16:9) videos to vertical (9:16) format

### Why This Works Best
✅ Low VRAM Usage – Whisper and LLaVA-7B can run efficiently on RTX 4060
✅ No Cloud Dependency – Fully local and open-source
✅ Precise Clipping – Combines audio and visual cues for meaningful segments

## Implementation Guide

### 1. Install Required Tools
- **FFmpeg** - For video processing and format conversion
- **Python** - For scripting automation
- **faster-whisper** - GPU-optimized version of Whisper for transcription
- **LLaVA-7B via Ollama** - For visual content analysis
- **Face detection library** - For intelligent cropping (e.g., MediaPipe or OpenCV)

```bash
# Core dependencies
pip install faster-whisper opencv-python mediapipe librosa pyscenedetect
```

### 2. Multi-Modal Analysis Pipeline

#### Audio Analysis with Whisper
```python
from faster_whisper import WhisperModel

model = WhisperModel("small.en", device="cuda")  # RTX 4060 can handle "medium.en" too
segments, _ = model.transcribe("podcast.mp4", word_timestamps=True)

# Identify engaging moments based on speech content
engaging_moments = []
for segment in segments:
    text = segment.text.lower()
    # Detect interesting topics, laughter, mindset discussions, fun facts
    if any(keyword in text for keyword in ["interesting", "funny", "mindset", "fact", "wow", "amazing"]):
        engaging_moments.append((segment.start, segment.end, text))
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

#### Visual Analysis with LLaVA-7B
```python
# Extract frames at regular intervals
# Send to LLaVA-7B via Ollama API
# Detect visual cues: speaker changes, facial expressions, laughter
# Combine with audio cues for better clip selection
```

### 3. Intelligent Video Processing

#### Face Detection and Vertical Cropping
```python
import cv2
import mediapipe as mp

# Detect faces in video frames
# Track face positions throughout clip
# Determine optimal 9:16 crop window that keeps faces centered
```

#### Generate Clips with FFmpeg
```python
import subprocess

# For each engaging moment:
# 1. Extract clip (ensuring <60 seconds)
# 2. Apply vertical crop based on face positions
# 3. Add any enhancements (audio normalization, etc.)

for i, (start, end, _) in enumerate(engaging_moments):
    # Ensure clip is under 60 seconds
    duration = min(end - start, 60)
    end = start + duration
    
    # Apply vertical crop with face-aware positioning
    cmd = f"ffmpeg -i podcast.mp4 -ss {start} -to {end} -vf 'crop=ih*9/16:ih:x:0' -c:v libx264 -c:a aac clip_{i}.mp4"
    subprocess.run(cmd, shell=True)
```

### 4. Advanced Features
- **Audio Energy Analysis** - Use librosa to detect emotional high points
- **Scene Detection** - Use PySceneDetect for visual cuts
- **Clip Ranking** - Score and prioritize the most engaging clips
- **Batch Processing** - Process multiple podcast episodes

## Performance Notes for RTX 4060
- **Whisper Model Choice**:
  - tiny.en / small.en → Fastest (good for quick cuts)
  - medium.en → More accurate (still runs well on 8GB VRAM)
- **LLaVA-7B**: Can analyze video frames to detect key visual moments
- **FFmpeg**: Uses almost no GPU, so it won't slow down other processes
