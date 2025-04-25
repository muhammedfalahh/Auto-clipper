# Auto-Clipper: AI-Powered Video Clip Generator

Auto-Clipper is an advanced tool that automatically creates engaging, vertical (9:16) short-form clips from horizontal (16:9) videos. It uses AI to detect the most engaging and motivational moments in videos, focusing on life advice, inspirational content, and encouraging messages.

## Features

- **YouTube Integration**: Download videos directly from YouTube URLs
- **Content Detection**: Identifies segments containing life advice, wisdom, motivation, and encouragement
- **Face-Centered Cropping**: Uses face detection to ensure subjects remain in frame
- **Viral Optimization**: Scores and selects the most viral-worthy moments based on:
  - Emotional impact
  - Advice value
  - Quote potential
  - Storytelling elements
- **Vertical Format**: Automatically converts horizontal videos to vertical (9:16) format for social media
- **Detailed Progress Reporting**: Shows real-time progress of each processing step

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg installed and available in PATH
- CUDA-compatible GPU (recommended for faster processing)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Auto-clipper.git
   cd Auto-clipper
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv

   # On Windows
   .venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Processing YouTube Videos

The easiest way to use Auto-Clipper is with the YouTube integration:

1. Activate the virtual environment:
   ```
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. Run the YouTube Auto-Clipper:
   ```
   # On Windows PowerShell
   $env:KMP_DUPLICATE_LIB_OK = "TRUE"
   python youtube_auto_clipper.py

   # Or use the batch file (Windows)
   .\run_youtube_clipper.bat

   # On macOS/Linux
   KMP_DUPLICATE_LIB_OK=TRUE python youtube_auto_clipper.py
   ```

3. When prompted, enter a YouTube URL of a podcast or interview video.

4. The tool will:
   - Download the video
   - Analyze it for motivational content
   - Select the most viral-worthy segments
   - Create vertical clips
   - Generate a summary report

### Command Line Options

You can customize the Auto-Clipper with various command-line options:

```
python youtube_auto_clipper.py -u "https://www.youtube.com/watch?v=EXAMPLE" --max-clips 10 --max-duration 45 --device cpu
```

Available options:
- `-u, --url`: YouTube URL to process
- `--max-duration`: Maximum duration of clips in seconds (default: 60)
- `--max-clips`: Maximum number of clips to generate (default: 5)
- `--whisper-model`: Whisper model size to use (tiny.en, base.en, small.en, medium.en)
- `--device`: Device to run models on (cpu or cuda)
- `--debug`: Enable debug logging

### Processing Local Videos

You can also process videos already on your computer:

```
python run_auto_clipper.py
```

This will process the video at `input/test.mp4` and create clips in the `output` directory.

## Output

The Auto-Clipper creates the following outputs:

1. **Video Clips**: Saved in `videos/{video_title}/clips/` directory
2. **Summary Report**: A text file with details about each clip in `videos/{video_title}/clip_summary.txt`
3. **Original Video**: The downloaded video is saved in `videos/{video_title}/original.mp4`
4. **Metadata**: Video information is saved in `videos/{video_title}/metadata.json`

## How It Works

1. **Video Download**: Uses yt-dlp to download videos from YouTube
2. **Audio Analysis**: Uses Whisper to transcribe and analyze the audio
3. **Content Detection**: Identifies motivational and engaging segments
4. **Viral Optimization**: Scores segments based on viral potential
5. **Face Detection**: Uses MediaPipe to detect and track faces
6. **Clip Creation**: Uses FFmpeg to crop and create the final clips

## Troubleshooting

### OpenMP Warning

If you see an OpenMP warning like:
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

Use the environment variable to suppress it:
```
$env:KMP_DUPLICATE_LIB_OK = "TRUE"  # PowerShell
set KMP_DUPLICATE_LIB_OK=TRUE  # CMD
export KMP_DUPLICATE_LIB_OK=TRUE  # Bash
```

### CUDA Issues

If you encounter CUDA-related errors, try using the CPU instead:
```
python youtube_auto_clipper.py --device cpu
```

### FFmpeg Not Found

Make sure FFmpeg is installed and in your PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Whisper](https://github.com/openai/whisper) for audio transcription
- [MediaPipe](https://github.com/google/mediapipe) for face detection
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [FFmpeg](https://ffmpeg.org/) for video processing