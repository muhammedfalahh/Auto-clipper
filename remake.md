# Auto-Clipper Enhancement Plan: YouTube Integration for Motivational Content

## Overview
Enhance the Auto-Clipper to download YouTube podcast/interview videos and automatically extract the most motivational, inspiring, and advice-rich segments to create viral short-form content.

## Core Features to Add

### 1. YouTube Integration with yt-dlp
- Add functionality to accept YouTube URLs as input
- Implement yt-dlp for high-quality, efficient video downloads
- Support for various YouTube video formats and resolutions
- Handle YouTube playlists for batch processing

### 2. Organized Content Structure
- Create a folder structure based on the video title/channel
- Format: `videos/{video_title}/`
- Store both the original downloaded video and generated clips in this folder
- Include metadata files with source information

### 3. Enhanced Content Detection for Motivational Content
- Expand the `ENGAGEMENT_KEYWORDS` in `AudioAnalyzer` to include:
  - Motivational terms: "success", "overcome", "achieve", "goal", "dream", "inspire"
  - Life advice indicators: "lesson", "advice", "learn", "experience", "wisdom"
  - Emotional triggers: "changed my life", "breakthrough", "revelation", "epiphany"
  - Storytelling markers: "story", "when I was", "happened to me", "realized"

### 4. Viral Content Optimization
- Prioritize segments with:
  - Strong emotional content (detected via tone analysis)
  - Clear, concise advice (single point delivery)
  - Powerful quotes or statements
  - Personal stories with universal lessons
- Implement a scoring system that weighs these factors

### 5. User Interface Improvements
- Simple command-line interface to input YouTube URLs
- Progress indicators for download and processing
- Summary of detected motivational segments
- Preview options for clips before final rendering

## Implementation Plan

### Phase 1: YouTube Integration
1. Add yt-dlp as a dependency in requirements.txt
2. Create a new module `youtube_downloader.py` to handle:
   - URL validation
   - Video metadata extraction
   - Download management
   - Folder structure creation

### Phase 2: Content Detection Enhancement
1. Modify `audio_analyzer.py` to:
   - Expand keyword lists for motivational content
   - Implement more sophisticated engagement detection
   - Add tone analysis for emotional content
2. Update scoring algorithm to prioritize motivational segments

### Phase 3: User Experience
1. Create a new main script `youtube_auto_clipper.py` that:
   - Prompts for YouTube URL
   - Handles the end-to-end process
   - Provides clear feedback and progress updates

### Phase 4: Output Optimization
1. Enhance `video_processor.py` to:
   - Add custom intro/outro templates for motivational content
   - Implement text overlay for key quotes
   - Optimize aspect ratio specifically for different platforms (TikTok, YouTube Shorts, Instagram Reels)

## Code Structure Changes

### New Files to Create:
1. `youtube_downloader.py` - YouTube integration module
2. `youtube_auto_clipper.py` - Main script for YouTube workflow
3. `viral_optimizer.py` - Enhanced scoring and optimization for viral content

### Files to Modify:
1. `audio_analyzer.py` - Enhance keyword detection for motivational content
2. `video_processor.py` - Add viral-optimized formatting
3. `requirements.txt` - Add yt-dlp and other dependencies

## Sample Workflow

1. User runs: `python youtube_auto_clipper.py`
2. Program prompts: "Enter YouTube URL of podcast/interview:"
3. User enters URL
4. Program:
   - Downloads video to `videos/{video_title}/original.mp4`
   - Analyzes content for motivational segments
   - Creates 5-10 optimized clips in `videos/{video_title}/clips/`
   - Generates a summary report of extracted content

## Technical Considerations

### yt-dlp Integration
```python
def download_youtube_video(url, output_dir):
    """Download a YouTube video using yt-dlp."""
    import yt_dlp
    
    # Options for yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, 'original.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
    }
    
    # Download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info
```

### Motivational Content Detection
Enhance the existing `ENGAGEMENT_KEYWORDS` list with motivational terms:

```python
MOTIVATIONAL_KEYWORDS = [
    # Success and achievement
    "success", "achieve", "accomplish", "overcome", "victory", "win",
    
    # Growth and improvement
    "grow", "improve", "better", "change", "transform", "develop",
    
    # Goals and dreams
    "goal", "dream", "vision", "aspire", "ambition", "purpose",
    
    # Inspiration and motivation
    "inspire", "motivate", "encourage", "drive", "passion", "energy",
    
    # Life advice
    "advice", "lesson", "wisdom", "experience", "insight", "perspective",
    
    # Emotional impact
    "changed my life", "breakthrough", "revelation", "epiphany", "realization",
    
    # Storytelling markers
    "story", "when I was", "happened to me", "realized", "discovered"
]
```

## Timeline
- Phase 1 (YouTube Integration): 1-2 days
- Phase 2 (Content Detection Enhancement): 2-3 days
- Phase 3 (User Experience): 1 day
- Phase 4 (Output Optimization): 2-3 days

Total estimated development time: 6-9 days

## Future Enhancements
- Web interface for easier interaction
- Batch processing of multiple videos
- Custom templates for different content types
- Social media direct posting integration
- Analytics on which clips perform best







## how to use 

- Activate the virtual environment:
   .venv\Scripts\activate

- Run the YouTube Auto-Clipper
   python youtube_auto_clipper.py

- Enter a YouTube URL when prompted, or provide it as a command-line argument.
   python youtube_auto_clipper.py --url https://www.youtube.com/watch?v=EXAMPLE

- Customize the output with command-line options
   python youtube_auto_clipper.py --url https://www.youtube.com/watch?v=EXAMPLE --max-clips 10 --max-duration 45 --device cpu


## results

Find your clips in the videos/{video_title}/clips/ directory
The Auto-Clipper will:

Download the YouTube video
Analyze it for motivational and life advice content
Select the most viral-worthy segments
Create vertical (9:16) clips optimized for social media
Generate a summary report with details about each clip