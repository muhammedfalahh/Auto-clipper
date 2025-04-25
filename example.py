#!/usr/bin/env python3
"""
Example script demonstrating how to use Auto-Clipper programmatically.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import Auto-Clipper modules
from audio_analyzer import AudioAnalyzer, EngagingMoment
from video_processor import VideoProcessor

def simple_example(input_video, output_dir="output", max_duration=60, max_clips=3):
    """
    Simple example of using Auto-Clipper to generate clips.
    
    Args:
        input_video: Path to the input video file
        output_dir: Directory to save output clips
        max_duration: Maximum duration of clips in seconds
        max_clips: Maximum number of clips to generate
    """
    print(f"Processing video: {input_video}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Audio Analysis with Whisper
    print("Step 1: Analyzing audio with Whisper")
    audio_analyzer = AudioAnalyzer(model_size="small.en", device="cuda")
    audio_moments = audio_analyzer.analyze(
        input_video,
        min_duration=5.0,
        max_duration=max_duration,
        max_gap=2.0
    )
    
    print(f"Found {len(audio_moments)} engaging audio moments")
    
    # Step 2: Process Video and Create Clips
    print("Step 2: Processing video and creating clips")
    video_processor = VideoProcessor(output_dir=output_dir)
    
    clip_paths = video_processor.process_moments(
        input_video,
        audio_moments,
        max_clips=max_clips
    )
    
    # Print summary
    print("\nClips created:")
    for i, path in enumerate(clip_paths):
        clip_name = Path(path).name
        print(f"  {i+1}. {clip_name}")
    
    print(f"\nAuto-Clipper example completed! Created {len(clip_paths)} clips.")
    print(f"Clips saved to: {output_dir}")

if __name__ == "__main__":
    # Check if input video is provided
    if len(sys.argv) < 2:
        print("Usage: python example.py <input_video> [output_dir] [max_duration] [max_clips]")
        sys.exit(1)
    
    # Get arguments
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    max_duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    max_clips = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    # Run the example
    simple_example(input_video, output_dir, max_duration, max_clips)
