#!/usr/bin/env python3
"""
Run Auto-Clipper with the correct Python path.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import necessary modules
from audio_analyzer import EngagingMoment
from video_processor import VideoProcessor

# Input and output paths
input_video = os.path.join("input", "test.mp4")
output_dir = "output"

print(f"Running Auto-Clipper with input video: {input_video}")
print(f"Output directory: {output_dir}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create a video processor
video_processor = VideoProcessor(output_dir=output_dir)

# Create 5 manual engaging moments at different parts of the video
print("Creating 5 manual engaging moments at different parts of the video")
manual_moments = [
    EngagingMoment(
        start_time=0.0,
        end_time=20.0,
        text="Clip 1",
        confidence=1.0,
        keywords=[],
        score=1.0
    ),
    EngagingMoment(
        start_time=30.0,
        end_time=50.0,
        text="Clip 2",
        confidence=1.0,
        keywords=[],
        score=1.0
    ),
    EngagingMoment(
        start_time=60.0,
        end_time=80.0,
        text="Clip 3",
        confidence=1.0,
        keywords=[],
        score=1.0
    ),
    EngagingMoment(
        start_time=90.0,
        end_time=110.0,
        text="Clip 4",
        confidence=1.0,
        keywords=[],
        score=1.0
    ),
    EngagingMoment(
        start_time=120.0,
        end_time=140.0,
        text="Clip 5",
        confidence=1.0,
        keywords=[],
        score=1.0
    )
]

# Process the moments to create clips
print("Processing the moments to create 5 clips")
clip_paths = video_processor.process_moments(
    input_video,
    manual_moments,
    max_clips=5
)

# Print summary
if clip_paths:
    print("\nClip created:")
    for i, path in enumerate(clip_paths):
        clip_name = Path(path).name
        print(f"  {i+1}. {clip_name}")
    print(f"\nClip saved to: {output_dir}")
else:
    print("\nNo clips were created. Check the logs for errors.")
