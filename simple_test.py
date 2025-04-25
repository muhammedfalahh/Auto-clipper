#!/usr/bin/env python3
"""
Simple test script to create a short clip from a video.
"""

import os
import subprocess
import sys

def create_simple_clip(input_video, output_dir="output", start_time=0, duration=10):
    """
    Create a simple clip from a video without any analysis.
    
    Args:
        input_video: Path to the input video file
        output_dir: Directory to save output clips
        start_time: Start time of the clip in seconds
        duration: Duration of the clip in seconds
    """
    print(f"Creating a simple clip from {input_video}")
    print(f"Start time: {start_time}s, Duration: {duration}s")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path
    output_filename = f"simple_clip_{start_time}_{duration}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Build the FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_video,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]
    
    # Execute the command
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
            
        print(f"Clip created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating clip: {e}")
        return None

if __name__ == "__main__":
    # Check if input video is provided
    if len(sys.argv) < 2:
        print("Usage: python simple_test.py <input_video> [output_dir] [start_time] [duration]")
        sys.exit(1)
    
    # Get arguments
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    start_time = float(sys.argv[3]) if len(sys.argv) > 3 else 0
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    # Create a simple clip
    create_simple_clip(input_video, output_dir, start_time, duration)
