#!/usr/bin/env python3
"""
YouTube Auto-Clipper

This script downloads YouTube videos and automatically extracts motivational
and life advice content to create viral short-form clips.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import our modules
from youtube_downloader import download_youtube_video, download_youtube_playlist
from audio_analyzer import AudioAnalyzer, EngagingMoment
from video_processor import VideoProcessor
from viral_optimizer import ViralOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('youtube_auto_clipper')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YouTube Auto-Clipper: Generate motivational clips from YouTube videos'
    )
    parser.add_argument(
        '-u', '--url',
        type=str,
        help='YouTube URL to process'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=60,
        help='Maximum duration of clips in seconds (default: 60)'
    )
    parser.add_argument(
        '--max-clips',
        type=int,
        default=5,
        help='Maximum number of clips to generate (default: 5)'
    )
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='small.en',
        choices=['tiny.en', 'base.en', 'small.en', 'medium.en'],
        help='Whisper model size (default: small.en)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device to run models on (default: cuda)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import yt_dlp
        import faster_whisper
        import cv2
        import mediapipe
        import subprocess

        # Check if ffmpeg is installed
        result = subprocess.run(['ffmpeg', '-version'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode != 0:
            logger.error("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
            return False

        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required dependencies: pip install -r requirements.txt")
        return False

def prompt_for_url():
    """Prompt the user for a YouTube URL."""
    print("\n=== YouTube Auto-Clipper ===")
    print("This tool creates motivational short clips from YouTube videos.")
    print("Enter a YouTube URL of a podcast or interview video.")
    print("The tool will extract the most motivational and advice-rich segments.")

    while True:
        url = input("\nEnter YouTube URL: ").strip()
        if url:
            if "youtube.com" in url or "youtu.be" in url:
                return url
            else:
                print("Invalid URL. Please enter a valid YouTube URL.")
        else:
            print("URL cannot be empty. Please try again.")

def process_youtube_video(url, max_duration=60, max_clips=5, whisper_model="small.en", device="cuda", content_type="all"):
    """
    Process a YouTube video to create motivational clips.

    Args:
        url: YouTube URL
        max_duration: Maximum duration of clips in seconds
        max_clips: Maximum number of clips to generate
        whisper_model: Whisper model size to use
        device: Device to run models on ('cpu' or 'cuda')

    Returns:
        List of paths to created clips
    """
    start_time = time.time()
    logger.info(f"Processing YouTube video: {url}")

    print("\n" + "="*80)
    print("YOUTUBE AUTO-CLIPPER STARTING")
    print("="*80)
    print(f"Processing YouTube URL: {url}")
    print(f"Settings:")
    print(f"  - Maximum clip duration: {max_duration} seconds")
    print(f"  - Maximum number of clips: {max_clips}")
    print(f"  - Whisper model: {whisper_model}")
    print(f"  - Device: {device}")
    print("-"*80)

    try:
        # Step 1: Download the YouTube video
        logger.info("Step 1: Downloading YouTube video")
        print("\n" + "="*80)
        print("STEP 1: DOWNLOADING YOUTUBE VIDEO")
        print("="*80)
        video_path, video_info = download_youtube_video(url)

        if not video_path:
            logger.error("Failed to download video. Exiting.")
            print("ERROR: Failed to download video. Exiting.")
            return []

        video_dir = os.path.dirname(video_path)
        clips_dir = os.path.join(video_dir, "clips")

        print(f"Download complete!")
        print(f"Video title: {video_info.get('title', 'Unknown')}")
        print(f"Channel: {video_info.get('channel', 'Unknown')}")
        print(f"Duration: {video_info.get('duration', 0)/60:.2f} minutes")
        print(f"Saved to: {video_path}")

        # Step 2: Audio Analysis with Whisper
        logger.info("Step 2: Analyzing audio with Whisper")
        print("\n" + "="*80)
        print("STEP 2: ANALYZING AUDIO WITH WHISPER")
        print("="*80)
        audio_analyzer = AudioAnalyzer(model_size=whisper_model, device=device)

        # Determine whether to filter for specific content
        motivational_only = (content_type == "motivational")

        audio_moments = audio_analyzer.analyze(
            video_path,
            min_duration=5.0,
            max_duration=max_duration,
            max_gap=2.0,
            motivational_only=False  # Look for all engaging content including life advice and wisdom
        )

        print(f"PROGRESS: Looking for all engaging content including life advice, wisdom, and motivational segments")

        logger.info(f"Found {len(audio_moments)} motivational moments")
        print(f"Found {len(audio_moments)} motivational moments in the video")

        # Step 3: Optimize for viral potential
        logger.info("Step 3: Optimizing for viral potential")
        print("\n" + "="*80)
        print("STEP 3: OPTIMIZING FOR VIRAL POTENTIAL")
        print("="*80)
        viral_optimizer = ViralOptimizer()
        optimized_moments = viral_optimizer.optimize_moments(
            audio_moments,
            max_clips=max_clips
        )

        logger.info(f"Selected {len(optimized_moments)} moments for clips")

        # Step 4: Process Video and Create Clips
        logger.info("Step 4: Processing video and creating clips")
        print("\n" + "="*80)
        print("STEP 4: PROCESSING VIDEO AND CREATING CLIPS")
        print("="*80)
        video_processor = VideoProcessor(output_dir=clips_dir)

        if not optimized_moments:
            logger.warning("No motivational moments found. Creating a clip from the beginning of the video.")
            print("WARNING: No motivational moments found. Creating a clip from the beginning of the video.")
            # Create a default moment from the beginning of the video
            default_moment = EngagingMoment(
                start_time=0.0,
                end_time=min(30.0, max_duration),
                text="Default clip",
                confidence=0.0,
                keywords=[],
                score=0.0
            )
            optimized_moments = [default_moment]

        clip_paths = video_processor.process_moments(
            video_path,
            optimized_moments,
            max_clips=max_clips
        )

        # Step 5: Generate summary report
        logger.info("Step 5: Generating summary report")
        print("\n" + "="*80)
        print("STEP 5: GENERATING SUMMARY REPORT")
        print("="*80)

        # Create a summary report
        report_path = os.path.join(video_dir, "clip_summary.txt")
        print(f"Creating summary report: {report_path}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"YouTube Auto-Clipper Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Source: {url}\n")
            f.write(f"Video Title: {video_info.get('title', 'Unknown')}\n")
            f.write(f"Channel: {video_info.get('channel', 'Unknown')}\n\n")
            f.write(f"Created {len(clip_paths)} motivational clips:\n\n")

            for i, (path, moment) in enumerate(zip(clip_paths, optimized_moments)):
                clip_name = os.path.basename(path)
                f.write(f"Clip {i+1}: {clip_name}\n")
                f.write(f"  Duration: {moment.end_time - moment.start_time:.2f} seconds\n")
                f.write(f"  Start Time: {moment.start_time:.2f}s\n")
                f.write(f"  Viral Score: {moment.score:.2f}\n")
                f.write(f"  Content: {moment.text[:200]}...\n")

                # Extract key quote if available
                key_quote = viral_optimizer.extract_key_quote(moment.text)
                if key_quote:
                    f.write(f"  Key Quote: \"{key_quote}\"\n")

                f.write(f"  Keywords: {', '.join(moment.keywords[:10])}\n\n")

        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("AUTO-CLIPPER COMPLETE!")
        print("="*80)
        print(f"Created {len(clip_paths)} motivational clips:")
        for i, path in enumerate(clip_paths):
            clip_name = os.path.basename(path)
            print(f"  {i+1}. {clip_name}")

        print(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Clips saved to: {clips_dir}")
        print(f"Summary report: {report_path}")
        print("="*80)

        logger.info("\nClips created:")
        for i, path in enumerate(clip_paths):
            clip_name = os.path.basename(path)
            logger.info(f"  {i+1}. {clip_name}")

        logger.info(f"\nYouTube Auto-Clipper completed in {elapsed_time:.2f} seconds!")
        logger.info(f"Created {len(clip_paths)} motivational clips")
        logger.info(f"Clips saved to: {clips_dir}")
        logger.info(f"Summary report: {report_path}")

        return clip_paths

    except Exception as e:
        logger.error(f"Error processing YouTube video: {e}", exc_info=True)
        print(f"\nERROR: Processing failed: {e}")
        print("Check the logs for more details.")
        return []

def main():
    """Main entry point for the application."""
    args = parse_arguments()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting YouTube Auto-Clipper")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Get YouTube URL from command line or prompt
    url = args.url
    if not url:
        url = prompt_for_url()

    # Process the video
    process_youtube_video(
        url,
        max_duration=args.max_duration,
        max_clips=args.max_clips,
        whisper_model=args.whisper_model,
        device=args.device,
        content_type="all"  # Include all engaging content, not just motivational
    )

if __name__ == "__main__":
    main()
