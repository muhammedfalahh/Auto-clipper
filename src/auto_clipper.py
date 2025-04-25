#!/usr/bin/env python3
"""
Auto-Clipper: Podcast Video Clip Generator

This application automatically clips podcasts into short, engaging vertical videos
optimized for social media. It detects exciting moments in conversations and creates
clips under 60 seconds in vertical format (9:16 ratio).
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Import our modules
from audio_analyzer import AudioAnalyzer, EngagingMoment
from visual_analyzer import VisualAnalyzer
from video_processor import VideoProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('auto_clipper')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Auto-Clipper: Generate short, engaging clips from podcast videos'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='output',
        help='Directory to save output clips (default: output)'
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
        '--use-visual',
        action='store_true',
        help='Enable visual analysis with LLaVA-7B'
    )
    parser.add_argument(
        '--ollama-host',
        type=str,
        default='http://localhost:11434',
        help='Ollama API host (default: http://localhost:11434)'
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

def main():
    """Main entry point for the application."""
    args = parse_arguments()

    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Set log level for all modules
        logging.getLogger('auto_clipper.audio_analyzer').setLevel(logging.DEBUG)
        logging.getLogger('auto_clipper.visual_analyzer').setLevel(logging.DEBUG)
        logging.getLogger('auto_clipper.video_processor').setLevel(logging.DEBUG)

    logger.info("Starting Auto-Clipper")

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing video: {input_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using Whisper model: {args.whisper_model}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Maximum clip duration: {args.max_duration} seconds")
    logger.info(f"Maximum number of clips: {args.max_clips}")

    try:
        # Step 1: Audio Analysis with Whisper
        logger.info("Step 1: Analyzing audio with Whisper")
        audio_analyzer = AudioAnalyzer(model_size=args.whisper_model, device=args.device)
        audio_moments = audio_analyzer.analyze(
            str(input_path),
            min_duration=5.0,
            max_duration=args.max_duration,
            max_gap=2.0
        )

        logger.info(f"Found {len(audio_moments)} engaging audio moments")

        # Step 2: Visual Analysis with LLaVA-7B (if enabled)
        visual_moments = []
        if args.use_visual:
            logger.info("Step 2: Analyzing video with LLaVA-7B")
            visual_analyzer = VisualAnalyzer(
                ollama_host=args.ollama_host,
                model_name="llava:7b",
                frame_interval=5
            )

            visual_moments = visual_analyzer.find_visual_moments(
                str(input_path),
                confidence_threshold=0.6
            )

            logger.info(f"Found {len(visual_moments)} interesting visual moments")

            # Merge audio and visual moments
            if visual_moments:
                audio_moments = visual_analyzer.merge_with_audio_moments(
                    visual_moments,
                    audio_moments,
                    max_gap=2.0
                )
                logger.info(f"Combined into {len(audio_moments)} enhanced moments")

        # Step 3: Process Video and Create Clips
        logger.info("Step 3: Processing video and creating clips")
        video_processor = VideoProcessor(output_dir=str(output_dir))

        clip_paths = video_processor.process_moments(
            str(input_path),
            audio_moments,
            max_clips=args.max_clips
        )

        # Print summary
        logger.info("\nClips created:")
        for i, path in enumerate(clip_paths):
            clip_name = Path(path).name
            logger.info(f"  {i+1}. {clip_name}")

        logger.info(f"\nAuto-Clipper completed successfully! Created {len(clip_paths)} clips.")
        logger.info(f"Clips saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=args.debug)
        sys.exit(1)

if __name__ == "__main__":
    main()
