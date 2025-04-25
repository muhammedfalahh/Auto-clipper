#!/usr/bin/env python3
"""
Test script for Auto-Clipper with more verbose logging.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_auto_clipper')

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import Auto-Clipper modules
from audio_analyzer import AudioAnalyzer, EngagingMoment
from video_processor import VideoProcessor

def test_auto_clipper(input_video, output_dir="output", max_duration=30, max_clips=1):
    """
    Test Auto-Clipper with more verbose logging.
    
    Args:
        input_video: Path to the input video file
        output_dir: Directory to save output clips
        max_duration: Maximum duration of clips in seconds
        max_clips: Maximum number of clips to generate
    """
    start_time = time.time()
    logger.info(f"Starting Auto-Clipper test with input: {input_video}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max duration: {max_duration}s, Max clips: {max_clips}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Audio Analysis with Whisper
        logger.info("Step 1: Starting audio analysis with Whisper")
        audio_analyzer = AudioAnalyzer(model_size="tiny.en", device="cpu")  # Using smaller model on CPU for testing
        
        logger.info("Loading Whisper model...")
        audio_analyzer.load_model()
        logger.info("Whisper model loaded successfully")
        
        logger.info("Transcribing audio...")
        segments = audio_analyzer.transcribe(input_video)
        logger.info(f"Transcription completed: {len(segments)} segments found")
        
        logger.info("Finding engaging moments...")
        moments = audio_analyzer.find_engaging_moments(
            segments,
            min_duration=3.0,  # Shorter for testing
            max_duration=max_duration
        )
        logger.info(f"Found {len(moments)} engaging moments")
        
        logger.info("Merging nearby moments...")
        audio_moments = audio_analyzer.merge_nearby_moments(
            moments,
            max_gap=2.0,
            max_duration=max_duration
        )
        logger.info(f"Merged into {len(audio_moments)} moments")
        
        # Step 2: Process Video and Create Clips
        logger.info("Step 2: Processing video and creating clips")
        video_processor = VideoProcessor(output_dir=output_dir)
        
        if not audio_moments:
            logger.warning("No engaging moments found. Creating a clip from the beginning of the video.")
            # Create a default moment from the beginning of the video
            default_moment = EngagingMoment(
                start_time=0.0,
                end_time=min(30.0, max_duration),
                text="Default clip",
                confidence=0.0,
                keywords=[],
                score=0.0
            )
            audio_moments = [default_moment]
        
        logger.info(f"Processing {len(audio_moments)} moments to create clips")
        clip_paths = video_processor.process_moments(
            input_video,
            audio_moments,
            max_clips=max_clips
        )
        
        # Print summary
        logger.info("\nClips created:")
        for i, path in enumerate(clip_paths):
            logger.info(f"  {i+1}. {path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\nAuto-Clipper test completed in {elapsed_time:.2f} seconds!")
        logger.info(f"Created {len(clip_paths)} clips")
        logger.info(f"Clips saved to: {output_dir}")
        
        return clip_paths
        
    except Exception as e:
        logger.error(f"Error in Auto-Clipper test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Check if input video is provided
    if len(sys.argv) < 2:
        print("Usage: python test_auto_clipper.py <input_video> [output_dir] [max_duration] [max_clips]")
        sys.exit(1)
    
    # Get arguments
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    max_duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    max_clips = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    # Run the test
    test_auto_clipper(input_video, output_dir, max_duration, max_clips)
