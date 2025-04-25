"""
Audio Analyzer Module

This module handles audio analysis using Whisper to transcribe speech and
identify engaging moments in podcast conversations.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger('auto_clipper.audio_analyzer')

@dataclass
class EngagingMoment:
    """Class to represent an engaging moment in the audio."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    keywords: List[str] = None
    score: float = 0.0

class AudioAnalyzer:
    """Class to analyze audio using Whisper for transcription and engagement detection."""

    # Keywords that might indicate engaging content
    ENGAGEMENT_KEYWORDS = [
        # Laughter and emotions
        "laugh", "funny", "hilarious", "joke", "haha",
        # Interesting facts
        "fact", "interesting", "amazing", "wow", "incredible", "fascinating",
        # Mindset and insights
        "mindset", "perspective", "insight", "philosophy", "wisdom", "advice",
        # Surprising information
        "surprising", "unexpected", "shocking", "unbelievable", "mind-blowing",
        # Strong opinions
        "absolutely", "definitely", "strongly", "believe", "passionate",
        # Life advice specific
        "life lesson", "taught me", "learned that", "realize", "understand",
        "truth", "reality", "important to know", "remember that", "never forget",
        # Wisdom indicators
        "wise", "smart", "intelligent", "clever", "brilliant", "genius",
        "knowledge", "understanding", "enlightenment", "awareness"
    ]

    # Keywords specifically for motivational content
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
        "life lesson", "best advice", "important lesson", "key lesson",
        "what I learned", "taught me", "changed how I", "secret to",

        # Wisdom and knowledge
        "wisdom", "knowledge", "understand", "realize", "enlighten",
        "truth", "reality", "fact of life", "life truth", "universal truth",
        "principle", "philosophy", "way of thinking", "mindset shift",

        # Emotional impact
        "changed my life", "breakthrough", "revelation", "epiphany", "realization",
        "transformative", "life-changing", "profound", "powerful", "impactful",

        # Storytelling markers
        "story", "when I was", "happened to me", "realized", "discovered",
        "learned that", "found out", "came to understand", "my journey",

        # Life guidance
        "should never", "always remember", "never forget", "key is to",
        "secret is", "most important thing", "critical to", "essential to",
        "focus on", "prioritize", "remember that", "don't waste", "invest in"
    ]

    def __init__(self, model_size: str = "small.en", device: str = "cuda"):
        """
        Initialize the AudioAnalyzer.

        Args:
            model_size: Size of the Whisper model to use
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_size = model_size
        self.device = device
        self.model = None

    def load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self.model = WhisperModel(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Transcribe the audio from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            List of segment dictionaries with start_time, end_time, text, etc.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio from: {video_path}")
        print(f"\n[PROGRESS] Starting transcription of video using Whisper {self.model_size}...")

        try:
            print(f"[PROGRESS] Running Whisper on video file (this may take several minutes)...")
            segments, info = self.model.transcribe(
                video_path,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            print(f"[PROGRESS] Whisper processing complete, organizing transcription segments...")

            # Convert generator to list for easier processing
            segment_list = []
            segment_count = 0
            total_duration = 0

            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'words': [{'word': word.word, 'start': word.start, 'end': word.end, 'probability': word.probability}
                             for word in segment.words] if segment.words else [],
                    'confidence': segment.avg_logprob
                }
                segment_list.append(segment_dict)
                segment_count += 1
                duration = segment.end - segment.start
                total_duration += duration

                # Print progress every 10 segments
                if segment_count % 10 == 0:
                    print(f"[PROGRESS] Processed {segment_count} segments ({total_duration:.2f} seconds of content)")

            logger.info(f"Transcription completed: {len(segment_list)} segments found")
            print(f"[PROGRESS] Transcription complete! Found {len(segment_list)} segments covering {total_duration:.2f} seconds")
            return segment_list

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            print(f"[ERROR] Transcription failed: {e}")
            raise

    def find_engaging_moments(self, segments: List[Dict],
                             min_duration: float = 5.0,
                             max_duration: float = 60.0,
                             min_confidence: float = -1.0,
                             motivational_only: bool = False) -> List[EngagingMoment]:
        """
        Find engaging moments in the transcribed segments.

        Args:
            segments: List of transcribed segments
            min_duration: Minimum duration of an engaging moment in seconds
            max_duration: Maximum duration of an engaging moment in seconds
            min_confidence: Minimum confidence score for segments to consider
            motivational_only: If True, only return segments with motivational content

        Returns:
            List of EngagingMoment objects
        """
        logger.info("Finding engaging moments in transcription")
        print(f"\n[PROGRESS] Analyzing {len(segments)} segments for engaging/motivational content...")
        if motivational_only:
            print(f"[PROGRESS] Looking for motivational content only")

        engaging_moments = []
        motivational_count = 0
        general_count = 0
        processed_count = 0
        total_segments = len(segments)

        # Process each segment
        for segment in segments:
            processed_count += 1

            # Print progress every 20 segments
            if processed_count % 20 == 0 or processed_count == total_segments:
                print(f"[PROGRESS] Analyzed {processed_count}/{total_segments} segments " +
                      f"({(processed_count/total_segments)*100:.1f}%) - " +
                      f"Found {len(engaging_moments)} engaging moments so far")

            # Skip segments with low confidence
            if segment['confidence'] < min_confidence:
                continue

            # Skip segments that are too short
            duration = segment['end'] - segment['start']
            if duration < min_duration:
                continue

            text = segment['text'].lower()

            # Check for engagement keywords
            found_keywords = []
            for keyword in self.ENGAGEMENT_KEYWORDS:
                if keyword in text:
                    found_keywords.append(keyword)

            # Check for motivational keywords
            found_motivational_keywords = []
            for keyword in self.MOTIVATIONAL_KEYWORDS:
                if keyword in text:
                    found_motivational_keywords.append(keyword)

            # Combine all found keywords
            all_keywords = found_keywords + found_motivational_keywords

            # Skip if we're looking for motivational content only and none was found
            if motivational_only and not found_motivational_keywords:
                continue

            # If we found keywords, this might be engaging
            if all_keywords:
                # Calculate engagement score
                # Give higher weight to motivational keywords (0.3 vs 0.2)
                score = len(found_keywords) * 0.2 + len(found_motivational_keywords) * 0.3

                # Create engaging moment
                moment = EngagingMoment(
                    start_time=segment['start'],
                    end_time=min(segment['start'] + max_duration, segment['end']),
                    text=segment['text'],
                    confidence=segment['confidence'],
                    keywords=all_keywords,
                    score=score
                )

                engaging_moments.append(moment)
                if found_motivational_keywords:
                    motivational_count += 1
                    # Print details for motivational moments
                    print(f"[FOUND] Motivational moment at {moment.start_time:.2f}s: " +
                          f"Score: {score:.2f}, Keywords: {', '.join(found_motivational_keywords[:3])}")
                    print(f"       Text: {moment.text[:100]}..." if len(moment.text) > 100 else f"       Text: {moment.text}")
                    logger.debug(f"Found motivational moment: {moment}")
                else:
                    general_count += 1
                    logger.debug(f"Found engaging moment: {moment}")

        # Sort by score (highest first)
        engaging_moments.sort(key=lambda x: x.score, reverse=True)

        print(f"\n[PROGRESS] Analysis complete! Found {len(engaging_moments)} engaging moments:")
        print(f"[PROGRESS] - {motivational_count} motivational moments")
        print(f"[PROGRESS] - {general_count} general engaging moments")

        logger.info(f"Found {len(engaging_moments)} engaging moments")
        return engaging_moments

    def merge_nearby_moments(self, moments: List[EngagingMoment],
                           max_gap: float = 2.0,
                           max_duration: float = 60.0) -> List[EngagingMoment]:
        """
        Merge nearby engaging moments to create longer, coherent clips.

        Args:
            moments: List of engaging moments
            max_gap: Maximum gap between moments to merge (in seconds)
            max_duration: Maximum duration of merged moments

        Returns:
            List of merged engaging moments
        """
        if not moments:
            return []

        # Sort by start time
        sorted_moments = sorted(moments, key=lambda x: x.start_time)

        merged_moments = []
        current_moment = sorted_moments[0]

        for next_moment in sorted_moments[1:]:
            # If the next moment starts soon after the current one ends
            if next_moment.start_time - current_moment.end_time <= max_gap:
                # Check if merging would exceed max duration
                merged_duration = next_moment.end_time - current_moment.start_time
                if merged_duration <= max_duration:
                    # Merge the moments
                    current_moment.end_time = next_moment.end_time
                    current_moment.text += " " + next_moment.text
                    current_moment.keywords = list(set(current_moment.keywords + next_moment.keywords))
                    current_moment.score = max(current_moment.score, next_moment.score)
                else:
                    # If merging would exceed max duration, add current and start a new one
                    merged_moments.append(current_moment)
                    current_moment = next_moment
            else:
                # If gap is too large, add current and start a new one
                merged_moments.append(current_moment)
                current_moment = next_moment

        # Add the last moment
        merged_moments.append(current_moment)

        logger.info(f"Merged into {len(merged_moments)} moments")
        return merged_moments

    def analyze(self, video_path: str,
               min_duration: float = 5.0,
               max_duration: float = 60.0,
               max_gap: float = 2.0,
               motivational_only: bool = False) -> List[EngagingMoment]:
        """
        Analyze a video file to find engaging moments.

        Args:
            video_path: Path to the video file
            min_duration: Minimum duration of an engaging moment
            max_duration: Maximum duration of an engaging moment
            max_gap: Maximum gap between moments to merge
            motivational_only: If True, only return segments with motivational content

        Returns:
            List of engaging moments
        """
        # Transcribe the video
        segments = self.transcribe(video_path)

        # Find engaging moments
        moments = self.find_engaging_moments(
            segments,
            min_duration=min_duration,
            max_duration=max_duration,
            motivational_only=motivational_only
        )

        # Merge nearby moments
        merged_moments = self.merge_nearby_moments(
            moments,
            max_gap=max_gap,
            max_duration=max_duration
        )

        return merged_moments
