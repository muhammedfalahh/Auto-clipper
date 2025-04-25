"""
Visual Analyzer Module

This module handles visual analysis using LLaVA-7B via Ollama to detect
interesting visual moments in podcast videos.
"""

import os
import cv2
import json
import logging
import requests
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger('auto_clipper.visual_analyzer')

@dataclass
class VisualMoment:
    """Class to represent an interesting visual moment."""
    start_time: float
    end_time: float
    description: str
    score: float
    frame_path: Optional[str] = None

class VisualAnalyzer:
    """Class to analyze video frames using LLaVA-7B via Ollama."""
    
    # Visual cues that might indicate engaging content
    VISUAL_CUES = [
        "people laughing",
        "excited gestures",
        "surprised expressions",
        "animated discussion",
        "speaker change",
        "emotional reaction",
        "hand gestures",
        "visual demonstration",
        "showing an object",
        "multiple people talking"
    ]
    
    def __init__(self, 
                ollama_host: str = "http://localhost:11434",
                model_name: str = "llava:7b",
                frame_interval: int = 5,
                temp_dir: Optional[str] = None):
        """
        Initialize the VisualAnalyzer.
        
        Args:
            ollama_host: URL of the Ollama API
            model_name: Name of the LLaVA model to use
            frame_interval: Interval between frames to analyze (in seconds)
            temp_dir: Directory to store temporary files
        """
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.frame_interval = frame_interval
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def extract_frames(self, video_path: str) -> List[Tuple[float, str]]:
        """
        Extract frames from a video at regular intervals.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples containing (timestamp, frame_path)
        """
        logger.info(f"Extracting frames from video: {video_path}")
        
        # Open the video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video properties: {fps} fps, {total_frames} frames, {duration:.2f} seconds")
        
        # Calculate frame interval in frames
        frame_interval_frames = int(self.frame_interval * fps)
        
        # Create temporary directory for frames
        frames_dir = Path(self.temp_dir) / "auto_clipper_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Extract frames
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Process every Nth frame
            if frame_count % frame_interval_frames == 0:
                timestamp = frame_count / fps
                frame_path = str(frames_dir / f"frame_{frame_count:06d}_{timestamp:.2f}.jpg")
                
                # Save the frame
                cv2.imwrite(frame_path, frame)
                
                frames.append((timestamp, frame_path))
                logger.debug(f"Extracted frame at {timestamp:.2f}s: {frame_path}")
                
            frame_count += 1
            
        video.release()
        logger.info(f"Extracted {len(frames)} frames")
        
        return frames
    
    def analyze_frame(self, frame_path: str) -> Dict:
        """
        Analyze a single frame using LLaVA-7B via Ollama.
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Dictionary with analysis results
        """
        logger.debug(f"Analyzing frame: {frame_path}")
        
        # Prepare the prompt for LLaVA
        prompt = (
            "Analyze this frame from a podcast video. "
            "Describe what's happening, focusing on: "
            "1. Are people laughing or showing strong emotions? "
            "2. Is there an animated discussion happening? "
            "3. Are there interesting gestures or facial expressions? "
            "4. Is there a speaker change or multiple people talking? "
            "Respond with a JSON object with these fields: "
            "description (brief description of what's happening), "
            "engaging_moment (true/false), "
            "confidence (0-1), "
            "visual_cues (array of detected cues)"
        )
        
        try:
            # Read the image file
            with open(frame_path, "rb") as f:
                image_data = f.read()
            
            # Prepare the API request
            url = f"{self.ollama_host}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data.hex()],
                "stream": False,
                "format": "json"
            }
            
            # Make the API request
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the JSON from the response
            try:
                # The response might be a string containing JSON
                analysis = json.loads(result.get("response", "{}"))
            except json.JSONDecodeError:
                # If it's not valid JSON, try to extract JSON from the text
                text_response = result.get("response", "")
                # Look for JSON-like content between curly braces
                start = text_response.find("{")
                end = text_response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = text_response[start:end]
                    try:
                        analysis = json.loads(json_str)
                    except json.JSONDecodeError:
                        analysis = {
                            "description": text_response,
                            "engaging_moment": False,
                            "confidence": 0.0,
                            "visual_cues": []
                        }
                else:
                    analysis = {
                        "description": text_response,
                        "engaging_moment": False,
                        "confidence": 0.0,
                        "visual_cues": []
                    }
            
            logger.debug(f"Frame analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                "description": "Error analyzing frame",
                "engaging_moment": False,
                "confidence": 0.0,
                "visual_cues": []
            }
    
    def find_visual_moments(self, 
                          video_path: str, 
                          confidence_threshold: float = 0.6) -> List[VisualMoment]:
        """
        Find interesting visual moments in a video.
        
        Args:
            video_path: Path to the video file
            confidence_threshold: Minimum confidence score for moments
            
        Returns:
            List of VisualMoment objects
        """
        logger.info(f"Finding visual moments in video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Analyze each frame
        visual_moments = []
        
        for timestamp, frame_path in frames:
            # Analyze the frame
            analysis = self.analyze_frame(frame_path)
            
            # Check if it's an engaging moment
            if analysis.get("engaging_moment", False) and analysis.get("confidence", 0) >= confidence_threshold:
                # Create a visual moment
                moment = VisualMoment(
                    start_time=timestamp,
                    end_time=timestamp + self.frame_interval,  # Approximate end time
                    description=analysis.get("description", ""),
                    score=analysis.get("confidence", 0),
                    frame_path=frame_path
                )
                
                visual_moments.append(moment)
                logger.debug(f"Found visual moment at {timestamp:.2f}s: {analysis.get('description', '')}")
        
        logger.info(f"Found {len(visual_moments)} visual moments")
        return visual_moments
    
    def merge_with_audio_moments(self, 
                               visual_moments: List[VisualMoment],
                               audio_moments: List,
                               max_gap: float = 2.0) -> List:
        """
        Merge visual moments with audio moments.
        
        Args:
            visual_moments: List of visual moments
            audio_moments: List of audio moments
            max_gap: Maximum gap between moments to consider them related
            
        Returns:
            List of enhanced audio moments
        """
        logger.info("Merging visual and audio moments")
        
        # If either list is empty, return the other
        if not visual_moments:
            return audio_moments
        if not audio_moments:
            return []
        
        enhanced_moments = []
        
        # For each audio moment, check if there are overlapping visual moments
        for audio_moment in audio_moments:
            overlapping_visuals = []
            
            for visual_moment in visual_moments:
                # Check if the visual moment overlaps with the audio moment
                if (visual_moment.start_time <= audio_moment.end_time + max_gap and
                    visual_moment.end_time + max_gap >= audio_moment.start_time):
                    overlapping_visuals.append(visual_moment)
            
            # If there are overlapping visual moments, enhance the audio moment
            if overlapping_visuals:
                # Combine descriptions
                visual_descriptions = [v.description for v in overlapping_visuals]
                
                # Enhance the score based on visual confirmation
                enhanced_score = audio_moment.score + sum(v.score for v in overlapping_visuals) / len(overlapping_visuals)
                
                # Create an enhanced moment (copying the audio moment and adding visual info)
                enhanced_moment = audio_moment
                enhanced_moment.score = enhanced_score
                
                # Add to the list
                enhanced_moments.append(enhanced_moment)
            else:
                # If no visual confirmation, keep the original audio moment
                enhanced_moments.append(audio_moment)
        
        # Sort by score
        enhanced_moments.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Created {len(enhanced_moments)} enhanced moments")
        return enhanced_moments
