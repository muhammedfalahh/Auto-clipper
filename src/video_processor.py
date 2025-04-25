"""
Video Processor Module

This module handles face detection and video processing to create
vertical clips from horizontal podcast videos.
"""

import os
import cv2
import logging
import subprocess
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger('auto_clipper.video_processor')

@dataclass
class FaceTrackingResult:
    """Class to represent face tracking results for a clip."""
    start_time: float
    end_time: float
    crop_x: int  # X-coordinate for the left edge of the crop window
    crop_width: int
    crop_height: int
    confidence: float

class VideoProcessor:
    """Class to process videos, detect faces, and create vertical clips."""

    def __init__(self,
                output_dir: str = "output",
                face_confidence: float = 0.5,
                sample_rate: int = 5):
        """
        Initialize the VideoProcessor.

        Args:
            output_dir: Directory to save output clips
            face_confidence: Minimum confidence for face detection
            sample_rate: Number of frames to sample per second for face tracking
        """
        self.output_dir = Path(output_dir)
        self.face_confidence = face_confidence
        self.sample_rate = sample_rate

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_faces(self, frame) -> List[Dict]:
        """
        Detect faces in a frame.

        Args:
            frame: OpenCV image frame

        Returns:
            List of face detection results
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        with self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range detection
            min_detection_confidence=self.face_confidence
        ) as face_detection:
            results = face_detection.process(rgb_frame)

            # Return empty list if no faces detected
            if not results.detections:
                return []

            # Extract face information
            faces = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                faces.append({
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'confidence': detection.score[0]
                })

            return faces

    def track_faces(self, video_path: str, start_time: float, end_time: float) -> FaceTrackingResult:
        """
        Track faces in a video segment to determine the optimal crop window.

        Args:
            video_path: Path to the video file
            start_time: Start time of the segment in seconds
            end_time: End time of the segment in seconds

        Returns:
            FaceTrackingResult with optimal crop parameters
        """
        logger.info(f"Tracking faces from {start_time:.2f}s to {end_time:.2f}s")
        print(f"PROGRESS: Analyzing video segment from {start_time:.2f}s to {end_time:.2f}s for face tracking")

        # Open the video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"PROGRESS: Video properties: {width}x{height} at {fps:.2f} fps")

        # Calculate frame positions
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame

        # Calculate sampling interval
        sample_interval = max(1, int(fps / self.sample_rate))
        print(f"PROGRESS: Sampling {self.sample_rate} frames per second (every {sample_interval} frames)")
        print(f"PROGRESS: Will analyze approximately {total_frames // sample_interval} frames")

        # Seek to start position
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Lists to store face positions
        face_positions = []

        # Process frames
        frame_count = 0
        faces_found = 0
        print(f"PROGRESS: Scanning frames for faces...")

        while frame_count < total_frames:
            ret, frame = video.read()
            if not ret:
                break

            # Process every Nth frame
            if frame_count % sample_interval == 0:
                # Detect faces
                faces = self.detect_faces(frame)

                # Store face positions
                for face in faces:
                    face_positions.append((
                        face['x'] + face['width'] // 2,  # Center X
                        face['y'] + face['height'] // 2,  # Center Y
                        face['width'],
                        face['height'],
                        face['confidence']
                    ))
                    faces_found += 1

                # Print progress every 10 processed frames
                if (frame_count // sample_interval) % 10 == 0:
                    current_time = start_time + (frame_count / fps)
                    print(f"PROGRESS: Processed frame at {current_time:.2f}s - Found {faces_found} faces so far")

            frame_count += 1

        video.release()
        print(f"PROGRESS: Face detection complete. Analyzed {frame_count} frames, found {faces_found} face instances")

        # If no faces detected, use center of frame
        if not face_positions:
            logger.warning("No faces detected, using center of frame")
            print(f"PROGRESS: No faces detected, using center of frame for vertical crop")
            return FaceTrackingResult(
                start_time=start_time,
                end_time=end_time,
                crop_x=width // 2 - (height * 9 // 16) // 2,  # Center the 9:16 crop
                crop_width=height * 9 // 16,  # 9:16 aspect ratio based on height
                crop_height=height,
                confidence=0.0
            )

        # Calculate average face position (weighted by confidence)
        total_weight = sum(conf for _, _, _, _, conf in face_positions)
        avg_x = sum(x * conf for x, _, _, _, conf in face_positions) / total_weight

        # Calculate optimal crop window (9:16 aspect ratio)
        crop_height = height
        crop_width = height * 9 // 16  # 9:16 aspect ratio

        # Center the crop window on the average face position
        crop_x = int(avg_x - crop_width // 2)

        # Ensure crop window is within frame bounds
        crop_x = max(0, min(width - crop_width, crop_x))

        logger.info(f"Optimal crop window: x={crop_x}, width={crop_width}, height={crop_height}")
        print(f"PROGRESS: Determined optimal crop window: x={crop_x}, width={crop_width}, height={crop_height}")

        return FaceTrackingResult(
            start_time=start_time,
            end_time=end_time,
            crop_x=crop_x,
            crop_width=crop_width,
            crop_height=crop_height,
            confidence=total_weight / len(face_positions)
        )

    def create_clip(self,
                  video_path: str,
                  output_path: str,
                  start_time: float,
                  end_time: float,
                  tracking_result: Optional[FaceTrackingResult] = None) -> str:
        """
        Create a vertical clip from a horizontal video.

        Args:
            video_path: Path to the input video
            output_path: Path to save the output clip
            start_time: Start time of the clip in seconds
            end_time: End time of the clip in seconds
            tracking_result: Face tracking result for optimal cropping

        Returns:
            Path to the created clip
        """
        logger.info(f"Creating clip from {start_time:.2f}s to {end_time:.2f}s")
        print(f"PROGRESS: Creating vertical clip from {start_time:.2f}s to {end_time:.2f}s")

        # If no tracking result provided, track faces
        if tracking_result is None:
            print(f"PROGRESS: No tracking result provided, tracking faces first...")
            tracking_result = self.track_faces(video_path, start_time, end_time)

        # Prepare FFmpeg command
        crop_filter = f"crop={tracking_result.crop_width}:{tracking_result.crop_height}:{tracking_result.crop_x}:0"
        print(f"PROGRESS: Using crop filter: {crop_filter}")

        # Build the FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ]

        # Execute the command
        try:
            print(f"PROGRESS: Running FFmpeg to create clip...")
            print(f"PROGRESS: Output will be saved to: {output_path}")
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                print(f"ERROR: FFmpeg failed with error: {result.stderr[:200]}...")
                raise RuntimeError(f"FFmpeg failed with error: {result.stderr}")

            logger.info(f"Clip created successfully: {output_path}")
            print(f"PROGRESS: Clip created successfully!")
            return output_path

        except Exception as e:
            logger.error(f"Error creating clip: {e}")
            print(f"ERROR: Failed to create clip: {e}")
            raise

    def process_moments(self,
                      video_path: str,
                      moments: List,
                      max_clips: int = 10) -> List[str]:
        """
        Process a list of moments to create clips.

        Args:
            video_path: Path to the input video
            moments: List of moments to process
            max_clips: Maximum number of clips to create

        Returns:
            List of paths to created clips
        """
        logger.info(f"Processing {len(moments)} moments (max {max_clips} clips)")
        print("\n" + "="*80)
        print("PROGRESS: CREATING VIRAL CLIPS")
        print("="*80)
        print(f"PROGRESS: Processing {len(moments)} moments (max {max_clips} clips)")

        # Sort moments by score (highest first)
        sorted_moments = sorted(moments, key=lambda x: x.score, reverse=True)

        # Limit to max_clips
        moments_to_process = sorted_moments[:max_clips]
        print(f"PROGRESS: Selected top {len(moments_to_process)} moments for clip creation")

        # Create clips
        clip_paths = []

        for i, moment in enumerate(moments_to_process):
            # Generate output path
            output_filename = f"clip_{i:03d}_{moment.start_time:.2f}_{moment.end_time:.2f}.mp4"
            output_path = str(self.output_dir / output_filename)

            print(f"\nPROGRESS: Creating clip {i+1}/{len(moments_to_process)}")
            print(f"PROGRESS: Time range: {moment.start_time:.2f}s to {moment.end_time:.2f}s ({moment.end_time - moment.start_time:.2f}s duration)")
            print(f"PROGRESS: Viral score: {moment.score:.2f}")
            print(f"PROGRESS: Content: {moment.text[:100]}..." if len(moment.text) > 100 else f"PROGRESS: Content: {moment.text}")

            # Track faces
            print(f"PROGRESS: Detecting and tracking faces...")
            tracking_result = self.track_faces(
                video_path,
                moment.start_time,
                moment.end_time
            )

            # Create the clip
            try:
                print(f"PROGRESS: Creating clip file: {output_filename}")
                clip_path = self.create_clip(
                    video_path,
                    output_path,
                    moment.start_time,
                    moment.end_time,
                    tracking_result
                )

                clip_paths.append(clip_path)
                print(f"PROGRESS: Successfully created clip: {output_filename}")

            except Exception as e:
                logger.error(f"Error creating clip {i}: {e}")
                print(f"ERROR: Failed to create clip {i+1}: {e}")
                continue

        print("\n" + "="*80)
        print(f"PROGRESS: CLIP CREATION COMPLETE! Created {len(clip_paths)} clips")
        print(f"PROGRESS: Clips saved to: {self.output_dir}")
        print("="*80)

        logger.info(f"Created {len(clip_paths)} clips")
        return clip_paths
