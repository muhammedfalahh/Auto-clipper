#!/usr/bin/env python3
"""
YouTube Downloader Module

This module handles downloading YouTube videos using yt-dlp and organizing them
into a structured folder system.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

# Set up logging
logger = logging.getLogger('auto_clipper.youtube_downloader')

def is_valid_youtube_url(url: str) -> bool:
    """
    Check if a URL is a valid YouTube URL.
    
    Args:
        url: URL to check
        
    Returns:
        True if valid YouTube URL, False otherwise
    """
    # YouTube URL patterns
    patterns = [
        r'^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'^(https?://)?(www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        if re.match(pattern, url):
            return True
    
    return False

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID or None if not found
    """
    # For standard YouTube URLs
    if 'youtube.com/watch' in url:
        parsed_url = urlparse(url)
        return parse_qs(parsed_url.query).get('v', [None])[0]
    
    # For shortened youtu.be URLs
    elif 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    return None

def sanitize_filename(title: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        title: String to sanitize
        
    Returns:
        Sanitized string
    """
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized

def create_video_directory(title: str) -> str:
    """
    Create a directory for a video based on its title.
    
    Args:
        title: Video title
        
    Returns:
        Path to the created directory
    """
    # Sanitize the title for use as a directory name
    dir_name = sanitize_filename(title)
    
    # Create the directory path
    video_dir = os.path.join("videos", dir_name)
    
    # Create the directory and clips subdirectory
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(os.path.join(video_dir, "clips"), exist_ok=True)
    
    logger.info(f"Created directory: {video_dir}")
    return video_dir

def download_youtube_video(url: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        url: YouTube URL
        
    Returns:
        Tuple of (video_path, video_info) or (None, None) if download fails
    """
    if not is_valid_youtube_url(url):
        logger.error(f"Invalid YouTube URL: {url}")
        return None, None
    
    try:
        # Import yt-dlp
        import yt_dlp
        
        logger.info(f"Downloading video: {url}")
        
        # First, extract info without downloading to get the title
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            
        # Create a directory based on the video title
        video_title = info.get('title', 'unknown_video')
        video_dir = create_video_directory(video_title)
        
        # Save metadata
        metadata_path = os.path.join(video_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'title': info.get('title'),
                'channel': info.get('channel'),
                'upload_date': info.get('upload_date'),
                'duration': info.get('duration'),
                'description': info.get('description'),
                'source_url': url
            }, f, indent=4)
        
        # Options for yt-dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(video_dir, 'original.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
            'progress_hooks': [lambda d: logger.info(f"Download progress: {d.get('status')} - {d.get('_percent_str', '0%')}")]
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Get the path to the downloaded video
        video_path = os.path.join(video_dir, 'original.mp4')
        if not os.path.exists(video_path):
            # Try with other extensions
            for ext in ['mkv', 'webm']:
                alt_path = os.path.join(video_dir, f'original.{ext}')
                if os.path.exists(alt_path):
                    video_path = alt_path
                    break
        
        if os.path.exists(video_path):
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path, info
        else:
            logger.error("Video download failed: File not found")
            return None, None
            
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return None, None

def download_youtube_playlist(playlist_url: str) -> list:
    """
    Download all videos in a YouTube playlist.
    
    Args:
        playlist_url: YouTube playlist URL
        
    Returns:
        List of tuples (video_path, video_info) for successfully downloaded videos
    """
    try:
        import yt_dlp
        
        logger.info(f"Processing playlist: {playlist_url}")
        
        # Extract playlist info
        with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
        if not playlist_info or 'entries' not in playlist_info:
            logger.error("Failed to extract playlist information")
            return []
            
        # Download each video
        results = []
        for entry in playlist_info['entries']:
            video_url = f"https://www.youtube.com/watch?v={entry['id']}"
            logger.info(f"Processing playlist video: {entry.get('title', 'Unknown')}")
            
            video_path, video_info = download_youtube_video(video_url)
            if video_path:
                results.append((video_path, video_info))
                
        logger.info(f"Playlist processing complete. Downloaded {len(results)} videos.")
        return results
        
    except Exception as e:
        logger.error(f"Error processing playlist: {e}")
        return []

if __name__ == "__main__":
    # Set up logging for standalone use
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Simple command-line interface for testing
    import sys
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <youtube_url>")
        sys.exit(1)
        
    url = sys.argv[1]
    if "playlist" in url:
        download_youtube_playlist(url)
    else:
        download_youtube_video(url)
