#!/usr/bin/env python3
"""
Viral Optimizer Module

This module enhances the scoring and selection of clips to optimize for viral potential,
focusing on motivational and life advice content.
"""

import os
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from audio_analyzer import EngagingMoment

logger = logging.getLogger('auto_clipper.viral_optimizer')

@dataclass
class ViralScore:
    """Class to represent the viral potential score of a clip."""
    moment: EngagingMoment
    emotional_score: float = 0.0
    advice_score: float = 0.0
    quote_score: float = 0.0
    story_score: float = 0.0
    total_score: float = 0.0

class ViralOptimizer:
    """Class to optimize clips for viral potential."""

    # Emotional impact indicators
    EMOTIONAL_INDICATORS = [
        "changed my life", "breakthrough", "revelation", "epiphany",
        "realization", "emotional", "powerful", "moving", "touched",
        "tears", "cry", "cried", "feel", "feeling", "felt"
    ]

    # Life advice indicators
    ADVICE_INDICATORS = [
        "advice", "lesson", "wisdom", "tip", "strategy", "technique",
        "method", "approach", "way to", "how to", "should", "must",
        "need to", "important to", "key is", "secret is", "remember"
    ]

    # Quote indicators
    QUOTE_INDICATORS = [
        '"', "'", "said", "quote", "saying", "always say", "tells me",
        "remember this", "favorite", "best advice"
    ]

    # Storytelling indicators
    STORY_INDICATORS = [
        "story", "when I was", "happened to me", "realized", "discovered",
        "learned", "found out", "experience", "journey", "path", "once"
    ]

    def __init__(self):
        """Initialize the ViralOptimizer."""
        pass

    def score_emotional_content(self, text: str) -> float:
        """
        Score the emotional impact of the text.

        Args:
            text: Text to analyze

        Returns:
            Emotional impact score (0.0-1.0)
        """
        text = text.lower()
        score = 0.0

        # Check for emotional indicators
        for indicator in self.EMOTIONAL_INDICATORS:
            if indicator in text:
                score += 0.2

        # Cap at 1.0
        return min(1.0, score)

    def score_advice_content(self, text: str) -> float:
        """
        Score the advice/instructional value of the text.

        Args:
            text: Text to analyze

        Returns:
            Advice score (0.0-1.0)
        """
        text = text.lower()
        score = 0.0

        # Check for advice indicators
        for indicator in self.ADVICE_INDICATORS:
            if indicator in text:
                score += 0.15

        # Cap at 1.0
        return min(1.0, score)

    def score_quote_potential(self, text: str) -> float:
        """
        Score the quote potential of the text.

        Args:
            text: Text to analyze

        Returns:
            Quote score (0.0-1.0)
        """
        text = text.lower()
        score = 0.0

        # Check for quote indicators
        for indicator in self.QUOTE_INDICATORS:
            if indicator in text:
                score += 0.25

        # Check for quotation marks
        if '"' in text or "'" in text:
            score += 0.5

        # Cap at 1.0
        return min(1.0, score)

    def score_story_value(self, text: str) -> float:
        """
        Score the storytelling value of the text.

        Args:
            text: Text to analyze

        Returns:
            Story score (0.0-1.0)
        """
        text = text.lower()
        score = 0.0

        # Check for storytelling indicators
        for indicator in self.STORY_INDICATORS:
            if indicator in text:
                score += 0.2

        # Cap at 1.0
        return min(1.0, score)

    def calculate_viral_score(self, moment: EngagingMoment) -> ViralScore:
        """
        Calculate the viral potential score for a moment.

        Args:
            moment: EngagingMoment to score

        Returns:
            ViralScore object with component scores
        """
        text = moment.text.lower()

        # Calculate component scores
        emotional_score = self.score_emotional_content(text)
        advice_score = self.score_advice_content(text)
        quote_score = self.score_quote_potential(text)
        story_score = self.score_story_value(text)

        # Calculate total score with weights
        # Emotional impact and advice are weighted higher for motivational content
        total_score = (
            emotional_score * 0.3 +
            advice_score * 0.3 +
            quote_score * 0.2 +
            story_score * 0.2 +
            moment.score * 0.2  # Include original score with lower weight
        )

        return ViralScore(
            moment=moment,
            emotional_score=emotional_score,
            advice_score=advice_score,
            quote_score=quote_score,
            story_score=story_score,
            total_score=total_score
        )

    def optimize_moments(self, moments: List[EngagingMoment], max_clips: int = 5) -> List[EngagingMoment]:
        """
        Optimize a list of moments for viral potential.

        Args:
            moments: List of EngagingMoment objects
            max_clips: Maximum number of clips to return

        Returns:
            List of optimized EngagingMoment objects
        """
        if not moments:
            return []

        logger.info(f"Optimizing {len(moments)} moments for viral potential")
        print("\n" + "="*80)
        print("PROGRESS: OPTIMIZING FOR VIRAL POTENTIAL")
        print("="*80)
        print(f"PROGRESS: Analyzing {len(moments)} moments to find the most viral-worthy content")
        print(f"PROGRESS: Will select up to {max_clips} clips with the highest viral potential")

        # Calculate viral scores for each moment
        print(f"PROGRESS: Calculating viral scores based on emotional impact, advice value, quotes, and storytelling...")
        viral_scores = []
        for i, moment in enumerate(moments):
            if i % 10 == 0 and len(moments) > 20:
                print(f"PROGRESS: Scored {i}/{len(moments)} moments...")
            score = self.calculate_viral_score(moment)
            viral_scores.append(score)

        # Sort by total score (highest first)
        viral_scores.sort(key=lambda x: x.total_score, reverse=True)

        # Log the top scores
        print(f"\nPROGRESS: Top {min(max_clips, len(viral_scores))} moments with highest viral potential:")
        print("-"*80)
        for i, score in enumerate(viral_scores[:max_clips]):
            logger.info(f"Viral clip {i+1}: Score={score.total_score:.2f} "
                      f"(Emotional={score.emotional_score:.2f}, "
                      f"Advice={score.advice_score:.2f}, "
                      f"Quote={score.quote_score:.2f}, "
                      f"Story={score.story_score:.2f})")

            print(f"CLIP {i+1}: Viral Score={score.total_score:.2f}")
            print(f"  - Time: {score.moment.start_time:.2f}s to {score.moment.end_time:.2f}s ({score.moment.end_time - score.moment.start_time:.2f}s)")
            print(f"  - Components: Emotional={score.emotional_score:.2f}, Advice={score.advice_score:.2f}, "
                  f"Quote={score.quote_score:.2f}, Story={score.story_score:.2f}")

            # Extract a key quote if available
            key_quote = self.extract_key_quote(score.moment.text)
            if key_quote:
                print(f"  - Key Quote: \"{key_quote}\"")

            print(f"  - Content: {score.moment.text[:100]}..." if len(score.moment.text) > 100 else f"  - Content: {score.moment.text}")
            print()

        # Return the top moments
        optimized_moments = [score.moment for score in viral_scores[:max_clips]]

        # Update the scores in the moments
        for i, moment in enumerate(optimized_moments):
            moment.score = viral_scores[i].total_score

        print(f"PROGRESS: Selected {len(optimized_moments)} moments for clip creation")
        print("="*80)

        return optimized_moments

    def extract_key_quote(self, text: str, max_length: int = 100) -> Optional[str]:
        """
        Extract a key quote from the text for overlay.

        Args:
            text: Text to extract quote from
            max_length: Maximum length of the quote

        Returns:
            Extracted quote or None if no suitable quote found
        """
        # Look for text in quotation marks
        quote_match = re.search(r'"([^"]+)"', text)
        if quote_match:
            quote = quote_match.group(1)
            if 10 <= len(quote) <= max_length:
                return quote

        # If no quoted text, look for sentences with advice indicators
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains advice indicators
            contains_advice = any(indicator in sentence.lower() for indicator in self.ADVICE_INDICATORS)

            if contains_advice and 10 <= len(sentence) <= max_length:
                return sentence

        # If no suitable sentence found, return the first sentence that's not too long
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and 10 <= len(sentence) <= max_length:
                return sentence

        # If all sentences are too long, truncate the first one
        if sentences and sentences[0].strip():
            return sentences[0].strip()[:max_length] + "..."

        return None

if __name__ == "__main__":
    # Set up logging for standalone use
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Simple test
    from audio_analyzer import EngagingMoment

    # Create some test moments
    test_moments = [
        EngagingMoment(
            start_time=10.0,
            end_time=30.0,
            text="I always tell people that the key to success is consistency. You have to show up every day.",
            confidence=0.9,
            keywords=["success", "key"],
            score=0.6
        ),
        EngagingMoment(
            start_time=60.0,
            end_time=90.0,
            text="When I was at my lowest point, I realized that failure is just a stepping stone to success. This changed my life.",
            confidence=0.8,
            keywords=["success", "failure", "changed my life"],
            score=0.7
        ),
        EngagingMoment(
            start_time=120.0,
            end_time=150.0,
            text="The best advice I ever got was \"Never give up on your dreams, no matter how impossible they seem.\"",
            confidence=0.9,
            keywords=["advice", "dreams"],
            score=0.8
        )
    ]

    # Test the optimizer
    optimizer = ViralOptimizer()
    optimized_moments = optimizer.optimize_moments(test_moments)

    # Print results
    for i, moment in enumerate(optimized_moments):
        print(f"Moment {i+1}: Score={moment.score:.2f}")
        print(f"  Text: {moment.text}")
        print(f"  Quote: {optimizer.extract_key_quote(moment.text)}")
        print()
