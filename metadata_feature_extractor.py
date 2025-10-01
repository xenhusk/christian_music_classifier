#!/usr/bin/env python3
"""
Metadata Feature Extractor - OPTIONAL Enhancement

This module extracts metadata features from audio files to supplement
audio-based features. Use this only if you want to experiment with
metadata+audio hybrid models.

Current audio-only performance: 92.2% accuracy
Expected gain with metadata: +1-3% (93-95%)
"""

import re
from pathlib import Path
from typing import Dict, Optional
import logging
from mutagen import File
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

logger = logging.getLogger(__name__)

# Christian keyword patterns (case-insensitive)
CHRISTIAN_KEYWORDS = {
    'explicit': [
        'jesus', 'christ', 'god', 'lord', 'holy spirit',
        'hallelujah', 'amen', 'blessed', 'salvation'
    ],
    'worship': [
        'worship', 'praise', 'glory', 'heaven', 'prayer',
        'grace', 'mercy', 'faith', 'believer'
    ],
    'theological': [
        'savior', 'redeemer', 'messiah', 'gospel', 'cross',
        'resurrection', 'kingdom', 'almighty', 'eternal'
    ]
}

# Known Christian artists/bands (partial list for demonstration)
KNOWN_CHRISTIAN_ARTISTS = {
    'chris tomlin', 'hillsong', 'elevation worship', 'bethel music',
    'lauren daigle', 'mercyme', 'casting crowns', 'third day',
    'newsboys', 'toby mac', 'francesca battistelli', 'for king and country',
    'crowder', 'matt redman', 'kari jobe', 'passion', 'jesus culture',
    'vertical worship', 'michael w smith', 'amy grant', 'dc talk',
    'switchfoot', 'skillet', 'needtobreathe', 'lecrae', 'kb'
}

# Christian genre tags
CHRISTIAN_GENRES = {
    'christian', 'gospel', 'worship', 'ccm', 'praise',
    'contemporary christian', 'christian rock', 'christian rap',
    'southern gospel', 'urban gospel'
}


class MetadataFeatureExtractor:
    """Extract metadata-based features from audio files."""
    
    def __init__(self):
        """Initialize metadata extractor."""
        self.all_keywords = []
        for category in CHRISTIAN_KEYWORDS.values():
            self.all_keywords.extend(category)
    
    def extract_metadata_features(self, file_path: str) -> Dict[str, float]:
        """
        Extract metadata features from an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary of metadata features (10-15 features)
        """
        features = {}
        
        try:
            # Load metadata using mutagen
            audio = File(file_path, easy=True)
            
            if audio is None:
                return self._get_default_features()
            
            # Extract basic tags
            title = self._get_tag(audio, 'title', '')
            artist = self._get_tag(audio, 'artist', '')
            album = self._get_tag(audio, 'album', '')
            genre = self._get_tag(audio, 'genre', '')
            
            # 1. Text-based features from title
            features['title_has_christian_keywords'] = float(
                self._contains_christian_keywords(title)
            )
            features['title_christian_keyword_count'] = float(
                self._count_christian_keywords(title)
            )
            features['title_word_count'] = float(len(title.split())) if title else 0.0
            
            # 2. Artist features
            features['artist_is_known_christian'] = float(
                self._is_known_christian_artist(artist)
            )
            features['artist_has_christian_keywords'] = float(
                self._contains_christian_keywords(artist)
            )
            
            # 3. Album features
            features['album_has_christian_keywords'] = float(
                self._contains_christian_keywords(album)
            )
            
            # 4. Genre features
            features['genre_is_christian'] = float(
                self._is_christian_genre(genre)
            )
            
            # 5. Combined text features
            combined_text = f"{title} {artist} {album}".lower()
            
            # Count keywords by category
            features['explicit_keyword_count'] = float(
                self._count_keywords_in_text(combined_text, CHRISTIAN_KEYWORDS['explicit'])
            )
            features['worship_keyword_count'] = float(
                self._count_keywords_in_text(combined_text, CHRISTIAN_KEYWORDS['worship'])
            )
            features['theological_keyword_count'] = float(
                self._count_keywords_in_text(combined_text, CHRISTIAN_KEYWORDS['theological'])
            )
            
            # 6. Metadata completeness (quality indicators)
            features['has_complete_metadata'] = float(
                bool(title and artist and album)
            )
            features['metadata_completeness_score'] = float(
                sum([bool(title), bool(artist), bool(album), bool(genre)]) / 4.0
            )
            
            # 7. Combined confidence score
            features['metadata_christian_confidence'] = self._calculate_confidence(
                title, artist, album, genre
            )
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting metadata from {file_path}: {e}")
            return self._get_default_features()
    
    def _get_tag(self, audio, tag_name: str, default: str = '') -> str:
        """Safely get a tag value."""
        try:
            value = audio.get(tag_name, [default])
            if isinstance(value, list) and len(value) > 0:
                return str(value[0]).strip()
            return str(value).strip() if value else default
        except:
            return default
    
    def _contains_christian_keywords(self, text: str) -> bool:
        """Check if text contains any Christian keywords."""
        if not text:
            return False
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.all_keywords)
    
    def _count_christian_keywords(self, text: str) -> int:
        """Count number of Christian keywords in text."""
        if not text:
            return 0
        
        text_lower = text.lower()
        return sum(1 for keyword in self.all_keywords if keyword in text_lower)
    
    def _count_keywords_in_text(self, text: str, keywords: list) -> int:
        """Count specific keywords in text."""
        if not text:
            return 0
        return sum(1 for keyword in keywords if keyword in text)
    
    def _is_known_christian_artist(self, artist: str) -> bool:
        """Check if artist is in known Christian artists list."""
        if not artist:
            return False
        
        artist_lower = artist.lower().strip()
        
        # Exact match
        if artist_lower in KNOWN_CHRISTIAN_ARTISTS:
            return True
        
        # Partial match (for featuring artists, etc.)
        return any(known in artist_lower for known in KNOWN_CHRISTIAN_ARTISTS)
    
    def _is_christian_genre(self, genre: str) -> bool:
        """Check if genre indicates Christian music."""
        if not genre:
            return False
        
        genre_lower = genre.lower().strip()
        return any(christian_genre in genre_lower for christian_genre in CHRISTIAN_GENRES)
    
    def _calculate_confidence(self, title: str, artist: str, album: str, genre: str) -> float:
        """
        Calculate overall confidence that this is Christian music based on metadata.
        
        Returns: Float between 0.0 and 1.0
        """
        score = 0.0
        
        # Genre is strongest signal (if present)
        if self._is_christian_genre(genre):
            score += 0.4
        
        # Known Christian artist
        if self._is_known_christian_artist(artist):
            score += 0.3
        
        # Keywords in title
        if self._contains_christian_keywords(title):
            score += 0.2
        
        # Keywords in album
        if self._contains_christian_keywords(album):
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when metadata is unavailable."""
        return {
            'title_has_christian_keywords': 0.0,
            'title_christian_keyword_count': 0.0,
            'title_word_count': 0.0,
            'artist_is_known_christian': 0.0,
            'artist_has_christian_keywords': 0.0,
            'album_has_christian_keywords': 0.0,
            'genre_is_christian': 0.0,
            'explicit_keyword_count': 0.0,
            'worship_keyword_count': 0.0,
            'theological_keyword_count': 0.0,
            'has_complete_metadata': 0.0,
            'metadata_completeness_score': 0.0,
            'metadata_christian_confidence': 0.0
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python metadata_feature_extractor.py <audio_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    extractor = MetadataFeatureExtractor()
    features = extractor.extract_metadata_features(file_path)
    
    print(f"\n📋 Metadata Features for: {Path(file_path).name}")
    print("=" * 60)
    
    for feature_name, value in features.items():
        print(f"   {feature_name:<35} : {value:.3f}")
    
    print(f"\n💡 Christian Confidence Score: {features['metadata_christian_confidence']:.1%}")
    
    if features['has_complete_metadata'] > 0:
        print("✅ Has complete metadata")
    else:
        print("⚠️  Missing some metadata - audio features will be primary")

