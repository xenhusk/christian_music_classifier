#!/usr/bin/env python3
"""
Improved Audio-Based Christian Music Classifier

This script fixes all the bias issues found in the original classifier:
1. Fixes label encoding problems
2. Removes constant/low-variance features
3. Addresses data imbalance with proper class balancing
4. Improves feature engineering for better discrimination
5. Fixes decision boundary bias
6. Includes comprehensive validation and evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import time

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedAudioFeatureExtractor:
    """Improved audio feature extractor with better feature engineering."""
    
    def __init__(self, sample_rate: int = 22050, duration: int = 10, max_workers: int = None):
        """
        Initialize improved audio feature extractor.
        
        Args:
            sample_rate: Sample rate for audio processing
            duration: Duration in seconds to analyze
            max_workers: Number of parallel workers
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.max_workers = max_workers or mp.cpu_count()
        
    def extract_features_single(self, file_path: str) -> Optional[Dict[str, float]]:
        """
        Extract enhanced features from a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary of extracted features or None if failed
        """
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            if len(y) == 0:
                return None
            
            features = {}
            
            # Basic properties (normalized/relative features instead of absolute)
            features['signal_length_ratio'] = float(len(y) / (self.sample_rate * self.duration))
            features['rms_energy_ratio'] = float(np.sqrt(np.mean(y**2)) / (np.max(np.abs(y)) + 1e-8))
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            features['spectral_centroid_skew'] = float(self._skewness(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # 2. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 3. MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # 4. Chroma features (key-related)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            # Individual chroma bins (12 semitones)
            chroma_bins = np.mean(chroma, axis=1)
            for i in range(12):
                features[f'chroma_bin_{i}'] = float(chroma_bins[i])
            
            # 5. Tonnetz features (harmonic network)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = float(np.mean(tonnetz))
            features['tonnetz_std'] = float(np.std(tonnetz))
            
            # 6. Rhythm and tempo features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if np.isfinite(tempo) else 120.0
            features['beat_strength'] = float(len(beats) / (len(y) / sr)) if len(y) > 0 else 0.0
            
            # 7. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            features['spectral_contrast_std'] = float(np.std(contrast))
            
            # 8. Spectral flatness (measure of noisiness)
            flatness = librosa.feature.spectral_flatness(y=y)
            features['spectral_flatness_mean'] = float(np.mean(flatness))
            features['spectral_flatness_std'] = float(np.std(flatness))
            
            # 9. Dynamic features
            features['dynamic_range'] = float(np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5))
            features['peak_to_rms_ratio'] = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-8))
            
            # 10. Harmonic-percussive separation features
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                total_energy = harmonic_energy + percussive_energy
                
                features['harmonic_ratio'] = float(harmonic_energy / (total_energy + 1e-8))
                features['percussive_ratio'] = float(percussive_energy / (total_energy + 1e-8))
            except:
                features['harmonic_ratio'] = 0.5
                features['percussive_ratio'] = 0.5
            
            # 11. Additional spectral features
            features['spectral_centroid_normalized'] = float(np.mean(spectral_centroids) / (sr / 2))
            
            # 12. Zero-padding and windowing artifacts detection
            features['silence_ratio'] = float(np.sum(np.abs(y) < 0.01) / len(y))
            
            return features
            
        except Exception as e:
            logger.debug(f"Error extracting features from {file_path}: {e}")
            return None
    
    def _skewness(self, data):
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def extract_features_parallel(self, file_paths: List[str]) -> Tuple[List[Dict[str, float]], List[str]]:
        """Extract features from multiple files in parallel."""
        logger.info(f"🚀 Extracting improved features from {len(file_paths)} files using {self.max_workers} workers...")
        
        features_list = []
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.extract_features_single, path): path 
                for path in file_paths
            }
            
            for future in tqdm(as_completed(future_to_path), total=len(file_paths), desc="Processing audio files"):
                path = future_to_path[future]
                try:
                    features = future.result()
                    if features is not None:
                        features_list.append(features)
                    else:
                        failed_files.append(path)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    failed_files.append(path)
        
        logger.info(f"✅ Successfully processed {len(features_list)} files, {len(failed_files)} failed")
        return features_list, failed_files

class ImprovedAudioDataLoader:
    """Improved audio data loader with better label handling."""
    
    def __init__(self, data_path: str = "TrainingData"):
        self.data_path = Path(data_path)
        self.christian_path = self.data_path / "ChristianMusic"
        self.secular_path = self.data_path / "SecularMusic"
        
    def load_audio_files(self) -> List[Dict[str, Any]]:
        """Load all audio files with their labels."""
        audio_files = []
        
        # Load Christian music files
        if self.christian_path.exists():
            for file_path in self.christian_path.rglob("*"):
                if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.opus', '.flac']:
                    audio_files.append({
                        'filepath': str(file_path),
                        'filename': file_path.name,
                        'label': 'Christian',  # String label
                        'label_numeric': 0,    # Numeric label: Christian = 0
                        'category': 'Christian'
                    })
        
        # Load Secular music files
        if self.secular_path.exists():
            for file_path in self.secular_path.rglob("*"):
                if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.opus', '.flac']:
                    audio_files.append({
                        'filepath': str(file_path),
                        'filename': file_path.name,
                        'label': 'Secular',    # String label
                        'label_numeric': 1,    # Numeric label: Secular = 1
                        'category': 'Secular'
                    })
        
        return audio_files

class ImprovedAudioClassifier:
    """Improved audio classifier with bias fixes."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.variance_selector = None
        self.feature_selector = None
        self.feature_names = None
        self.selected_feature_names = None
        self.label_map = {0: 'Christian', 1: 'Secular'}
        self.class_weights = None
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train the improved classifier."""
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names
        
        # 1. Remove constant and low-variance features
        logger.info("🔧 Removing constant and low-variance features...")
        self.variance_selector = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
        X_variance_filtered = self.variance_selector.fit_transform(X)
        
        # Get remaining feature names after variance filtering
        variance_mask = self.variance_selector.get_support()
        remaining_features = [feature_names[i] for i, mask in enumerate(variance_mask) if mask]
        
        logger.info(f"   Removed {len(feature_names) - len(remaining_features)} low-variance features")
        logger.info(f"   Remaining features: {len(remaining_features)}")
        
        # 2. Scale features
        X_scaled = self.scaler.fit_transform(X_variance_filtered)
        
        # 3. Feature selection - select top k features
        k_best = min(30, X_scaled.shape[1])  # Select top 30 features or all if fewer
        logger.info(f"🎯 Selecting top {k_best} most discriminative features...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selection_mask = self.feature_selector.get_support()
        self.selected_feature_names = [remaining_features[i] for i, mask in enumerate(selection_mask) if mask]
        
        logger.info(f"   Selected {len(self.selected_feature_names)} features")
        logger.info(f"   Top features: {self.selected_feature_names[:5]}")
        
        # 4. Calculate class weights to handle imbalance
        unique_classes = np.unique(y)
        self.class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, self.class_weights))
        
        logger.info(f"📊 Class distribution: {np.bincount(y)}")
        logger.info(f"⚖️ Class weights: {class_weight_dict}")
        
        # 5. Initialize model with class balancing
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle class imbalance
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',  # Handle class imbalance
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 6. Train model
        logger.info(f"🤖 Training balanced {self.model_type}...")
        self.model.fit(X_selected, y)
        
        # 7. Validate with cross-validation
        logger.info("🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_selected, y, cv=5, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        return {
            'train_accuracy': self.model.score(X_selected, y),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'n_features_original': X.shape[1],
            'n_features_selected': X_selected.shape[1],
            'n_samples': X.shape[0],
            'training_time': training_time,
            'class_distribution': np.bincount(y).tolist(),
            'class_weights': class_weight_dict,
            'selected_features': self.selected_feature_names
        }
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Apply the same preprocessing pipeline used during training."""
        if hasattr(self.scaler, 'mean_'):
            # Apply variance threshold first
            X_variance_filtered = self.variance_selector.transform(X)
            # Scale features
            X_scaled = self.scaler.transform(X_variance_filtered)
            # Apply feature selection
            X_selected = self.feature_selector.transform(X_scaled)
            return X_selected
        else:
            raise ValueError("Model not trained yet!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_processed = self._prepare_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_processed = self._prepare_features(X)
        return self.model.predict_proba(X_processed)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'variance_selector': self.variance_selector,
            'feature_selector': self.feature_selector,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'selected_feature_names': self.selected_feature_names,
            'label_map': self.label_map,
            'class_weights': self.class_weights
        }
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Improved model saved to {filepath}")

def extract_all_features_improved(audio_files: List[Dict[str, Any]], 
                                feature_extractor: ImprovedAudioFeatureExtractor) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract features from all audio files using improved processing."""
    
    # Prepare file paths and labels
    file_paths = [file_info['filepath'] for file_info in audio_files]
    
    # Extract features in parallel
    features_list, failed_files = feature_extractor.extract_features_parallel(file_paths)
    
    if not features_list:
        raise ValueError("No features extracted from any files!")
    
    # Convert to numpy arrays
    feature_names = list(features_list[0].keys())
    X = np.array([[features[name] for name in feature_names] for features in features_list])
    
    # Filter labels to match successful extractions
    successful_labels = []
    failed_set = set(failed_files)
    
    for file_info in audio_files:
        if file_info['filepath'] not in failed_set:
            successful_labels.append(file_info['label_numeric'])  # Use numeric labels
    
    y = np.array(successful_labels, dtype=int)
    
    logger.info(f"📊 Feature extraction complete:")
    logger.info(f"   Successful: {len(features_list)}")
    logger.info(f"   Failed: {len(failed_files)}")
    logger.info(f"   Feature dimensions: {X.shape[1]}")
    logger.info(f"   Label distribution: Christian={np.sum(y==0)}, Secular={np.sum(y==1)}")
    
    return X, y, feature_names

def create_improved_visualizations(y_true: np.ndarray, y_pred: np.ndarray, 
                                 feature_importance: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None,
                                 class_names: List[str] = ['Christian', 'Secular']):
    """Create improved visualization plots."""
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create combined labels showing counts and percentages
    labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Improved Audio Classifier - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('visualizations/improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    if feature_importance is not None and feature_names is not None:
        plt.figure(figsize=(12, 10))
        top_indices = np.argsort(feature_importance)[-25:]  # Top 25 features
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_importance, color=colors)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top 25 Most Important Audio Features (Improved Model)')
        plt.tight_layout()
        plt.savefig('visualizations/improved_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("📊 Improved visualizations saved to 'visualizations/' directory")

def main():
    """Main improved training function."""
    print("🎯 Improved Audio-Based Christian Music Classifier")
    print("=" * 60)
    
    # System info
    print(f"💻 CPU Cores: {mp.cpu_count()}")
    
    # Initialize improved components
    data_loader = ImprovedAudioDataLoader()
    feature_extractor = ImprovedAudioFeatureExtractor(max_workers=mp.cpu_count())
    
    # Load audio files
    print("\n📁 Loading audio files...")
    audio_files = data_loader.load_audio_files()
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    print(f"✅ Found {len(audio_files)} audio files")
    
    # Show distribution
    christian_count = sum(1 for f in audio_files if f['label'] == 'Christian')
    secular_count = sum(1 for f in audio_files if f['label'] == 'Secular')
    imbalance_ratio = max(christian_count, secular_count) / min(christian_count, secular_count)
    
    print(f"   Christian: {christian_count} ({christian_count/len(audio_files)*100:.1f}%)")
    print(f"   Secular: {secular_count} ({secular_count/len(audio_files)*100:.1f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Extract features with improved processing
    print("\n🔍 Extracting improved audio features...")
    start_time = time.time()
    X, y, feature_names = extract_all_features_improved(audio_files, feature_extractor)
    extraction_time = time.time() - start_time
    
    print(f"⏱️ Feature extraction completed in {extraction_time:.1f} seconds")
    print(f"🚀 Processing speed: {len(audio_files)/extraction_time:.1f} files/second")
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split (stratified):")
    print(f"   Training: {len(X_train)} samples (Christian: {np.sum(y_train==0)}, Secular: {np.sum(y_train==1)})")
    print(f"   Testing: {len(X_test)} samples (Christian: {np.sum(y_test==0)}, Secular: {np.sum(y_test==1)})")
    
    # Train improved models
    models = {}
    for model_type in ['random_forest', 'svm']:
        print(f"\n🤖 Training improved {model_type}...")
        
        classifier = ImprovedAudioClassifier(model_type=model_type)
        
        # Train
        train_metrics = classifier.train(X_train, y_train, feature_names)
        
        # Test
        y_pred = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        models[model_type] = {
            'classifier': classifier,
            'train_accuracy': train_metrics['train_accuracy'],
            'cv_mean': train_metrics['cv_mean'],
            'cv_std': train_metrics['cv_std'],
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'training_time': train_metrics['training_time'],
            'n_features_selected': train_metrics['n_features_selected'],
            'class_weights': train_metrics['class_weights']
        }
        
        print(f"   Training accuracy: {train_metrics['train_accuracy']:.3f}")
        print(f"   CV accuracy: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}")
        print(f"   Test accuracy: {test_accuracy:.3f}")
        print(f"   Features selected: {train_metrics['n_features_selected']}")
        print(f"   Training time: {train_metrics['training_time']:.1f}s")
    
    # Choose best model based on CV score
    best_model_type = max(models.keys(), key=lambda k: models[k]['cv_mean'])
    best_model = models[best_model_type]['classifier']
    best_accuracy = models[best_model_type]['test_accuracy']
    
    print(f"\n🏆 Best model: {best_model_type}")
    print(f"   Test accuracy: {best_accuracy:.3f}")
    print(f"   CV accuracy: {models[best_model_type]['cv_mean']:.3f}")
    
    # Detailed evaluation
    y_pred_best = models[best_model_type]['predictions']
    class_names = ['Christian', 'Secular']
    
    print(f"\n📈 Detailed Results:")
    print(classification_report(y_test, y_pred_best, target_names=class_names))
    
    # Feature importance
    feature_importance = None
    if hasattr(best_model.model, 'feature_importances_'):
        feature_importance = best_model.model.feature_importances_
    
    # Create improved visualizations
    print("\n📊 Creating improved visualizations...")
    create_improved_visualizations(y_test, y_pred_best, feature_importance, 
                                 best_model.selected_feature_names, class_names)
    
    # Save improved model
    model_path = f"models/improved_audio_classifier_{best_model_type}.joblib"
    os.makedirs('models', exist_ok=True)
    best_model.save_model(model_path)
    
    print(f"\n💾 Improved model saved to: {model_path}")
    print("🎉 Improved training complete!")
    
    return best_model, feature_names

if __name__ == "__main__":
    main()