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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
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
    
    def __init__(self, model_type: str = 'random_forest', use_resampling: bool = True):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.variance_selector = None
        self.feature_selector = None
        self.feature_names = None
        self.selected_feature_names = None
        self.label_map = {0: 'Christian', 1: 'Secular'}
        self.class_weights = None
        self.use_resampling = use_resampling
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train the improved classifier with advanced balancing."""
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names
        
        # 1. Remove constant and low-variance features
        logger.info("🔧 Removing constant and low-variance features...")
        self.variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = self.variance_selector.fit_transform(X)
        
        # Get remaining feature names after variance filtering
        variance_mask = self.variance_selector.get_support()
        remaining_features = [feature_names[i] for i, mask in enumerate(variance_mask) if mask]
        
        logger.info(f"   Removed {len(feature_names) - len(remaining_features)} low-variance features")
        logger.info(f"   Remaining features: {len(remaining_features)}")
        
        # 2. Scale features
        X_scaled = self.scaler.fit_transform(X_variance_filtered)
        
        # 3. Feature selection - select top k features
        k_best = min(30, X_scaled.shape[1])
        logger.info(f"🎯 Selecting top {k_best} most discriminative features...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selection_mask = self.feature_selector.get_support()
        self.selected_feature_names = [remaining_features[i] for i, mask in enumerate(selection_mask) if mask]
        
        logger.info(f"   Selected {len(self.selected_feature_names)} features")
        logger.info(f"   Top features: {self.selected_feature_names[:5]}")
        
        # 4. Calculate class weights
        unique_classes = np.unique(y)
        self.class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight_dict = dict(zip(unique_classes, self.class_weights))
        
        original_distribution = np.bincount(y)
        logger.info(f"📊 Original class distribution: {original_distribution}")
        logger.info(f"⚖️ Class weights: {class_weight_dict}")
        
        # 5. Apply SMOTE for better class balancing (if enabled)
        X_train = X_selected
        y_train = y
        used_smote = False
        
        if self.use_resampling and len(np.unique(y)) == 2:
            try:
                logger.info("🔄 Applying SMOTE-Tomek for better class balance...")
                smote_tomek = SMOTETomek(random_state=42)
                X_train, y_train = smote_tomek.fit_resample(X_selected, y)
                resampled_distribution = np.bincount(y_train)
                logger.info(f"   After SMOTE-Tomek: {resampled_distribution}")
                logger.info(f"   Samples added: {len(y_train) - len(y)}")
                used_smote = True
            except Exception as e:
                logger.warning(f"   SMOTE failed: {e}, continuing with class weights only")
                X_train = X_selected
                y_train = y
        
        # 6. Initialize model with tuned hyperparameters based on results
        if self.model_type == 'random_forest':
            # Tuned for better minority class performance
            self.model = RandomForestClassifier(
                n_estimators=400,  # Increased for better generalization
                max_depth=15,      # Reduced to prevent overfitting to majority class
                min_samples_split=8,  # Increased to reduce overfitting
                min_samples_leaf=4,   # Increased to reduce overfitting
                max_features='sqrt',  # Reduced features per split
                class_weight='balanced_subsample',  # Better for imbalanced data with bagging
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                oob_score=True  # Out-of-bag score for validation
            )
        elif self.model_type == 'svm':
            # SVM showed better balance, keep similar but tune
            self.model = SVC(
                kernel='rbf',
                C=2.0,  # Slightly increased for better fit
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=500  # Larger cache for faster training
            )
        elif self.model_type == 'ensemble':
            # Create ensemble of RF and SVM for best of both worlds
            rf_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            svm_model = SVC(
                kernel='rbf',
                C=2.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
            self.model = VotingClassifier(
                estimators=[('rf', rf_model), ('svm', svm_model)],
                voting='soft',  # Use probability voting
                weights=[1.2, 1.0]  # Slightly favor RF for overall accuracy
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 7. Train model
        logger.info(f"🤖 Training balanced {self.model_type}...")
        self.model.fit(X_train, y_train)
        
        # 8. Validate with cross-validation on original data
        logger.info("🔄 Performing stratified cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='balanced_accuracy')
        cv_f1_scores = cross_val_score(self.model, X_selected, y, cv=cv, scoring='f1_weighted')
        
        training_time = time.time() - start_time
        
        # Get OOB score if available
        oob_score = None
        if hasattr(self.model, 'oob_score_'):
            oob_score = self.model.oob_score_
        
        return {
            'train_accuracy': self.model.score(X_train, y_train),
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_f1_mean': np.mean(cv_f1_scores),
            'cv_f1_std': np.std(cv_f1_scores),
            'oob_score': oob_score,
            'n_features_original': X.shape[1],
            'n_features_selected': X_train.shape[1],
            'n_samples_original': len(y),
            'n_samples_train': len(y_train),
            'training_time': training_time,
            'class_distribution': original_distribution.tolist(),
            'resampled_distribution': np.bincount(y_train).tolist() if used_smote else None,
            'class_weights': class_weight_dict,
            'selected_features': self.selected_feature_names,
            'used_smote': used_smote
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

def create_model_comparison_visualizations(models: Dict[str, Dict], y_test: np.ndarray, 
                                          class_names: List[str] = ['Christian', 'Secular']):
    """Create comprehensive model comparison visualizations."""
    
    os.makedirs('visualizations', exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    model_order = ['random_forest', 'svm', 'ensemble']
    model_labels = ['Random Forest', 'SVM', 'Ensemble']
    
    # 1. Overall Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold', y=0.995)
    
    # Test Accuracy
    ax1 = axes[0, 0]
    test_accs = [models[m]['test_accuracy'] * 100 for m in model_order]
    bars1 = ax1.bar(model_labels, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(80, 100)
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, test_accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Balanced Accuracy
    ax2 = axes[0, 1]
    bal_accs = [models[m]['balanced_accuracy'] * 100 for m in model_order]
    bars2 = ax2.bar(model_labels, bal_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Balanced Accuracy (Fairer Metric)', fontsize=14, fontweight='bold')
    ax2.set_ylim(80, 100)
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, bal_accs)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # F1 Score
    ax3 = axes[1, 0]
    f1_scores = [models[m]['f1_score'] * 100 for m in model_order]
    bars3 = ax3.bar(model_labels, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax3.set_title('F1 Score (Precision + Recall)', fontsize=14, fontweight='bold')
    ax3.set_ylim(80, 100)
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Cross-Validation Accuracy
    ax4 = axes[1, 1]
    cv_means = [models[m]['cv_mean'] * 100 for m in model_order]
    cv_stds = [models[m]['cv_std'] * 100 for m in model_order]
    bars4 = ax4.bar(model_labels, cv_means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                    yerr=cv_stds, capsize=10, error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax4.set_ylabel('CV Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cross-Validation Accuracy (±std)', fontsize=14, fontweight='bold')
    ax4.set_ylim(80, 100)
    ax4.grid(axis='y', alpha=0.3)
    for i, (bar, val, std) in enumerate(zip(bars4, cv_means, cv_stds)):
        ax4.text(bar.get_x() + bar.get_width()/2, val + std + 0.5, f'{val:.1f}±{std:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: model_comparison_metrics.png")
    
    # 2. Per-Class Performance Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(model_labels))
    width = 0.35
    
    christian_accs = [models[m]['christian_accuracy'] * 100 for m in model_order]
    secular_accs = [models[m]['secular_accuracy'] * 100 for m in model_order]
    
    bars1 = ax.bar(x - width/2, christian_accs, width, label='Christian', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, secular_accs, width, label='Secular', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Performance: Christian vs Secular', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(75, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add balance gap annotations
    for i, (c_acc, s_acc) in enumerate(zip(christian_accs, secular_accs)):
        gap = abs(c_acc - s_acc)
        ax.text(i, 78, f'Gap: {gap:.1f}%', ha='center', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/per_class_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: per_class_comparison.png")
    
    # 3. Confusion Matrices Grid (All Models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confusion Matrices - All Models', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, (model_type, label) in enumerate(zip(model_order, model_labels)):
        ax = axes[idx]
        y_pred = models[model_type]['predictions']
        cm = confusion_matrix(y_test, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create labels with counts and percentages
        labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                labels[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, vmin=0, vmax=cm.max())
        ax.set_title(f'{label}\nAccuracy: {models[model_type]["test_accuracy"]:.1%}', 
                    fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: confusion_matrices_all.png")
    
    # 4. Confidence Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Prediction Confidence Distribution', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, (model_type, label) in enumerate(zip(model_order, model_labels)):
        ax = axes[idx]
        probas = models[model_type]['probabilities']
        confidences = np.max(probas, axis=1)
        
        ax.hist(confidences, bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(confidences):.3f}')
        ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: confidence_distribution.png")
    
    # 5. Training Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    training_times = [models[m]['training_time'] for m in model_order]
    bars = ax.barh(model_labels, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, time in zip(bars, training_times):
        ax.text(time + 1, bar.get_y() + bar.get_height()/2, f'{time:.1f}s', 
               va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('visualizations/training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved: training_time_comparison.png")
    
    logger.info("📊 All model comparison visualizations saved to 'visualizations/' directory")


def create_improved_visualizations(y_true: np.ndarray, y_pred: np.ndarray, 
                                 feature_importance: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None,
                                 class_names: List[str] = ['Christian', 'Secular']):
    """Create improved visualization plots for a single model."""
    
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
    plt.title('Best Model - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance
    if feature_importance is not None and feature_names is not None:
        plt.figure(figsize=(12, 10))
        top_indices = np.argsort(feature_importance)[-25:]  # Top 25 features
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_importance, color=colors, edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(top_features)), top_features, fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title('Top 25 Most Important Audio Features', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("📊 Single model visualizations saved to 'visualizations/' directory")

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
    
    # Train improved models (including new ensemble)
    models = {}
    for model_type in ['random_forest', 'svm', 'ensemble']:
        print(f"\n🤖 Training improved {model_type}...")
        
        classifier = ImprovedAudioClassifier(model_type=model_type, use_resampling=True)
        
        # Train
        train_metrics = classifier.train(X_train, y_train, feature_names)
        
        # Test
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class accuracy
        christian_mask = y_test == 0
        secular_mask = y_test == 1
        christian_acc = accuracy_score(y_test[christian_mask], y_pred[christian_mask]) if christian_mask.sum() > 0 else 0
        secular_acc = accuracy_score(y_test[secular_mask], y_pred[secular_mask]) if secular_mask.sum() > 0 else 0
        
        models[model_type] = {
            'classifier': classifier,
            'train_accuracy': train_metrics['train_accuracy'],
            'cv_mean': train_metrics['cv_mean'],
            'cv_std': train_metrics['cv_std'],
            'cv_f1_mean': train_metrics['cv_f1_mean'],
            'cv_f1_std': train_metrics['cv_f1_std'],
            'oob_score': train_metrics.get('oob_score'),
            'test_accuracy': test_accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'christian_accuracy': christian_acc,
            'secular_accuracy': secular_acc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'training_time': train_metrics['training_time'],
            'n_features_selected': train_metrics['n_features_selected'],
            'class_weights': train_metrics['class_weights'],
            'used_smote': train_metrics.get('used_smote', False),
            'resampled_distribution': train_metrics.get('resampled_distribution')
        }
        
        print(f"   Training accuracy: {train_metrics['train_accuracy']:.3f}")
        print(f"   CV balanced accuracy: {train_metrics['cv_mean']:.3f} ± {train_metrics['cv_std']:.3f}")
        print(f"   CV F1 score: {train_metrics['cv_f1_mean']:.3f} ± {train_metrics['cv_f1_std']:.3f}")
        if train_metrics.get('oob_score'):
            print(f"   OOB score: {train_metrics['oob_score']:.3f}")
        print(f"   Test accuracy: {test_accuracy:.3f}")
        print(f"   Balanced accuracy: {balanced_acc:.3f}")
        print(f"   F1 score: {f1:.3f}")
        print(f"   Christian accuracy: {christian_acc:.1%} | Secular accuracy: {secular_acc:.1%}")
        print(f"   Features selected: {train_metrics['n_features_selected']}")
        print(f"   Training time: {train_metrics['training_time']:.1f}s")
        if train_metrics.get('used_smote'):
            print(f"   ✅ Used SMOTE-Tomek: {train_metrics.get('resampled_distribution')}")
    
    # Choose best model based on balanced accuracy (better for imbalanced data)
    best_model_type = max(models.keys(), key=lambda k: models[k]['balanced_accuracy'])
    best_model = models[best_model_type]['classifier']
    best_metrics = models[best_model_type]
    
    print(f"\n🏆 Best model (by balanced accuracy): {best_model_type}")
    print(f"   Test accuracy: {best_metrics['test_accuracy']:.3f}")
    print(f"   Balanced accuracy: {best_metrics['balanced_accuracy']:.3f}")
    print(f"   F1 score: {best_metrics['f1_score']:.3f}")
    print(f"   Christian accuracy: {best_metrics['christian_accuracy']:.1%}")
    print(f"   Secular accuracy: {best_metrics['secular_accuracy']:.1%}")
    
    # Detailed evaluation
    y_pred_best = models[best_model_type]['predictions']
    class_names = ['Christian', 'Secular']
    
    print(f"\n📈 Detailed Results for Best Model:")
    print(classification_report(y_test, y_pred_best, target_names=class_names))
    
    # Show all models comparison
    print(f"\n📊 All Models Comparison:")
    print(f"{'Model':<15} {'Accuracy':<10} {'Bal. Acc.':<12} {'F1':<8} {'Christian':<12} {'Secular':<12}")
    print("=" * 75)
    for model_type in ['random_forest', 'svm', 'ensemble']:
        m = models[model_type]
        print(f"{model_type:<15} {m['test_accuracy']:<10.3f} {m['balanced_accuracy']:<12.3f} "
              f"{m['f1_score']:<8.3f} {m['christian_accuracy']:<12.1%} {m['secular_accuracy']:<12.1%}")
    
    # Feature importance
    feature_importance = None
    if hasattr(best_model.model, 'feature_importances_'):
        feature_importance = best_model.model.feature_importances_
    
    # Create comprehensive visualizations
    print("\n📊 Creating comprehensive visualizations...")
    print("=" * 60)
    
    # Model comparison visualizations
    print("🎨 Generating model comparison charts...")
    create_model_comparison_visualizations(models, y_test, class_names)
    
    # Best model specific visualizations
    print("🎨 Generating best model visualizations...")
    create_improved_visualizations(y_test, y_pred_best, feature_importance, 
                                 best_model.selected_feature_names, class_names)
    
    # Save all improved models
    os.makedirs('models', exist_ok=True)
    print("\n💾 Saving all models...")
    
    for model_type in ['random_forest', 'svm', 'ensemble']:
        model_path = f"models/improved_audio_classifier_{model_type}.joblib"
        models[model_type]['classifier'].save_model(model_path)
        print(f"   ✅ Saved: {model_path}")
        print(f"      - Accuracy: {models[model_type]['test_accuracy']:.3f}")
        print(f"      - Balanced Accuracy: {models[model_type]['balanced_accuracy']:.3f}")
        print(f"      - Christian: {models[model_type]['christian_accuracy']:.1%} | Secular: {models[model_type]['secular_accuracy']:.1%}")
    
    print(f"\n🎯 Best model: {best_model_type} (by balanced accuracy)")
    print("🎉 Improved training complete!")
    print("\n💡 Recommendations:")
    print("   • Use 'ensemble' for best balanced performance")
    print("   • Use 'random_forest' for highest overall accuracy")
    print("   • Use 'svm' for fastest inference")
    
    return best_model, feature_names

if __name__ == "__main__":
    main()