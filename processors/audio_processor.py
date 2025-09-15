"""
오디오 감정 처리 모듈
완성도: 89%
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioEmotionProcessor:
    """오디오 기반 감정 분석"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = 512
        
        # 감정별 음향 특징 프로파일
        self.emotion_profiles = {
            'joy': {
                'pitch_mean': 250,
                'pitch_std': 80,
                'energy': 0.8,
                'tempo': 120,
                'spectral_centroid': 2500
            },
            'sadness': {
                'pitch_mean': 180,
                'pitch_std': 30,
                'energy': 0.3,
                'tempo': 60,
                'spectral_centroid': 1500
            },
            'anger': {
                'pitch_mean': 280,
                'pitch_std': 100,
                'energy': 0.9,
                'tempo': 140,
                'spectral_centroid': 3000
            },
            'fear': {
                'pitch_mean': 320,
                'pitch_std': 120,
                'energy': 0.7,
                'tempo': 110,
                'spectral_centroid': 2800
            }
        }
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, float]:
        """오디오 특징 추출"""
        features = {}
        
        # 기본 특징
        features['energy'] = self._compute_energy(audio)
        features['zcr'] = self._compute_zcr(audio)
        
        # 스펙트럴 특징
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        features['spectral_centroid'] = self._spectral_centroid(spectrum, freqs)
        features['spectral_spread'] = self._spectral_spread(spectrum, freqs, features['spectral_centroid'])
        features['spectral_flux'] = self._spectral_flux(audio)
        features['spectral_rolloff'] = self._spectral_rolloff(spectrum, freqs)
        
        # 프로소디 특징
        features['f0_mean'], features['f0_std'] = self._estimate_pitch(audio)
        features['jitter'] = self._compute_jitter(audio)
        features['shimmer'] = self._compute_shimmer(audio)
        
        # MFCC (간단한 버전)
        mfcc = self._compute_mfcc_simple(audio)
        for i, coef in enumerate(mfcc[:13]):
            features[f'mfcc_{i}'] = coef
        
        # 템포 추정
        features['tempo'] = self._estimate_tempo(audio)
        
        return features
    
    def analyze_emotion(self, audio: np.ndarray) -> Dict[str, float]:
        """오디오에서 감정 분석"""
        features = self.extract_features(audio)
        
        # 각 감정과의 거리 계산
        emotion_scores = {}
        
        for emotion, profile in self.emotion_profiles.items():
            distance = 0
            weights = {
                'pitch_mean': 0.25,
                'energy': 0.25,
                'tempo': 0.2,
                'spectral_centroid': 0.3
            }
            
            for feat, weight in weights.items():
                if feat == 'pitch_mean':
                    feat_key = 'f0_mean'
                else:
                    feat_key = feat
                
                if feat_key in features:
                    observed = features[feat_key]
                    expected = profile[feat]
                    
                    # 정규화된 거리
                    if feat == 'energy':
                        dist = abs(observed - expected)
                    else:
                        dist = abs(observed - expected) / (expected + 1e-9)
                    
                    distance += weight * dist
            
            # 거리를 유사도로 변환
            similarity = np.exp(-distance)
            emotion_scores[emotion] = similarity
        
        # 정규화
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # 7차원 벡터로 변환
        emotion_vector = [
            emotion_scores.get('joy', 0),
            emotion_scores.get('sadness', 0),
            emotion_scores.get('anger', 0),
            emotion_scores.get('fear', 0),
            emotion_scores.get('disgust', 0),
            emotion_scores.get('surprise', 0),
            emotion_scores.get('neutral', 0.1)
        ]
        
        return {
            'emotion_vector': emotion_vector,
            'features': features,
            'dominant_emotion': max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
        }
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """에너지 계산"""
        return float(np.sqrt(np.mean(audio**2)))
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """영교차율"""
        signs = np.sign(audio)
        signs[signs == 0] = 1
        zcr = np.mean(np.abs(np.diff(signs)) / 2)
        return float(zcr)
    
    def _spectral_centroid(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """스펙트럴 중심"""
        magnitude = np.abs(spectrum)
        if magnitude.sum() > 0:
            return float(np.sum(freqs * magnitude) / np.sum(magnitude))
        return 0.0
    
    def _spectral_spread(self, spectrum: np.ndarray, freqs: np.ndarray, centroid: float) -> float:
        """스펙트럴 분산"""
        magnitude = np.abs(spectrum)
        if magnitude.sum() > 0:
            return float(np.sqrt(np.sum(((freqs - centroid)**2) * magnitude) / np.sum(magnitude)))
        return 0.0
    
    def _spectral_flux(self, audio: np.ndarray) -> float:
        """스펙트럴 플럭스"""
        frames = self._frame_audio(audio)
        flux = []
        
        prev_spectrum = np.zeros(self.frame_length // 2 + 1)
        for frame in frames:
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            flux.append(np.sum((spectrum - prev_spectrum)**2))
            prev_spectrum = spectrum
        
        return float(np.mean(flux)) if flux else 0.0
    
    def _spectral_rolloff(self, spectrum: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
        """스펙트럴 롤오프"""
        magnitude = np.abs(spectrum)
        cumsum = np.cumsum(magnitude)
        
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= threshold * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                return float(freqs[rolloff_idx[0]])
        return float(freqs[-1])
    
    def _estimate_pitch(self, audio: np.ndarray) -> Tuple[float, float]:
        """피치 추정 (자기상관 기반)"""
        # 간단한 자기상관 기반 피치 추정
        autocorr = np.correlate(audio, audio, mode='same')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 피크 찾기
        min_period = int(self.sample_rate / 400)  # 400 Hz max
        max_period = int(self.sample_rate / 50)   # 50 Hz min
        
        if len(autocorr) > max_period:
            autocorr_segment = autocorr[min_period:max_period]
            if len(autocorr_segment) > 0:
                peak_idx = np.argmax(autocorr_segment) + min_period
                f0 = self.sample_rate / peak_idx
                
                # 여러 프레임에서 피치 추정
                frames = self._frame_audio(audio)
                pitches = []
                for frame in frames:
                    frame_corr = np.correlate(frame, frame, mode='same')
                    frame_corr = frame_corr[len(frame_corr)//2:]
                    if len(frame_corr) > min_period:
                        segment = frame_corr[min_period:min(max_period, len(frame_corr))]
                        if len(segment) > 0 and segment.max() > 0:
                            idx = np.argmax(segment) + min_period
                            pitches.append(self.sample_rate / idx)
                
                if pitches:
                    return float(np.mean(pitches)), float(np.std(pitches))
                return f0, 10.0
        
        return 200.0, 50.0  # 기본값
    
    def _compute_jitter(self, audio: np.ndarray) -> float:
        """지터 (피치 변동성)"""
        frames = self._frame_audio(audio)
        periods = []
        
        for frame in frames:
            autocorr = np.correlate(frame, frame, mode='same')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 20:
                peak_idx = np.argmax(autocorr[20:]) + 20
                periods.append(peak_idx)
        
        if len(periods) > 1:
            period_diffs = np.abs(np.diff(periods))
            jitter = np.mean(period_diffs) / (np.mean(periods) + 1e-9)
            return float(jitter)
        
        return 0.01
    
    def _compute_shimmer(self, audio: np.ndarray) -> float:
        """쉬머 (진폭 변동성)"""
        frames = self._frame_audio(audio)
        amplitudes = [np.max(np.abs(frame)) for frame in frames]
        
        if len(amplitudes) > 1:
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer = np.mean(amp_diffs) / (np.mean(amplitudes) + 1e-9)
            return float(shimmer)
        
        return 0.02
    
    def _compute_mfcc_simple(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """간단한 MFCC 계산"""
        # Mel 필터뱅크 시뮬레이션
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        
        # 로그 멜 스펙트럼 (간소화)
        log_spectrum = np.log(spectrum + 1e-9)
        
        # DCT로 MFCC 추출
        mfcc = np.zeros(n_mfcc)
        for i in range(n_mfcc):
            mfcc[i] = np.sum(log_spectrum * np.cos(np.pi * i * np.arange(len(log_spectrum)) / len(log_spectrum)))
        
        return mfcc / (np.linalg.norm(mfcc) + 1e-9)
    
    def _estimate_tempo(self, audio: np.ndarray) -> float:
        """템포 추정"""
        # 에너지 엔벨로프
        frames = self._frame_audio(audio)
        energy_envelope = [np.sqrt(np.mean(frame**2)) for frame in frames]
        
        if len(energy_envelope) > 10:
            # 자기상관으로 비트 찾기
            autocorr = np.correlate(energy_envelope, energy_envelope, mode='same')
            autocorr = autocorr[len(autocorr)//2:]
            
            # BPM 범위: 40-200
            min_lag = int(60 * len(frames) / (200 * len(audio) / self.sample_rate))
            max_lag = int(60 * len(frames) / (40 * len(audio) / self.sample_rate))
            
            if len(autocorr) > max_lag:
                segment = autocorr[min_lag:max_lag]
                if len(segment) > 0:
                    peak_lag = np.argmax(segment) + min_lag
                    tempo = 60 * len(frames) / (peak_lag * len(audio) / self.sample_rate)
                    return float(np.clip(tempo, 40, 200))
        
        return 90.0  # 기본 템포
    
    def _frame_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """오디오 프레이밍"""
        frames = []
        for i in range(0, len(audio) - self.frame_length, self.hop_length):
            frames.append(audio[i:i + self.frame_length])
        return frames

class MultimodalAudioIntegrator:
    """멀티모달 오디오 통합"""
    
    def __init__(self):
        self.audio_processor = AudioEmotionProcessor()
    
    def integrate_with_text(
        self,
        audio_features: Dict[str, float],
        text_emotion: np.ndarray,
        weights: Tuple[float, float] = (0.4, 0.6)
    ) -> np.ndarray:
        """오디오와 텍스트 감정 통합"""
        audio_emotion = audio_features.get('emotion_vector', [0]*7)
        
        # 가중 평균
        integrated = (
            weights[0] * np.array(audio_emotion) +
            weights[1] * np.array(text_emotion)
        )
        
        # 정규화
        return integrated / (np.linalg.norm(integrated) + 1e-9)
    
    def synchronize_with_eeg(
        self,
        audio_timeline: List[Dict],
        eeg_timeline: List[Dict],
        window_ms: int = 500
    ) -> List[Dict]:
        """오디오-EEG 시간 동기화"""
        synchronized = []
        
        for audio_event in audio_timeline:
            audio_time = audio_event['timestamp']
            
            # 시간 윈도우 내 EEG 이벤트 찾기
            matched_eeg = []
            for eeg_event in eeg_timeline:
                if abs(eeg_event['timestamp'] - audio_time) < window_ms / 1000:
                    matched_eeg.append(eeg_event)
            
            if matched_eeg:
                # 평균 EEG 특징
                avg_eeg = np.mean([e['features'] for e in matched_eeg], axis=0)
                synchronized.append({
                    'timestamp': audio_time,
                    'audio': audio_event['features'],
                    'eeg': avg_eeg
                })
        
        return synchronized

# 추천: 실시간 오디오 스트리밍 처리 추가 (적합성 95%)
# class RealtimeAudioStream:
#     def __init__(self):
#         import sounddevice as sd
#         self.stream = sd.InputStream(callback=self.audio_callback)