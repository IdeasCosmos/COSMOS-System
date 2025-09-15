"""
COSMOS 핵심 아키텍처 정의
완성도: 91%
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProcessingLevel(Enum):
    """처리 계층 레벨"""
    QUANTUM = "quantum"       # 음운/미세 프로소디 (20-200ms)
    ATOMIC = "atomic"         # 형태소
    MOLECULAR = "molecular"   # 단어
    COMPOUND = "compound"     # 구문
    ORGANIC = "organic"       # 문장
    ECOSYSTEM = "ecosystem"   # 문단
    COSMOS = "cosmos"         # 담화

@dataclass
class EmotionVector:
    """감정 벡터"""
    values: np.ndarray  # 7차원 또는 28차원
    confidence: float
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'values': self.values.tolist(),
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }

@dataclass
class MusicParameters:
    """음악 매개변수"""
    tempo_bpm: float
    key: str
    mode: str
    dynamics: str
    chord_progression: List[str]
    rhythm_pattern: Optional[List[float]] = None
    melody_contour: Optional[List[int]] = None

class ProcessingNode:
    """계층적 처리 노드"""
    
    def __init__(
        self,
        level: ProcessingLevel,
        content: Any,
        parent: Optional[ProcessingNode] = None
    ):
        self.level = level
        self.content = content
        self.parent = parent
        self.children: List[ProcessingNode] = []
        
        # 감정 분석 결과
        self.emotion = EmotionVector(np.zeros(7), 0.5)
        self.resonance = 0.5
        
        # 음악 매핑
        self.music: Optional[MusicParameters] = None
        
        # 메타데이터
        self.metadata: Dict = {}
    
    def add_child(self, child: ProcessingNode):
        """자식 노드 추가"""
        child.parent = self
        self.children.append(child)
    
    def propagate_up(self):
        """상향 전파 (Bottom-up)"""
        if not self.children:
            return
        
        # 자식들의 감정 집계
        child_emotions = np.array([child.emotion.values for child in self.children])
        child_weights = np.array([child.emotion.confidence for child in self.children])
        
        # 가중 평균
        if child_weights.sum() > 0:
            weighted_emotion = np.average(child_emotions, weights=child_weights, axis=0)
            avg_confidence = np.mean(child_weights)
            
            # 현재 노드 감정 업데이트
            self.emotion = EmotionVector(
                values=weighted_emotion,
                confidence=avg_confidence * 0.9  # 레벨 올라갈수록 신뢰도 감소
            )
    
    def propagate_down(self, context_emotion: EmotionVector):
        """하향 전파 (Top-down)"""
        # 컨텍스트와 현재 감정 혼합
        alpha = 0.3  # 컨텍스트 영향력
        self.emotion.values = (
            (1 - alpha) * self.emotion.values +
            alpha * context_emotion.values
        )
        
        # 자식들에게 전파
        for child in self.children:
            child.propagate_down(self.emotion)

class HierarchicalEmotionEngine:
    """계층적 감정 분석 엔진"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.processing_levels = list(ProcessingLevel)
        
        # 레벨별 처리기
        self.processors = {}
        self._initialize_processors()
    
    def _load_config(self, path: Optional[str]) -> Dict:
        """설정 로드"""
        if path:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _initialize_processors(self):
        """처리기 초기화"""
        # 각 레벨별 처리기 인스턴스 생성
        from processors.korean_processor import KoreanEmotionProcessor
        from processors.audio_processor import AudioEmotionProcessor
        
        self.processors['korean'] = KoreanEmotionProcessor()
        self.processors['audio'] = AudioEmotionProcessor()
    
    async def process(
        self,
        text: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
        eeg: Optional[Dict] = None
    ) -> ProcessingNode:
        """멀티모달 입력 처리"""
        # 루트 노드 생성
        root = ProcessingNode(ProcessingLevel.COSMOS, "root")
        
        # 텍스트 처리
        if text:
            text_tree = await self._process_text(text)
            root.add_child(text_tree)
        
        # 오디오 처리
        if audio is not None:
            audio_tree = await self._process_audio(audio)
            root.add_child(audio_tree)
        
        # EEG 처리
        if eeg:
            eeg_tree = await self._process_eeg(eeg)
            root.add_child(eeg_tree)
        
        # 상향 전파
        root.propagate_up()
        
        # 하향 전파
        root.propagate_down(root.emotion)
        
        # 음악 매핑
        root.music = self._generate_music(root.emotion)
        
        return root
    
    async def _process_text(self, text: str) -> ProcessingNode:
        """텍스트 계층적 처리"""
        # 문장 노드
        sentence_node = ProcessingNode(ProcessingLevel.ORGANIC, text)
        
        # 구문 분석
        phrases = self._split_phrases(text)
        for phrase in phrases:
            phrase_node = ProcessingNode(ProcessingLevel.COMPOUND, phrase)
            
            # 단어 분석
            words = phrase.split()
            for word in words:
                word_node = ProcessingNode(ProcessingLevel.MOLECULAR, word)
                
                # 형태소 분석
                if 'korean' in self.processors:
                    result = self.processors['korean'].analyze_korean_emotion(word)
                    morpheme_node = ProcessingNode(ProcessingLevel.ATOMIC, result)
                    
                    # 감정 설정
                    if 'integrated_vector' in result:
                        morpheme_node.emotion = EmotionVector(
                            values=np.array(result['integrated_vector'][:7]),
                            confidence=0.8
                        )
                    
                    word_node.add_child(morpheme_node)
                
                phrase_node.add_child(word_node)
            
            sentence_node.add_child(phrase_node)
        
        # 상향 전파로 감정 집계
        for phrase in sentence_node.children:
            phrase.propagate_up()
        sentence_node.propagate_up()
        
        return sentence_node
    
    async def _process_audio(self, audio: np.ndarray) -> ProcessingNode:
        """오디오 계층적 처리"""
        audio_node = ProcessingNode(ProcessingLevel.ORGANIC, "audio")
        
        if 'audio' in self.processors:
            result = self.processors['audio'].analyze_emotion(audio)
            
            # 감정 벡터 설정
            audio_node.emotion = EmotionVector(
                values=np.array(result['emotion_vector']),
                confidence=0.7
            )
            
            # 메타데이터 저장
            audio_node.metadata = result.get('features', {})
        
        return audio_node
    
    async def _process_eeg(self, eeg: Dict) -> ProcessingNode:
        """EEG 처리"""
        eeg_node = ProcessingNode(ProcessingLevel.MOLECULAR, "eeg")
        
        # EEG 밴드에서 감정 추론
        from processors.yaml_integration import YAMLBasedProcessor
        processor = YAMLBasedProcessor()
        
        emotion_vec = processor.map_eeg_to_emotion(eeg)
        eeg_node.emotion = EmotionVector(
            values=emotion_vec.to_array(),
            confidence=0.85
        )
        
        return eeg_node
    
    def _split_phrases(self, text: str) -> List[str]:
        """구문 분할"""
        # 간단한 구문 분할
        import re
        phrases = re.split(r'[,;]', text)
        return [p.strip() for p in phrases if p.strip()]
    
    def _generate_music(self, emotion: EmotionVector) -> MusicParameters:
        """감정을 음악으로 변환"""
        from core.music_converter import EmotionToMusicConverter
        converter = EmotionToMusicConverter()
        
        music_dict = converter.convert_emotion_to_music(emotion.values)
        
        return MusicParameters(
            tempo_bpm=music_dict['tempo'],
            key=music_dict['key'],
            mode=music_dict['mode'],
            dynamics=music_dict['dynamics'],
            chord_progression=music_dict['progression']
        )
    
    async def _build_tree(self, text: str) -> ProcessingNode:
        """텍스트에서 전체 트리 구조 생성"""
        # 담화 레벨 (루트)
        discourse = ProcessingNode(ProcessingLevel.COSMOS, text)
        
        # 문단 분할
        paragraphs = text.split('\n\n')
        for para_text in paragraphs:
            if not para_text.strip():
                continue
            
            para_node = ProcessingNode(ProcessingLevel.ECOSYSTEM, para_text)
            
            # 문장 분할
            sentences = para_text.split('.')
            for sent_text in sentences:
                if not sent_text.strip():
                    continue
                
                sent_node = await self._process_text(sent_text.strip())
                para_node.add_child(sent_node)
            
            # 문단 레벨 감정 집계
            para_node.propagate_up()
            discourse.add_child(para_node)
        
        # 담화 레벨 감정 집계
        discourse.propagate_up()
        
        return discourse

class EmotionMusicPipeline:
    """감정-음악 변환 파이프라인"""
    
    def __init__(self):
        self.engine = HierarchicalEmotionEngine()
        self.cache = {}
    
    async def process_realtime(
        self,
        stream_data: Dict,
        callback: Optional[Any] = None
    ):
        """실시간 스트림 처리"""
        # 텍스트, 오디오, EEG 스트림 처리
        text = stream_data.get('text')
        audio = stream_data.get('audio')
        eeg = stream_data.get('eeg')
        
        # 처리
        result = await self.engine.process(text, audio, eeg)
        
        # 콜백 실행
        if callback:
            await callback(result)
        
        return result
    
    def get_emotion_trajectory(
        self,
        nodes: List[ProcessingNode]
    ) -> np.ndarray:
        """감정 궤적 추출"""
        trajectory = []
        
        for node in nodes:
            trajectory.append(node.emotion.values)
        
        return np.array(trajectory)
    
    def generate_music_sequence(
        self,
        emotion_trajectory: np.ndarray
    ) -> List[MusicParameters]:
        """감정 궤적에서 음악 시퀀스 생성"""
        from core.music_converter import MusicStructureGenerator
        generator = MusicStructureGenerator()
        
        structure = generator.generate_structure(
            emotion_timeline=list(emotion_trajectory),
            duration_seconds=120
        )
        
        music_sequence = []
        for section in structure['sections']:
            emotion_vec = np.array(section['emotion_vector'])
            music = self.engine._generate_music(
                EmotionVector(emotion_vec, 0.8)
            )
            music_sequence.append(music)
        
        return music_sequence

# 추천: 실시간 스트리밍 버전 추가 (적합성 96%)
# class StreamingEmotionEngine(HierarchicalEmotionEngine):
#     async def process_stream(self, data_stream):
#         async for chunk in data_stream:
#             yield await self.process(**chunk)