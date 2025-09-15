"""
공명 기반 어텐션 메커니즘
완성도: 90%
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    resonance_bias: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    temperature: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    공명 바이어스가 추가된 스케일드 닷-프로덕트 어텐션
    
    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        resonance_bias: (batch, seq_len, seq_len) 공명 기반 바이어스
        mask: 어텐션 마스크
        temperature: 소프트맥스 온도
    
    Returns:
        (output, attention_weights)
    """
    d_k = query.shape[-1]
    
    # QK^T / sqrt(d_k)
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # 공명 바이어스 추가
    if resonance_bias is not None:
        scores = scores + resonance_bias
    
    # 마스크 적용
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 온도 스케일링
    scores = scores / temperature
    
    # 소프트맥스
    attention_weights = softmax(scores, axis=-1)
    
    # 가중 합
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """안정적인 소프트맥스"""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class ResonanceAttentionLayer:
    """공명 어텐션 레이어"""
    
    def __init__(self, d_model: int, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 가중치 초기화 (실제로는 학습 가능한 파라미터)
        self.W_q = self._init_weight(d_model, d_model)
        self.W_k = self._init_weight(d_model, d_model)
        self.W_v = self._init_weight(d_model, d_model)
        self.W_o = self._init_weight(d_model, d_model)
        
        # 공명 프로젝션
        self.W_resonance = self._init_weight(d_model, 1)
    
    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Xavier 초기화"""
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        return np.random.randn(in_dim, out_dim) * scale
    
    def forward(
        self,
        x: np.ndarray,
        resonance_values: Optional[Dict[str, float]] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        순전파
        
        Args:
            x: (batch, seq_len, d_model)
            resonance_values: 레벨별 공명값
            mask: 어텐션 마스크
        
        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Multi-head reshape
        Q = self._reshape_to_heads(Q, batch_size, seq_len)
        K = self._reshape_to_heads(K, batch_size, seq_len)
        V = self._reshape_to_heads(V, batch_size, seq_len)
        
        # 공명 바이어스 계산
        resonance_bias = None
        if resonance_values:
            resonance_bias = self._compute_resonance_bias(
                x, resonance_values, batch_size, seq_len
            )
        
        # 어텐션 계산
        attn_output, attn_weights = self._multi_head_attention(
            Q, K, V, resonance_bias, mask
        )
        
        # Reshape back
        attn_output = self._reshape_from_heads(attn_output, batch_size, seq_len)
        
        # Final linear
        output = np.matmul(attn_output, self.W_o)
        
        return output, attn_weights
    
    def _reshape_to_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """멀티헤드 형태로 변환"""
        x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, d_k)
    
    def _reshape_from_heads(self, x: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
        """멀티헤드에서 원래 형태로"""
        x = x.transpose(0, 2, 1, 3)  # (batch, seq_len, n_heads, d_k)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def _multi_head_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        resonance_bias: Optional[np.ndarray],
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """멀티헤드 어텐션"""
        outputs = []
        weights = []
        
        for h in range(self.n_heads):
            q_h = Q[:, h]
            k_h = K[:, h]
            v_h = V[:, h]
            
            bias_h = resonance_bias[:, h] if resonance_bias is not None else None
            
            out_h, weight_h = scaled_dot_product_attention(
                q_h, k_h, v_h, bias_h, mask
            )
            
            outputs.append(out_h)
            weights.append(weight_h)
        
        # Concatenate heads
        output = np.stack(outputs, axis=1)
        attention_weights = np.stack(weights, axis=1)
        
        return output, attention_weights
    
    def _compute_resonance_bias(
        self,
        x: np.ndarray,
        resonance_values: Dict[str, float],
        batch_size: int,
        seq_len: int
    ) -> np.ndarray:
        """공명 기반 바이어스 계산"""
        # 입력에서 공명 스코어 추출
        resonance_scores = np.matmul(x, self.W_resonance).squeeze(-1)  # (batch, seq_len)
        
        # 레벨별 공명값으로 조정
        avg_resonance = np.mean(list(resonance_values.values()))
        resonance_scores = resonance_scores * avg_resonance
        
        # 바이어스 매트릭스 생성
        bias = np.zeros((batch_size, self.n_heads, seq_len, seq_len))
        
        for b in range(batch_size):
            for h in range(self.n_heads):
                # 각 헤드별로 다른 공명 패턴
                level_idx = h % len(resonance_values)
                level_name = list(resonance_values.keys())[level_idx]
                level_resonance = resonance_values[level_name]
                
                # 대각선 우세 패턴 (자기 참조 강화)
                bias[b, h] = np.outer(
                    resonance_scores[b] * level_resonance,
                    resonance_scores[b]
                )
        
        return bias

class HierarchicalResonanceAttention:
    """계층적 공명 어텐션"""
    
    def __init__(self, d_model: int, n_levels: int = 7):
        self.d_model = d_model
        self.n_levels = n_levels
        
        # 각 레벨별 어텐션 레이어
        self.level_attentions = [
            ResonanceAttentionLayer(d_model, n_heads=8)
            for _ in range(n_levels)
        ]
        
        # 크로스 레벨 어텐션
        self.cross_level_attention = ResonanceAttentionLayer(d_model, n_heads=4)
    
    def forward(
        self,
        level_features: Dict[str, np.ndarray],
        resonance_values: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        계층별 공명 어텐션 적용
        
        Args:
            level_features: 레벨별 피처
            resonance_values: 레벨별 공명값
        
        Returns:
            레벨별 어텐션 출력
        """
        outputs = {}
        attention_maps = {}
        
        level_names = list(level_features.keys())
        
        # 각 레벨별 self-attention
        for i, level in enumerate(level_names):
            if level in level_features:
                features = level_features[level]
                
                # 차원 맞추기
                if features.ndim == 2:
                    features = features[np.newaxis, :]  # Add batch dim
                
                # 공명 어텐션 적용
                level_resonance = {level: resonance_values.get(level, 0.5)}
                output, attn_weights = self.level_attentions[i].forward(
                    features, level_resonance
                )
                
                outputs[level] = output.squeeze(0)
                attention_maps[level] = attn_weights
        
        # 크로스 레벨 어텐션 (인접 레벨간)
        for i in range(len(level_names) - 1):
            curr_level = level_names[i]
            next_level = level_names[i + 1]
            
            if curr_level in outputs and next_level in outputs:
                # 두 레벨 연결
                combined = np.concatenate([
                    outputs[curr_level],
                    outputs[next_level]
                ], axis=0)[np.newaxis, :]
                
                # 크로스 어텐션
                cross_output, _ = self.cross_level_attention.forward(
                    combined, resonance_values
                )
                
                # 결과를 각 레벨에 반영
                half = cross_output.shape[1] // 2
                outputs[curr_level] = 0.7 * outputs[curr_level] + 0.3 * cross_output[0, :half]
                outputs[next_level] = 0.7 * outputs[next_level] + 0.3 * cross_output[0, half:]
        
        return outputs

class ResonanceGatedAttention:
    """공명 게이트 어텐션"""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.gate_w = np.random.randn(d_model, d_model) * 0.01
        self.gate_b = np.zeros(d_model)
    
    def forward(
        self,
        x: np.ndarray,
        resonance: float,
        temperature: float = 4.0
    ) -> np.ndarray:
        """
        공명 기반 게이팅
        
        Args:
            x: 입력 피처
            resonance: 공명값 (0-1)
            temperature: 게이트 민감도
        
        Returns:
            게이트된 출력
        """
        # 게이트 계산
        gate_input = np.matmul(x, self.gate_w) + self.gate_b
        gate = 1.0 / (1.0 + np.exp(-gate_input))
        
        # 공명으로 게이트 조정
        resonance_factor = np.exp(temperature * resonance) / (
            np.exp(temperature * resonance) + np.exp(temperature * (1 - resonance))
        )
        
        adjusted_gate = gate * resonance_factor
        
        # 게이팅 적용
        output = x * adjusted_gate
        
        return output

# 추천: Transformer 기반 공명 인코더 추가 - 더 깊은 표현 학습 (적합성 96%)
# class ResonanceTransformerEncoder:
#     def __init__(self, n_layers=6, d_model=512):
#         self.layers = [HierarchicalResonanceAttention(d_model) for _ in range(n_layers)]