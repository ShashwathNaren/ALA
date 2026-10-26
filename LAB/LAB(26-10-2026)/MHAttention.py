import numpy as np 
import unittest

class Hyperparameters:
    def __init__(self, d_in : int = 768, d_attn : int = 64, d_mid : int = 64, d_out : int = 768, H : int = 12, seed : int = 42):
        np.random.seed(seed)
        self.d_attn = d_attn
        self.H = H
        self.d_mid = d_mid
        
        W_q_heads = np.random.randn(H, d_in, d_attn) * 0.02
        W_k_heads = np.random.randn(H, d_in, d_attn) * 0.02
        W_v_heads = np.random.randn(H, d_in, d_mid) * 0.02
        
        b_q_heads = np.zeros((H, d_attn))
        b_k_heads = np.zeros((H, b_q_heads.shape[1]))
        b_v_heads = np.zeros((H, d_mid))
        
        W_o = np.random.randn(H * d_mid, d_out) * 0.02
        b_o = np.zeros(d_out)
        
        self.w = (W_q_heads, W_k_heads, W_v_heads, b_q_heads, b_k_heads, b_v_heads, W_o, b_o)
        
    def attention_head(self, x : np.ndarray, h : int, w_qkv : tuple) -> np.ndarray:
        W_q, W_k, W_v, b_q, b_k, b_v = w_qkv
        
        Q = np.dot(x, W_q[h]) + b_q[h]
        K = np.dot(x, W_k[h]) + b_k[h]
        V = np.dot(x, W_v[h]) + b_v[h]
        
        S = np.dot(Q, K.T) / np.sqrt(self.d_attn)
        
        exp_S = np.exp(S - np.max(S, axis=-1, keepdims=True))
        attention_weights = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        
        return np.dot(attention_weights, V)

    def mhattention(self, x : np.ndarray, w : tuple = ()) -> np.ndarray:
        if len(w) == 0:
            w = self.w
            
        W_q, W_k, W_v, b_q, b_k, b_v, W_o, b_o = w
        w_qkv = (W_q, W_k, W_v, b_q, b_k, b_v)
        
        head_outputs = []
        for h in range(self.H):
            head_out = self.attention_head(x, h, w_qkv)
            head_outputs.append(head_out)
            
        Y = np.concatenate(head_outputs, axis=-1)
        
        return np.dot(Y, W_o) + b_o
    
def test_mhattention_matrix_shapes():
    d_in = 768
    d_attn = 64
    d_mid = 64
    d_out = 768
    H = 12
    sequence_length = 5
    
    hp = Hyperparameters(d_in=d_in, d_attn=d_attn, d_mid=d_mid, d_out=d_out, H=H)
    W_q, W_k, W_v, b_q, b_k, b_v, W_o, b_o = hp.w
    
    assert W_q.shape == (H, d_in, d_attn)
    assert W_k.shape == (H, d_in, d_attn)
    assert W_v.shape == (H, d_in, d_mid)
    assert b_q.shape == (H, d_attn)
    assert b_k.shape == (H, d_attn)
    assert b_v.shape == (H, d_mid)
    assert W_o.shape == (H * d_mid, d_out)
    assert b_o.shape == (d_out,)
    
    mock_x = np.random.randn(sequence_length, d_in)
    output_mha = hp.mhattention(mock_x)
    
    assert output_mha.shape == (sequence_length, d_out)
    print("All matrix shape validations successfully passed structural checks.")

class TestMHAttentionValues(unittest.TestCase):
    def test_numerical_properties(self):
        d_in = 768
        d_attn = 64
        d_mid = 64
        d_out = 768
        H = 12
        sequence_length = 5
        
        hp = Hyperparameters(d_in=d_in, d_attn=d_attn, d_mid=d_mid, d_out=d_out, H=H)
        mock_x = np.random.randn(sequence_length, d_in)
        
        output_mha = hp.mhattention(mock_x)
        
        self.assertFalse(np.isnan(output_mha).any())
        self.assertFalse(np.isinf(output_mha).any())
        
        W_q, W_k, W_v, b_q, b_k, b_v, _, _ = hp.w
        w_qkv = (W_q, W_k, W_v, b_q, b_k, b_v)
        
        for h in range(H):
            Q = np.dot(mock_x, W_q[h]) + b_q[h]
            K = np.dot(mock_x, W_k[h]) + b_k[h]
            S = np.dot(Q, K.T) / np.sqrt(d_attn)
            
            exp_S = np.exp(S - np.max(S, axis=-1, keepdims=True))
            attn_weights = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
            
            row_sums = np.sum(attn_weights, axis=-1)
            np.testing.assert_allclose(row_sums, np.ones(sequence_length), rtol=1e-6)

if __name__ == "__main__":
    test_mhattention_matrix_shapes()
    unittest.main()
