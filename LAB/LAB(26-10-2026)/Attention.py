import numpy as np 
import unittest

class Hyperparameters:
    def __init__(self, d_in : int = 768, d_attn : int = 64, d_out : int = 64, seed : int = 42):
        
        np.random.seed(seed)
        self.d_attn = d_attn
        
        W_q = np.random.randn(d_in, d_attn) * 0.02
        W_k = np.random.randn(d_in, d_attn) * 0.02
        W_v = np.random.randn(d_in, d_out) * 0.02
        
        b_q = np.zeros(d_attn)
        b_k = np.zeros(d_attn)
        b_v = np.zeros(d_out)
        
        self.w = (W_q, W_k, W_v, b_q, b_k, b_v)
        
        
    def attention(self, x : np.ndarray, w : tuple = ()) -> np.ndarray:
        
        if len(w) == 0:
            w = self.w
        
        W_q, W_k, W_v, b_q, b_k, b_v = w
        
        Q = np.dot(x, W_q) + b_q
        K = np.dot(x, W_k) + b_k
        V = np.dot(x, W_v) + b_v
        
        S = np.dot(Q, K.T) / np.sqrt(self.d_attn)
        
        exp_S = np.exp(S - np.max(S, axis = -1, keepdims = True))
        attention_weights = exp_S / np.sum(exp_S, axis = -1, keepdims = True)
        
        return np.dot(attention_weights, V)
    
def test_attention_matrix_shapes():
    d_in = 768
    d_attn = 64
    d_out= 64
    sequence_length  = 5
    
    hp = Hyperparameters(d_in = d_in, d_attn = d_attn, d_out = d_out)
    
    W_q, W_k, W_v, b_q, b_k, b_v = hp.w
    
    assert W_q.shape == (d_in, d_attn), f"Expected {(d_in, d_attn)}, got {W_q.shape}"
    assert W_k.shape == (d_in, d_attn), f"Expected {(d_in, d_attn)}, got {W_k.shape}"
    assert W_v.shape == (d_in, d_out), f"Expected {(d_in, d_out)}, got {W_v.shape}"
    assert b_q.shape == (d_attn, ), f"Expected {(d_attn)}, got {b_q.shape}"
    assert b_k.shape == (d_attn, ), f"Expected {(d_attn)}, got {b_k.shape}"
    assert b_v.shape == (d_out, ), f"Expected {(d_out)}, got {b_v.shape}"
    
    mock_x = np.random.randn(sequence_length, d_in)
    assert mock_x.shape == (sequence_length, d_in)
    
    Q = np.dot(mock_x, W_q) + b_q
    K = np.dot(mock_x, W_k) + b_k
    V = np.dot(mock_x, W_v) + b_v
    
    assert Q.shape == (sequence_length, d_attn), f"Q shape mismatch : {Q.shape}"
    assert K.shape == (sequence_length, d_attn), f"K shape mismatch : {K.shape}"
    assert V.shape == (sequence_length, d_out), f"V shape mismatch : {V.shape}"
    
    S = np.dot(Q, K.T) / np.sqrt(d_attn)
    assert S.shape == (sequence_length, sequence_length), f"S shape mismatch : {S.shape}"
    
    v_tilde = hp.attention(mock_x)
    assert v_tilde.shape == (sequence_length, d_out), f"Output shape mismatch : {v_tilde.shape}"
    
    print("All matrix shape validations successfully passes structural checks.")

class TestAttentionValues(unittest.TestCase):
    def test_numerical_properties(self):
        d_in, d_attn, d_out, sequence_length = 768, 64, 64, 5
        hp = Hyperparameters(d_in=d_in, d_attn=d_attn, d_out=d_out)
        mock_x = np.random.randn(sequence_length, d_in)
        
        v_tilde = hp.attention(mock_x)
        
        self.assertFalse(np.isnan(v_tilde).any(), "Output matrix contains NaN values!")
        self.assertFalse(np.isinf(v_tilde).any(), "Output matrix contains Infinite values!")
        
        W_q, W_k, _, b_q, b_k, _ = hp.w
        Q = np.dot(mock_x, W_q) + b_q
        K = np.dot(mock_x, W_k) + b_k
        S = np.dot(Q, K.T) / np.sqrt(d_attn)
        
        exp_S = np.exp(S - np.max(S, axis=-1, keepdims=True))
        attention_weights = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        
        row_sums = np.sum(attention_weights, axis=-1)
        
        np.testing.assert_allclose(
            row_sums, 
            np.ones(sequence_length), 
            rtol=1e-6, 
            err_msg="Softmax distribution rows do not sum to 1.0!"
        )
        
        self.assertTrue((attention_weights >= 0.0).all(), "Weights below 0.0!")
        self.assertTrue((attention_weights <= 1.0).all(), "Weights above 1.0!")


if __name__=="__main__":
    test_attention_matrix_shapes()
    unittest.main()