import numpy as np 
import unittest

class Hyperparameters:
    def __init__(self, vocab_size : int = 50257, d_e : int = 768, seed : int = 42):
        np.random.seed(seed)
        
        W_e = np.random.randn(vocab_size, d_e) * 0.02
        
        self.w = (W_e,)
        
    def token_embedding(self, v : np.ndarray, w : tuple = ()) -> np.ndarray:
        if len(w) == 0:
            w = self.w
        
        W_e, = w
        
        return W_e[v]
    
def test_embedding_matrix_shapes():
    vocab_size = 50257
    d_e = 768
    sequence_length = 5
    
    hp = Hyperparameters(vocab_size = vocab_size, d_e = d_e)
    W_e, = hp.w
    
    assert W_e.shape == (vocab_size, d_e), f"Expected {(vocab_size, d_e)}, got {W_e.shape}"
    
    mock_v = np.random.randint(0, vocab_size, size=(sequence_length,))
    assert mock_v.shape == (sequence_length,)
    
    e = hp.token_embedding(mock_v)
    assert e.shape == (sequence_length, d_e), f"Output shape mismatch : {e.shape}"
    
    print("All matrix shape validations successfully passed structural checks.")

class TestEmbeddingValues(unittest.TestCase):
    def test_numerical_properties(self):
        vocab_size = 50257
        d_e = 768
        sequence_length = 5
        
        hp = Hyperparameters(vocab_size = vocab_size, d_e = d_e)
        mock_v = np.random.randint(0, vocab_size, size=(sequence_length,))
        
        e = hp.token_embedding(mock_v)
        
        self.assertFalse(np.isnan(e).any(), "Output matrix contains NaN values!")
        self.assertFalse(np.isinf(e).any(), "Output matrix contains Infinite values!")
        
        W_e, = hp.w
        for idx, token_id in enumerate(mock_v):
            np.testing.assert_allclose(
                e[idx], 
                W_e[token_id], 
                err_msg=f"Value error: Row {idx} does not exactly match token ID {token_id} embedding weights vector!"
            )

if __name__ == "__main__":
    test_embedding_matrix_shapes()
    unittest.main()
