import numpy as np 
import unittest

class Hyperparameters:
    def __init__(self, d_e : int = 768, seed : int = 42):
        np.random.seed(seed)
        
        gamma = np.ones(d_e)
        beta = np.zeros(d_e)
        
        self.w = (gamma, beta)
        
    def layer_norm(self, e : np.ndarray, w : tuple = ()) -> np.ndarray:
        if len(w) == 0:
            w = self.w
        
        gamma, beta = w
        
        m = np.mean(e, axis=-1, keepdims=True)
        v = np.var(e, axis=-1, keepdims=True)
        
        epsilon = 1e-5
        e_hat = (e - m) / np.sqrt(v + epsilon)
        
        return e_hat * gamma + beta
    
def test_layer_norm_matrix_shapes():
    d_e = 768
    sequence_length = 5
    
    hp = Hyperparameters(d_e = d_e)
    gamma, beta = hp.w
    
    assert gamma.shape == (d_e,), f"Expected {(d_e,)}, got {gamma.shape}"
    assert beta.shape == (d_e,), f"Expected {(d_e,)}, got {beta.shape}"
    
    mock_e = np.random.randn(sequence_length, d_e)
    assert mock_e.shape == (sequence_length, d_e)
    
    normalized_e = hp.layer_norm(mock_e)
    assert normalized_e.shape == (sequence_length, d_e), f"Output shape mismatch : {normalized_e.shape}"
    
    print("All matrix shape validations successfully passed structural checks.")

class TestLayerNormValues(unittest.TestCase):
    def test_numerical_properties(self):
        d_e = 768
        sequence_length = 5
        
        hp = Hyperparameters(d_e = d_e)
        mock_e = np.random.randn(sequence_length, d_e)
        
        v_raw = np.var(mock_e, axis=-1, keepdims=True)
        
        normalized_e = hp.layer_norm(mock_e)
        
        self.assertFalse(np.isnan(normalized_e).any(), "Output matrix contains NaN values!")
        self.assertFalse(np.isinf(normalized_e).any(), "Output matrix contains Infinite values!")
        
        expected_mean = np.zeros((sequence_length, 1))
        expected_var = v_raw / (v_raw + 1e-5)
        
        np.testing.assert_allclose(
            np.mean(normalized_e, axis=-1, keepdims=True),
            expected_mean,
            atol=1e-6,
            err_msg="Value error: Normalized outputs do not have a zero mean!"
        )
        
        np.testing.assert_allclose(
            np.var(normalized_e, axis=-1, keepdims=True),
            expected_var,
            atol=1e-6,
            err_msg="Value error: Normalized outputs do not have a unit variance!"
        )

if __name__ == "__main__":
    test_layer_norm_matrix_shapes()
    unittest.main()
