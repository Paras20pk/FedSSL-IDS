import pickle
import numpy as np
from typing import Dict, Tuple

class IDSIntegrator:
    def __init__(self):
        self.model_paths = {
            'phishing': r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\Phishing\phishing.pkl",         
            'ddos': r"C:\Users\Paras\Desktop\IDSFedSSL\CyberAttacks\DDOSAttack\ddos.pkl",                 
            # 'malware': 'models/malware_model.pkl',            
            # 'spoofing': 'models/spoofing_model.pkl',          
            # 'bruteforce': 'models/bruteforce_model.pkl',      
            # 'mitm': 'models/mitm_model.pkl'                  
        }
        self.models = self._load_models()
    
    def _load_models(self) -> Dict:
        """Load all models from PKL files"""
        models = {}
        for attack_type, path in self.model_paths.items():
            try:
                with open(path, 'rb') as f:
                    models[attack_type] = pickle.load(f)
                print(f" Loaded {attack_type} model from {path}")
            except Exception as e:
                print(f" Failed to load {attack_type} model: {str(e)}")
        return models
    
    def predict(self, input_data: np.ndarray) -> Tuple[str, float]:
        # Get predictions from all models
        results = {}
        for attack_type, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_data.reshape(1, -1))
                    confidence = np.max(proba)
                else:  # For models without predict_proba
                    pred = model.predict(input_data.reshape(1, -1))
                    confidence = 1.0 if pred[0] == 1 else 0.0
                results[attack_type] = confidence
            except Exception as e:
                print(f"Prediction failed for {attack_type}: {str(e)}")
                results[attack_type] = 0.0
        
        # Return attack with highest confidence
        best_attack = max(results.items(), key=lambda x: x[1])
        return best_attack[0], best_attack[1]

# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = IDSIntegrator()
    
    # Test prediction (replace with your actual preprocessed data)
    dummy_data = np.random.rand(100)  # Replace with real features
    attack_type, confidence = integrator.predict(dummy_data)
    print(f"\n Prediction: {attack_type} (Confidence: {confidence:.2%})")