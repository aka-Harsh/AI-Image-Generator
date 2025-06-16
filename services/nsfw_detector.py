"""
NSFW Content Detection Service
Detects inappropriate content in text prompts using transformers
"""

import re
from typing import Dict, List, Tuple
from detoxify import Detoxify
from config import Config
import logging

logger = logging.getLogger(__name__)

class NSFWDetector:
    def __init__(self):
        """Initialize NSFW detector with detoxify model"""
        self.detector = None
        self.explicit_keywords = [
            'nsfw', 'nude', 'naked', 'sex', 'porn', 'explicit', 'adult',
            'xxx', 'erotic', 'fetish', 'sexual', 'intimate', 'provocative',
            'suggestive', 'seductive', 'revealing', 'exposed', 'uncensored'
        ]
        self.load_detector()
    
    def load_detector(self):
        """Load the detoxify model for content detection"""
        try:
            self.detector = Detoxify('original')
            logger.info("NSFW detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NSFW detector: {e}")
            self.detector = None
    
    def check_explicit_keywords(self, text: str) -> Tuple[bool, List[str]]:
        """Check for explicit keywords in text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.explicit_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    def check_toxicity(self, text: str) -> Dict[str, float]:
        """Check text toxicity using detoxify"""
        if not self.detector:
            return {"error": "Detector not loaded"}
        
        try:
            results = self.detector.predict(text)
            return results
        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}")
            return {"error": str(e)}
    
    def is_safe_prompt(self, prompt: str) -> Tuple[bool, str, Dict]:
        """
        Main function to check if prompt is safe
        Returns: (is_safe, reason, details)
        """
        # Basic checks
        if not prompt or len(prompt.strip()) == 0:
            return False, "Empty prompt", {}
        
        # Check for explicit keywords
        has_keywords, found_keywords = self.check_explicit_keywords(prompt)
        if has_keywords:
            return False, f"Contains explicit keywords: {', '.join(found_keywords)}", {
                "keywords": found_keywords
            }
        
        # Check toxicity
        toxicity_results = self.check_toxicity(prompt)
        
        if "error" in toxicity_results:
            # If detector fails, allow but log warning
            logger.warning(f"NSFW detection failed: {toxicity_results['error']}")
            return True, "Detector unavailable, proceeding with caution", toxicity_results
        
        # Check various toxicity categories
        toxicity_checks = {
            'toxicity': Config.NSFW_THRESHOLD,
            'severe_toxicity': 0.5,
            'obscene': 0.6,
            'threat': 0.8,
            'insult': 0.8,
            'identity_attack': 0.8
        }
        
        failed_checks = []
        for category, threshold in toxicity_checks.items():
            if category in toxicity_results and toxicity_results[category] > threshold:
                failed_checks.append(f"{category}: {toxicity_results[category]:.2f}")
        
        if failed_checks:
            return False, f"High toxicity detected: {', '.join(failed_checks)}", toxicity_results
        
        return True, "Prompt is safe", toxicity_results
    
    def suggest_safe_alternative(self, unsafe_prompt: str) -> str:
        """Suggest a safer alternative for unsafe prompts"""
        # Remove explicit keywords
        safe_prompt = unsafe_prompt
        for keyword in self.explicit_keywords:
            safe_prompt = re.sub(r'\b' + re.escape(keyword) + r'\b', '', safe_prompt, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        safe_prompt = re.sub(r'\s+', ' ', safe_prompt).strip()
        
        # Add safety prefix if prompt is too short
        if len(safe_prompt) < 10:
            safe_prompt = "((best quality)), ((masterpiece)), safe and appropriate art"
        
        return safe_prompt
    
    def get_safety_report(self, prompt: str) -> Dict:
        """Get detailed safety report for a prompt"""
        is_safe, reason, details = self.is_safe_prompt(prompt)
        
        report = {
            "is_safe": is_safe,
            "reason": reason,
            "prompt": prompt,
            "safe_alternative": None if is_safe else self.suggest_safe_alternative(prompt),
            "toxicity_scores": details,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        return report

# Global detector instance
_detector = None

def get_nsfw_detector() -> NSFWDetector:
    """Get singleton NSFW detector instance"""
    global _detector
    if _detector is None:
        _detector = NSFWDetector()
    return _detector

def check_prompt_safety(prompt: str) -> Tuple[bool, str, Dict]:
    """Convenience function to check prompt safety"""
    detector = get_nsfw_detector()
    return detector.is_safe_prompt(prompt)

# Test function
if __name__ == "__main__":
    detector = NSFWDetector()
    
    test_prompts = [
        "beautiful anime girl portrait",  # Safe
        "anime warrior with sword",       # Safe
        "explicit adult content",         # Unsafe
        "nude photography",              # Unsafe
        "",                              # Invalid
    ]
    
    for prompt in test_prompts:
        is_safe, reason, details = detector.is_safe_prompt(prompt)
        print(f"Prompt: '{prompt}'")
        print(f"Safe: {is_safe} - {reason}")
        if not is_safe:
            print(f"Alternative: {detector.suggest_safe_alternative(prompt)}")
        print("-" * 50)