"""
LLM Service for Prompt Enhancement
Integrates with Ollama for local prompt improvement
"""

import requests
import json
import re
from typing import Dict, Optional, List
from config import Config
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initialize LLM service with Ollama"""
        self.host = Config.OLLAMA_HOST
        self.model = Config.OLLAMA_MODEL
        self.is_available = False
        self.check_availability()
    
    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.error("Ollama server not responding")
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                # Try to pull the model
                self.pull_model()
            
            self.is_available = True
            logger.info(f"LLM service ready with model: {self.model}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            self.is_available = False
            return False
        except Exception as e:
            logger.error(f"LLM service initialization error: {e}")
            self.is_available = False
            return False
    
    def pull_model(self) -> bool:
        """Pull the required model if not available"""
        try:
            logger.info(f"Pulling model: {self.model}")
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model: {self.model}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Generate response using Ollama"""
        if not self.is_available:
            logger.warning("LLM service not available")
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"LLM generation failed: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None
    
    def enhance_prompt(self, original_prompt: str, style: str = "anime") -> Dict[str, any]:
        """
        Enhance the user's prompt for better image generation
        """
        if not self.is_available:
            return {
                "enhanced_prompt": original_prompt,
                "success": False,
                "error": "LLM service not available"
            }
        
        # System prompt for enhancement
        system_prompt = f"""You are an expert AI image generation prompt engineer. Your job is to enhance prompts for {style} style image generation.

Rules:
1. Keep the original intent and main subject
2. Add quality modifiers like ((best quality)), ((masterpiece))
3. Add style-appropriate descriptive words
4. Add technical terms for better results
5. Keep it under 150 words
6. Don't add NSFW content
7. Return ONLY the enhanced prompt, no explanation"""

        # User prompt for enhancement
        enhancement_request = f"""Enhance this image generation prompt for {style} style:

Original: "{original_prompt}"

Enhanced prompt:"""

        try:
            enhanced = self.generate_response(enhancement_request, system_prompt)
            
            if enhanced:
                # Clean up the response
                enhanced = self.clean_enhanced_prompt(enhanced)
                
                return {
                    "enhanced_prompt": enhanced,
                    "original_prompt": original_prompt,
                    "success": True,
                    "improvement_notes": self.get_improvement_notes(original_prompt, enhanced)
                }
            else:
                return {
                    "enhanced_prompt": original_prompt,
                    "success": False,
                    "error": "Failed to generate enhancement"
                }
                
        except Exception as e:
            logger.error(f"Prompt enhancement error: {e}")
            return {
                "enhanced_prompt": original_prompt,
                "success": False,
                "error": str(e)
            }
    
    def clean_enhanced_prompt(self, prompt: str) -> str:
        """Clean and format the enhanced prompt"""
        # Remove common unwanted phrases
        unwanted_phrases = [
            "enhanced prompt:", "here's the enhanced prompt:", 
            "enhanced version:", "improved prompt:",
            "here is the", "here's the"
        ]
        
        cleaned = prompt
        for phrase in unwanted_phrases:
            cleaned = re.sub(f"^{re.escape(phrase)}", "", cleaned, flags=re.IGNORECASE)
        
        # Remove quotes if the entire prompt is quoted
        cleaned = cleaned.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def get_improvement_notes(self, original: str, enhanced: str) -> List[str]:
        """Analyze what improvements were made"""
        notes = []
        
        if "((best quality))" in enhanced and "((best quality))" not in original:
            notes.append("Added quality modifiers")
        
        if "detailed" in enhanced and "detailed" not in original:
            notes.append("Added detail descriptors")
        
        if len(enhanced) > len(original) * 1.5:
            notes.append("Expanded with descriptive terms")
        
        if any(word in enhanced for word in ["lighting", "atmosphere", "composition"]):
            notes.append("Added technical art terms")
        
        return notes
    
    def suggest_style_variations(self, prompt: str) -> List[str]:
        """Suggest different style variations of the prompt"""
        if not self.is_available:
            return []
        
        system_prompt = """You are an AI art style expert. Given a prompt, suggest 3 different artistic style variations. 

Rules:
1. Keep the main subject the same
2. Only change the style/artistic approach
3. Each suggestion should be 1-2 words max
4. Focus on art styles, lighting, or techniques
5. Return as a simple comma-separated list

Examples:
- realistic, anime, watercolor
- dramatic lighting, soft pastels, cyberpunk
- photorealistic, oil painting, sketch"""

        request = f"Suggest 3 artistic style variations for this prompt: '{prompt}'\n\nStyle suggestions:"
        
        try:
            suggestions = self.generate_response(request, system_prompt)
            if suggestions:
                # Parse comma-separated suggestions
                styles = [s.strip() for s in suggestions.split(',')]
                return styles[:3]  # Limit to 3
        except Exception as e:
            logger.error(f"Style suggestion error: {e}")
        
        return []
    
    def get_service_status(self) -> Dict[str, any]:
        """Get LLM service status information"""
        status = {
            "available": self.is_available,
            "host": self.host,
            "model": self.model,
            "connected": False,
            "models_available": []
        }
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                status["connected"] = True
                models = response.json().get('models', [])
                status["models_available"] = [m['name'] for m in models]
        except:
            pass
        
        return status

# Global LLM service instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get singleton LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

def enhance_prompt(prompt: str, style: str = "anime") -> Dict[str, any]:
    """Convenience function to enhance a prompt"""
    llm = get_llm_service()
    return llm.enhance_prompt(prompt, style)

# Test function
if __name__ == "__main__":
    llm = LLMService()
    
    test_prompts = [
        "anime girl",
        "warrior with sword",
        "fantasy landscape"
    ]
    
    print("LLM Service Test:")
    print(f"Status: {llm.get_service_status()}")
    print("-" * 50)
    
    for prompt in test_prompts:
        result = llm.enhance_prompt(prompt)
        print(f"Original: {prompt}")
        print(f"Enhanced: {result['enhanced_prompt']}")
        print(f"Success: {result['success']}")
        if result.get('improvement_notes'):
            print(f"Improvements: {', '.join(result['improvement_notes'])}")
        print("-" * 50)