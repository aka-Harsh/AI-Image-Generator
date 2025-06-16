"""
Image Service - Enhanced Image Generation Logic
Combines all services for complete image generation workflow
"""

import datetime
import json
import io
from typing import Dict, Tuple, Optional
from PIL import Image
import logging

from .model_manager import get_model_manager
from .nsfw_detector import get_nsfw_detector
from .storage_service import get_storage_service
from .llm_service import get_llm_service
from .nft_service import get_nft_service
from config import Config

logger = logging.getLogger(__name__)

class ImageGenerationService:
    def __init__(self):
        """Initialize the comprehensive image generation service"""
        self.model_manager = get_model_manager()
        self.nsfw_detector = get_nsfw_detector()
        self.storage_service = get_storage_service()
        self.llm_service = get_llm_service()
        self.nft_service = get_nft_service()
        
        self.generation_history = []
        logger.info("Image Generation Service initialized")
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, str, str]:
        """
        Validate prompt for safety and quality
        Returns: (is_valid, message, safe_prompt)
        """
        if not prompt or len(prompt.strip()) == 0:
            return False, "Empty prompt provided", ""
        
        # Check for safety
        try:
            is_safe, reason, details = self.nsfw_detector.is_safe_prompt(prompt)
            if not is_safe:
                safe_alternative = self.nsfw_detector.suggest_safe_alternative(prompt)
                return False, f"Unsafe content detected: {reason}", safe_alternative
            
            return True, "Prompt is safe", prompt
            
        except Exception as e:
            logger.warning(f"NSFW check failed: {e}")
            return True, "Safety check unavailable, proceeding with caution", prompt
    
    def enhance_prompt_quality(self, prompt: str, model_style: str = "anime") -> Tuple[str, str]:
        """
        Enhance prompt using LLM
        Returns: (enhanced_prompt, enhancement_message)
        """
        try:
            result = self.llm_service.enhance_prompt(prompt, model_style)
            if result["success"]:
                improvements = result.get("improvement_notes", [])
                message = f"Enhanced! Changes: {', '.join(improvements)}" if improvements else "Prompt enhanced"
                return result["enhanced_prompt"], message
            else:
                return prompt, f"Enhancement failed: {result.get('error', 'Unknown error')}"
        except Exception as e:
            logger.error(f"Prompt enhancement error: {e}")
            return prompt, f"Enhancement service unavailable: {e}"
    
    def prepare_generation_metadata(self, **kwargs) -> Dict:
        """Prepare comprehensive metadata for generation"""
        return {
            "prompt": kwargs.get("prompt", ""),
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "model": kwargs.get("model_name", "unknown"),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "steps": kwargs.get("steps", 25),
            "cfg_scale": kwargs.get("cfg_scale", 7.5),
            "seed": kwargs.get("seed", -1),
            "generation_mode": kwargs.get("mode", "txt2img"),
            "timestamp": datetime.datetime.now().isoformat(),
            "app_version": "1.0.0"
        }
    
    def generate_txt2img(self, 
                        prompt: str,
                        negative_prompt: str = "",
                        model_name: str = "ReV Animated",
                        width: int = 512,
                        height: int = 512,
                        steps: int = 25,
                        cfg_scale: float = 7.5,
                        seed: int = -1,
                        validate_safety: bool = True,
                        enhance_prompt: bool = False) -> Dict:
        """
        Complete text-to-image generation workflow
        """
        result = {
            "success": False,
            "image": None,
            "message": "",
            "metadata": {},
            "storage_info": {},
            "generation_time": 0
        }
        
        generation_start = datetime.datetime.now()
        
        try:
            # Step 1: Validate prompt safety
            if validate_safety:
                is_valid, safety_message, safe_prompt = self.validate_prompt(prompt)
                if not is_valid:
                    result["message"] = f"Safety check failed: {safety_message}"
                    result["suggested_prompt"] = safe_prompt
                    return result
                prompt = safe_prompt
            
            # Step 2: Enhance prompt (optional)
            if enhance_prompt and self.llm_service.is_available:
                model_style = self.model_manager.current_model.get("style", "anime") if self.model_manager.current_model else "anime"
                enhanced_prompt, enhancement_msg = self.enhance_prompt_quality(prompt, model_style)
                prompt = enhanced_prompt
                result["enhancement_message"] = enhancement_msg
            
            # Step 3: Ensure model is loaded
            if self.model_manager.current_model_name != model_name:
                success, load_message = self.model_manager.load_model(model_name)
                if not success:
                    result["message"] = f"Failed to load model: {load_message}"
                    return result
            
            # Step 4: Generate image
            success, image, gen_message = self.model_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                seed=seed
            )
            
            if not success:
                result["message"] = f"Generation failed: {gen_message}"
                return result
            
            # Extract seed from message
            actual_seed = seed
            if "seed:" in gen_message:
                try:
                    actual_seed = int(gen_message.split("seed: ")[-1])
                except:
                    pass
            
            # Step 5: Prepare metadata
            metadata = self.prepare_generation_metadata(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=actual_seed,
                mode="txt2img"
            )
            
            generation_time = (datetime.datetime.now() - generation_start).total_seconds()
            metadata["generation_time"] = generation_time
            
            # Step 6: Save image (dual storage)
            storage_result = self.storage_service.save_image_dual(image, metadata)
            
            # Step 7: Prepare successful result
            result.update({
                "success": True,
                "image": image,
                "message": f"✅ {gen_message}\n⏱️ Generated in {generation_time:.1f}s",
                "metadata": metadata,
                "storage_info": storage_result,
                "generation_time": generation_time,
                "actual_seed": actual_seed
            })
            
            # Add to history
            self.generation_history.append({
                "type": "txt2img",
                "timestamp": metadata["timestamp"],
                "prompt": prompt[:100],
                "model": model_name,
                "success": True
            })
            
            logger.info(f"Text-to-image generation completed in {generation_time:.1f}s")
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            result["message"] = error_msg
            logger.error(error_msg)
        
        return result
    
    def generate_img2img(self,
                        prompt: str,
                        init_image: Image.Image,
                        strength: float = 0.75,
                        negative_prompt: str = "",
                        model_name: str = "ReV Animated",
                        steps: int = 25,
                        cfg_scale: float = 7.5,
                        seed: int = -1,
                        validate_safety: bool = True,
                        enhance_prompt: bool = False) -> Dict:
        """
        Complete image-to-image generation workflow
        """
        result = {
            "success": False,
            "image": None,
            "message": "",
            "metadata": {},
            "storage_info": {},
            "generation_time": 0
        }
        
        generation_start = datetime.datetime.now()
        
        try:
            # Step 1: Validate inputs
            if init_image is None:
                result["message"] = "No input image provided"
                return result
            
            # Step 2: Validate prompt safety
            if validate_safety:
                is_valid, safety_message, safe_prompt = self.validate_prompt(prompt)
                if not is_valid:
                    result["message"] = f"Safety check failed: {safety_message}"
                    result["suggested_prompt"] = safe_prompt
                    return result
                prompt = safe_prompt
            
            # Step 3: Enhance prompt (optional)
            if enhance_prompt and self.llm_service.is_available:
                model_style = self.model_manager.current_model.get("style", "anime") if self.model_manager.current_model else "anime"
                enhanced_prompt, enhancement_msg = self.enhance_prompt_quality(prompt, model_style)
                prompt = enhanced_prompt
                result["enhancement_message"] = enhancement_msg
            
            # Step 4: Ensure model is loaded
            if self.model_manager.current_model_name != model_name:
                success, load_message = self.model_manager.load_model(model_name)
                if not success:
                    result["message"] = f"Failed to load model: {load_message}"
                    return result
            
            # Step 5: Generate image
            success, image, gen_message = self.model_manager.generate_img2img(
                prompt=prompt,
                init_image=init_image,
                strength=strength,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                seed=seed
            )
            
            if not success:
                result["message"] = f"Generation failed: {gen_message}"
                return result
            
            # Extract seed from message
            actual_seed = seed
            if "seed:" in gen_message:
                try:
                    actual_seed = int(gen_message.split("seed: ")[-1])
                except:
                    pass
            
            # Step 6: Prepare metadata
            metadata = self.prepare_generation_metadata(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                width=image.width,
                height=image.height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=actual_seed,
                mode="img2img"
            )
            metadata["img2img_strength"] = strength
            
            generation_time = (datetime.datetime.now() - generation_start).total_seconds()
            metadata["generation_time"] = generation_time
            
            # Step 7: Save image (dual storage)
            storage_result = self.storage_service.save_image_dual(image, metadata)
            
            # Step 8: Prepare successful result
            result.update({
                "success": True,
                "image": image,
                "message": f"✅ {gen_message}\n⏱️ Generated in {generation_time:.1f}s",
                "metadata": metadata,
                "storage_info": storage_result,
                "generation_time": generation_time,
                "actual_seed": actual_seed
            })
            
            # Add to history
            self.generation_history.append({
                "type": "img2img",
                "timestamp": metadata["timestamp"],
                "prompt": prompt[:100],
                "model": model_name,
                "success": True
            })
            
            logger.info(f"Image-to-image generation completed in {generation_time:.1f}s")
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            result["message"] = error_msg
            logger.error(error_msg)
        
        return result
    
    def mint_nft_from_result(self, generation_result: Dict) -> Dict:
        """
        Mint NFT from generation result
        """
        if not generation_result.get("success") or not generation_result.get("image"):
            return {"success": False, "error": "No valid image to mint"}
        
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            generation_result["image"].save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
            
            # Use metadata from generation
            metadata = generation_result.get("metadata", {})
            
            # Mint NFT (demo)
            nft_result = self.nft_service.simulate_mint_nft(image_data, metadata)
            
            if nft_result["success"]:
                logger.info(f"NFT minted successfully: Token ID {nft_result['token_id']}")
            
            return nft_result
            
        except Exception as e:
            error_msg = f"NFT minting error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_service_status(self) -> Dict:
        """Get comprehensive service status"""
        return {
            "model_manager": {
                "available": self.model_manager is not None,
                "current_model": self.model_manager.current_model_name if self.model_manager else None,
                "available_models": list(self.model_manager.get_available_models().keys()) if self.model_manager else []
            },
            "nsfw_detector": {
                "available": self.nsfw_detector is not None and self.nsfw_detector.detector is not None
            },
            "storage_service": {
                "available": self.storage_service is not None,
                "local_storage": self.storage_service.get_storage_status() if self.storage_service else {}
            },
            "llm_service": {
                "available": self.llm_service is not None and self.llm_service.is_available,
                "model": self.llm_service.model if self.llm_service else None
            },
            "nft_service": {
                "available": self.nft_service is not None,
                "network_status": self.nft_service.get_network_status() if self.nft_service else {}
            },
            "generation_history": {
                "total_generations": len(self.generation_history),
                "recent": self.generation_history[-5:] if self.generation_history else []
            }
        }

# Global service instance
_image_service = None

def get_image_service() -> ImageGenerationService:
    """Get singleton image service instance"""
    global _image_service
    if _image_service is None:
        _image_service = ImageGenerationService()
    return _image_service

# Test function
if __name__ == "__main__":
    service = ImageGenerationService()
    
    print("Image Generation Service Test:")
    status = service.get_service_status()
    
    for service_name, service_status in status.items():
        print(f"{service_name}: {service_status}")
    
    # Test prompt validation
    test_prompts = [
        "beautiful anime girl",
        "explicit content",
        ""
    ]
    
    for prompt in test_prompts:
        is_valid, message, safe_prompt = service.validate_prompt(prompt)
        print(f"Prompt: '{prompt}' -> Valid: {is_valid}, Message: {message}")