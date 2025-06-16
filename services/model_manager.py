"""
Model Manager Service
Handles loading and switching between multiple Stable Diffusion models
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from typing import Dict, Optional, Tuple
import os
import gc
from config import Config
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        """Initialize model manager"""
        self.current_model = None
        self.current_model_name = None
        self.current_pipeline = None
        self.img2img_pipeline = None
        
        # Detect best available device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16  # Use half precision for GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.device = "cpu"
            self.dtype = torch.float32  # Use full precision for CPU
            logger.info("No GPU detected, using CPU")
        
        logger.info(f"Model Manager initialized - Device: {self.device}, dtype: {self.dtype}")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available models with their info"""
        available = {}
        
        for model_name, model_info in Config.AVAILABLE_MODELS.items():
            model_path = model_info["path"]
            if os.path.exists(model_path):
                available[model_name] = {
                    **model_info,
                    "file_size": self.get_file_size(model_path),
                    "available": True
                }
            else:
                available[model_name] = {
                    **model_info,
                    "available": False,
                    "error": f"File not found: {model_path}"
                }
        
        return available
    
    def get_file_size(self, file_path: str) -> str:
        """Get human-readable file size"""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024**3:
                return f"{size_bytes / (1024**2):.1f} MB"
            else:
                return f"{size_bytes / (1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def clear_current_model(self):
        """Clear currently loaded model from memory"""
        if self.current_pipeline:
            del self.current_pipeline
            self.current_pipeline = None
        
        if self.img2img_pipeline:
            del self.img2img_pipeline
            self.img2img_pipeline = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.current_model = None
        self.current_model_name = None
        logger.info("Cleared model from memory")
    
    def load_model(self, model_name: str) -> Tuple[bool, str]:
        """Load a specific model"""
        if model_name == self.current_model_name:
            return True, f"Model {model_name} already loaded"
        
        # Check if model exists
        available_models = self.get_available_models()
        if model_name not in available_models:
            return False, f"Model {model_name} not found"
        
        model_info = available_models[model_name]
        if not model_info["available"]:
            return False, model_info.get("error", "Model not available")
        
        try:
            # Clear previous model
            self.clear_current_model()
            
            logger.info(f"Loading model: {model_name}")
            model_path = model_info["path"]
            
            # Load text-to-image pipeline
            self.current_pipeline = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None,  # Disable safety checker for speed
                requires_safety_checker=False
            )
            
            self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Optimize pipeline
            self.current_pipeline.enable_attention_slicing()
            
            if self.device == "cuda":
                try:
                    # Enable GPU optimizations
                    self.current_pipeline.enable_model_cpu_offload()  # Smart memory management
                    logger.info("GPU memory optimization enabled")
                except Exception as e:
                    logger.warning(f"Could not enable CPU offload: {e}")
                
                try:
                    # Enable xformers if available
                    self.current_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
                    
                # Clear GPU cache
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            
            # Create img2img pipeline using from_pipe method (more reliable)
            try:
                self.img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pipe(self.current_pipeline)
                logger.info("Img2img pipeline created using from_pipe method")
            except AttributeError:
                # Fallback for older diffusers versions
                try:
                    self.img2img_pipeline = StableDiffusionImg2ImgPipeline(
                        vae=self.current_pipeline.vae,
                        text_encoder=self.current_pipeline.text_encoder,
                        tokenizer=self.current_pipeline.tokenizer,
                        unet=self.current_pipeline.unet,
                        scheduler=self.current_pipeline.scheduler,
                        safety_checker=getattr(self.current_pipeline, 'safety_checker', None),
                        requires_safety_checker=False,
                        feature_extractor=getattr(self.current_pipeline, 'feature_extractor', None)
                    )
                    logger.info("Img2img pipeline created using component initialization")
                except Exception as e:
                    logger.warning(f"Could not create img2img pipeline: {e}")
                    self.img2img_pipeline = None
            
            self.current_model_name = model_name
            self.current_model = model_info
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True, f"Model {model_name} loaded successfully"
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            self.clear_current_model()
            return False, error_msg
    
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: str = "",
                      width: int = 512,
                      height: int = 512,
                      num_inference_steps: int = 25,
                      guidance_scale: float = 7.5,
                      seed: int = -1,
                      progress_callback=None) -> Tuple[bool, any, str]:
        """Generate image using current model"""
        
        if not self.current_pipeline:
            return False, None, "No model loaded"
        
        try:
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Create progress callback if provided
            def callback_func(step, timestep, latents):
                if progress_callback:
                    progress = (step + 1) / num_inference_steps * 100
                    progress_callback(progress, f"Step {step+1}/{num_inference_steps}")
            
            # Generate image
            logger.info(f"Starting generation: {num_inference_steps} steps on {self.device}")
            
            with torch.autocast(self.device):
                result = self.current_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback=callback_func,
                    callback_steps=1
                )
            
            image = result.images[0]
            logger.info("Generation completed successfully")
            return True, image, f"Generated successfully with seed: {seed}"
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def generate_img2img(self,
                        prompt: str,
                        init_image,
                        strength: float = 0.75,
                        negative_prompt: str = "",
                        num_inference_steps: int = 25,
                        guidance_scale: float = 7.5,
                        seed: int = -1) -> Tuple[bool, any, str]:
        """Generate image-to-image using current model"""
        
        if not self.img2img_pipeline:
            return False, None, "No model loaded or img2img not available"
        
        try:
            # Set seed
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate image
            with torch.autocast(self.device):
                result = self.img2img_pipeline(
                    prompt=prompt,
                    image=init_image,
                    strength=strength,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
            
            image = result.images[0]
            return True, image, f"Img2img generated successfully with seed: {seed}"
            
        except Exception as e:
            error_msg = f"Img2img generation failed: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about currently loaded model"""
        if not self.current_model_name:
            return {"loaded": False, "message": "No model loaded"}
        
        return {
            "loaded": True,
            "model_name": self.current_model_name,
            "model_info": self.current_model,
            "device": self.device,
            "dtype": str(self.dtype),
            "txt2img_available": self.current_pipeline is not None,
            "img2img_available": self.img2img_pipeline is not None
        }
    
    def get_recommended_settings(self, model_name: str = None) -> Dict[str, any]:
        """Get recommended settings for current or specified model"""
        if model_name is None:
            model_name = self.current_model_name
        
        if not model_name:
            return {}
        
        model_info = Config.AVAILABLE_MODELS.get(model_name, {})
        
        base_settings = {
            "width": 512,
            "height": 512,
            "steps": 25,
            "cfg_scale": 7.5
        }
        
        # Model-specific recommendations
        if model_info.get("style") == "anime":
            base_settings.update({
                "steps": 28,
                "cfg_scale": 8.0,
                "recommended_prompts": [
                    "((best quality)), ((masterpiece)), (detailed)",
                    "anime style, beautiful, detailed face",
                    "high quality, sharp focus"
                ]
            })
        elif model_info.get("style") == "realistic":
            base_settings.update({
                "steps": 30,
                "cfg_scale": 7.0,
                "recommended_prompts": [
                    "photorealistic, high quality",
                    "detailed, sharp focus, professional",
                    "masterpiece, best quality"
                ]
            })
        
        # Add resolution recommendations
        if "recommended_resolution" in model_info:
            width, height = model_info["recommended_resolution"]
            base_settings.update({"width": width, "height": height})
        
        return base_settings

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get singleton model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

# Test function
if __name__ == "__main__":
    manager = ModelManager()
    
    print("Available Models:")
    available = manager.get_available_models()
    for name, info in available.items():
        print(f"- {name}: {info}")
    
    print(f"\nCurrent Model Info: {manager.get_model_info()}")
    
    # Test loading first available model
    for model_name in available:
        if available[model_name]["available"]:
            success, message = manager.load_model(model_name)
            print(f"\nLoad {model_name}: {success} - {message}")
            if success:
                print(f"Model Info: {manager.get_model_info()}")
                print(f"Recommended Settings: {manager.get_recommended_settings()}")
            break