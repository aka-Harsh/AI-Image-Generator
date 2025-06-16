"""
Enhanced AI Image Generator
Multi-model, multi-feature application with Web3 NFT support
"""

import os
import gradio as gr
import json
import datetime
import random
import io
from PIL import Image
import logging
import torch
# Import our services
from services.model_manager import get_model_manager
from services.nsfw_detector import get_nsfw_detector, check_prompt_safety
from services.storage_service import get_storage_service
from services.llm_service import get_llm_service, enhance_prompt
from services.nft_service import get_nft_service, mint_nft_demo
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
model_manager = None
nsfw_detector = None
storage_service = None
llm_service = None
nft_service = None

def initialize_services():
    """Initialize all services"""
    global model_manager, nsfw_detector, storage_service, llm_service, nft_service
    
    try:
        model_manager = get_model_manager()
        nsfw_detector = get_nsfw_detector()
        storage_service = get_storage_service()
        llm_service = get_llm_service()
        nft_service = get_nft_service()
        
        logger.info("All services initialized successfully")
        return "✅ All services initialized successfully"
    except Exception as e:
        error_msg = f"❌ Service initialization failed: {e}"
        logger.error(error_msg)
        return error_msg

def load_model_func(model_name):
    """Load selected model"""
    if not model_manager:
        return "❌ Model manager not initialized"
    
    try:
        success, message = model_manager.load_model(model_name)
        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    except Exception as e:
        return f"❌ Error loading model: {e}"

def check_nsfw_func(prompt):
    """Check if prompt is safe"""
    if not nsfw_detector:
        return "⚠️ NSFW detector not available", prompt
    
    try:
        is_safe, reason, details = check_prompt_safety(prompt)
        if is_safe:
            return f"✅ Prompt is safe: {reason}", prompt
        else:
            safe_alternative = nsfw_detector.suggest_safe_alternative(prompt)
            return f"❌ Unsafe prompt: {reason}", safe_alternative
    except Exception as e:
        return f"⚠️ NSFW check failed: {e}", prompt

def enhance_prompt_func(prompt, style="anime"):
    """Enhance prompt using LLM"""
    if not llm_service:
        return prompt, "⚠️ LLM service not available"
    
    try:
        result = enhance_prompt(prompt, style)
        if result["success"]:
            improvements = result.get("improvement_notes", [])
            status = f"✅ Enhanced! Improvements: {', '.join(improvements)}" if improvements else "✅ Prompt enhanced"
            return result["enhanced_prompt"], status
        else:
            return prompt, f"❌ Enhancement failed: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return prompt, f"❌ Enhancement error: {e}"

def generate_image_func(prompt, negative_prompt, model_name, width, height, steps, cfg_scale, seed, enable_img2img, init_image, img2img_strength, progress=gr.Progress()):
    """Main image generation function with progress tracking"""
    if not model_manager:
        return None, "❌ Model manager not initialized", "", ""
    
    try:
        # Update progress
        progress(0.1, desc="Initializing...")
        
        # Ensure model is loaded
        if model_manager.current_model_name != model_name:
            progress(0.2, desc="Loading model...")
            success, message = model_manager.load_model(model_name)
            if not success:
                return None, f"❌ Failed to load model: {message}", "", ""
        
        # Check prompt safety
        progress(0.3, desc="Checking prompt safety...")
        if nsfw_detector:
            is_safe, reason, _ = check_prompt_safety(prompt)
            if not is_safe:
                return None, f"❌ Unsafe prompt blocked: {reason}", "", ""
        
        # Progress tracking callback
        def progress_callback(step_progress, step_desc):
            # Map generation progress to 40-90% of total progress
            total_progress = 0.4 + (step_progress / 100) * 0.5
            progress(total_progress, desc=step_desc)
        
        # Generate image
        generation_start = datetime.datetime.now()
        progress(0.4, desc="Starting generation...")
        
        if enable_img2img and init_image is not None:
            # Image-to-image generation
            success, image, message = model_manager.generate_img2img(
                prompt=prompt,
                init_image=init_image,
                strength=img2img_strength,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                seed=seed
            )
        else:
            # Text-to-image generation
            success, image, message = model_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                seed=seed,
                progress_callback=progress_callback
            )
        
        if not success:
            return None, f"❌ Generation failed: {message}", "", ""
        
        progress(0.95, desc="Saving image...")
        generation_time = (datetime.datetime.now() - generation_start).total_seconds()
        
        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_name,
            "width": width if not enable_img2img else image.width,
            "height": height if not enable_img2img else image.height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": int(message.split("seed: ")[-1]) if "seed:" in message else seed,
            "generation_time": generation_time,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "img2img" if enable_img2img else "txt2img"
        }
        
        if enable_img2img:
            metadata["img2img_strength"] = img2img_strength
        
        # Save image (dual storage)
        if storage_service:
            storage_result = storage_service.save_image_dual(image, metadata)
            
            storage_status = []
            if storage_result["local_success"]:
                storage_status.append("✅ Local")
            else:
                storage_status.append("❌ Local")
            
            if storage_result["cloud_success"]:
                storage_status.append("✅ Cloud")
            else:
                storage_status.append("❌ Cloud")
            
            storage_info = f"Storage: {' | '.join(storage_status)}"
            if storage_result["errors"]:
                storage_info += f" (Errors: {', '.join(storage_result['errors'])})"
        else:
            storage_info = "⚠️ Storage service unavailable"
        
        progress(1.0, desc="Complete!")
        success_message = f"✅ {message}\n⏱️ Generated in {generation_time:.1f}s\n{storage_info}"
        
        return image, success_message, json.dumps(metadata, indent=2), storage_result.get("filename", "unknown.png")
        
    except Exception as e:
        error_msg = f"❌ Generation error: {e}"
        logger.error(error_msg)
        return None, error_msg, "", ""

def mint_nft_func(image, metadata_str, filename):
    """Mint NFT from generated image"""
    if not nft_service or not image:
        return "❌ NFT service not available or no image provided"
    
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_data = img_byte_arr.getvalue()
        
        # Parse metadata
        try:
            metadata = json.loads(metadata_str)
        except:
            metadata = {"prompt": "Unknown", "model": "Unknown"}
        
        # Mint NFT (demo)
        result = mint_nft_demo(image_data, metadata)
        
        if result["success"]:
            nft_info = f"""✅ NFT Minted Successfully! (Demo)

🎯 Token ID: {result['token_id']}
🔗 Transaction: {result['transaction_hash'][:20]}...
🌐 Network: {result['network']}
💰 Gas Cost: {result['estimated_gas']}

🖼️ OpenSea: {result['opensea_url']}
📊 Explorer: {result['etherscan_url']}

⚠️ Note: This is a demo implementation using testnet.
In production, this would create a real NFT on the blockchain."""
            return nft_info
        else:
            return f"❌ NFT minting failed: {result.get('error', 'Unknown error')}"
    
    except Exception as e:
        return f"❌ NFT minting error: {e}"

def get_random_seed():
    """Generate random seed"""
    return random.randint(0, 2**32 - 1)

def apply_preset(preset_name):
    """Apply prompt preset"""
    return Config.PROMPT_PRESETS.get(preset_name, "")

def refresh_gallery():
    """Refresh image gallery"""
    if storage_service:
        return storage_service.get_recent_images(8)
    return []

def get_system_status():
    """Get comprehensive system status"""
    status_parts = []
    
    # GPU Status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = torch.cuda.memory_reserved(0) / 1024**3
        status_parts.append(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        status_parts.append("💻 Device: CPU (slow generation)")
    
    # Model Manager Status
    if model_manager:
        model_info = model_manager.get_model_info()
        if model_info["loaded"]:
            status_parts.append(f"🤖 Model: {model_info['model_name']} ({model_info['device']})")
        else:
            status_parts.append("🤖 Model: Not loaded")
    else:
        status_parts.append("🤖 Model: Manager not initialized")
    
    # NSFW Detector Status
    if nsfw_detector and nsfw_detector.detector:
        status_parts.append("🛡️ NSFW Detection: Active")
    else:
        status_parts.append("🛡️ NSFW Detection: Inactive")
    
    # LLM Service Status
    if llm_service and llm_service.is_available:
        status_parts.append(f"🧠 LLM: {llm_service.model}")
    else:
        status_parts.append("🧠 LLM: Unavailable")
    
    # Storage Status
    if storage_service:
        storage_status = storage_service.get_storage_status()
        status_parts.append(f"💾 Storage: Local({'✅' if storage_status['local_available'] else '❌'}) | Cloud({'✅' if storage_status['s3_available'] else '❌'})")
    else:
        status_parts.append("💾 Storage: Unavailable")
    
    # NFT Service Status
    if nft_service:
        network_status = nft_service.get_network_status()
        status_parts.append(f"🔗 Blockchain: {network_status['network']} ({'✅' if network_status['connected'] else '❌'})")
    else:
        status_parts.append("🔗 Blockchain: Unavailable")
    
    return "\n".join(status_parts)

def create_interface():
    """Create the enhanced Gradio interface"""
    
    # Initialize services
    init_status = initialize_services()
    
    # Get available models
    available_models = list(Config.AVAILABLE_MODELS.keys()) if model_manager else ["No models available"]
    
    with gr.Blocks(title="Enhanced AI Image Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎨 Enhanced AI Image Generator")
        gr.Markdown("**Multi-Model | NSFW Detection | Prompt Enhancement | Cloud Storage | NFT Minting**")
        
        # System Status
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 System Status")
                system_status = gr.Markdown(get_system_status())
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
        
        # Model Management
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Model Management")
                model_selector = gr.Dropdown(
                    choices=available_models,
                    label="Select Model",
                    value=available_models[0] if available_models else None
                )
                load_model_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", value=init_status, interactive=False)
        
        gr.Markdown("---")
        
        # Main Interface
        with gr.Row():
            with gr.Column(scale=1):
                # Prompt Section
                gr.Markdown("### 🎨 Prompt & Settings")
                
                # Quick Presets
                preset_dropdown = gr.Dropdown(
                    choices=list(Config.PROMPT_PRESETS.keys()),
                    label="Quick Presets",
                    value=None
                )
                
                # Main prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt or select a preset above",
                    lines=3,
                    value="((best quality)), ((masterpiece)), (detailed), anime girl portrait"
                )
                
                # Prompt tools
                with gr.Row():
                    enhance_btn = gr.Button("🧠 Enhance Prompt", size="sm")
                    check_nsfw_btn = gr.Button("🛡️ Check Safety", size="sm")
                
                # Enhancement status
                enhancement_status = gr.Textbox(label="Prompt Status", interactive=False)
                
                # Negative prompt
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value=Config.DEFAULT_NEGATIVE_PROMPT
                )
                
                # Generation Settings
                gr.Markdown("### ⚙️ Generation Settings")
                
                with gr.Row():
                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height = gr.Slider(256, 1024, 512, step=64, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(5, 50, 25, step=1, label="Steps")
                    cfg_scale = gr.Slider(1, 20, 7.5, step=0.5, label="CFG Scale")
                
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    random_seed_btn = gr.Button("🎲", size="sm")
                
                # Image-to-Image Settings
                gr.Markdown("### 🖼️ Image-to-Image (Optional)")
                enable_img2img = gr.Checkbox(label="Enable Image-to-Image", value=False)
                init_image = gr.Image(label="Input Image", type="pil", visible=False)
                img2img_strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Strength", visible=False)
                
                # Generate Button
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output Section
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Generated Image", type="pil")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
                
                # Image Actions
                with gr.Row():
                    mint_nft_btn = gr.Button("🏆 Mint as NFT", variant="secondary")
                    refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", size="sm")
                
                # NFT Status
                nft_status = gr.Textbox(label="NFT Status", interactive=False, lines=8)
                
                # Gallery
                gallery = gr.Gallery(
                    label="Recent Generations",
                    columns=2,
                    rows=2,
                    height="300px"
                )
                
                # Metadata (Hidden)
                metadata_json = gr.Textbox(visible=False)
                current_filename = gr.Textbox(visible=False)
        
        # Event Handlers
        refresh_status_btn.click(get_system_status, outputs=system_status)
        load_model_btn.click(load_model_func, inputs=model_selector, outputs=model_status)
        
        preset_dropdown.change(apply_preset, inputs=preset_dropdown, outputs=prompt)
        random_seed_btn.click(lambda: get_random_seed(), outputs=seed)
        
        enhance_btn.click(
            enhance_prompt_func,
            inputs=[prompt, gr.State("anime")],
            outputs=[prompt, enhancement_status]
        )
        
        check_nsfw_btn.click(
            check_nsfw_func,
            inputs=prompt,
            outputs=[enhancement_status, prompt]
        )
        
        # Show/hide img2img controls
        enable_img2img.change(
            lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=enable_img2img,
            outputs=[init_image, img2img_strength]
        )
        
        # Main generation
        generate_btn.click(
            generate_image_func,
            inputs=[prompt, negative_prompt, model_selector, width, height, steps, cfg_scale, seed, 
                   enable_img2img, init_image, img2img_strength],
            outputs=[output_image, generation_status, metadata_json, current_filename]
        ).then(
            refresh_gallery,
            outputs=gallery
        )
        
        # NFT Minting
        mint_nft_btn.click(
            mint_nft_func,
            inputs=[output_image, metadata_json, current_filename],
            outputs=nft_status
        )
        
        refresh_gallery_btn.click(refresh_gallery, outputs=gallery)
        
        # Initial gallery load
        app.load(refresh_gallery, outputs=gallery)
        
        gr.Markdown("---")
        gr.Markdown("""
        ### 💡 Features & Tips
        - **Multi-Model**: Switch between ReV Animated (anime) and DreamShaper (realistic)
        - **Safety First**: Automatic NSFW detection protects against inappropriate content
        - **Smart Enhancement**: AI improves your prompts for better results
        - **Dual Storage**: Images saved locally and to AWS S3 cloud
        - **NFT Ready**: Mint your creations as NFTs on Polygon testnet
        - **Image-to-Image**: Upload reference images for style transfer
        
        🔧 **Setup Required**: Ensure Ollama is running and AWS credentials are configured
        """)
    
    return app

if __name__ == "__main__":
    # Import the validate_config function
    from config import validate_config
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("⚠️ Configuration Issues Found:")
        for error in errors:
            print(f"  - {error}")
        print("\nSome features may not work properly.")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting Enhanced AI Image Generator...")
    print("📁 Make sure your models are in the 'models/' folder")
    print("🤖 Ensure Ollama is running for prompt enhancement")
    print("☁️ Configure AWS credentials for cloud storage")
    
    # Create and launch interface
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )