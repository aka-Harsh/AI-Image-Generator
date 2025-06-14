import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os

def test_gpu():
    """Test GPU functionality"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        return "❌ No GPU detected"

def generate_gpu(prompt, steps=20):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"Loading model on {device}...")
        pipe = StableDiffusionPipeline.from_single_file(
            "./models/revAnimated_v2Rebirth.safetensors",
            torch_dtype=dtype,
            use_safetensors=True
        ).to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xformers enabled")
            except:
                print("xformers not available")
        
        print(f"Generating on {device}...")
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
        
        image = pipe(prompt, num_inference_steps=steps).images[0]
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            generation_time = "N/A"
        
        return image, f"✅ Generated on {device.upper()} in {generation_time:.1f}s" if device == "cuda" else f"✅ Generated on CPU"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 GPU Test Generator")
    
    with gr.Row():
        gpu_status = gr.Textbox(label="GPU Status", value=test_gpu())
    
    prompt = gr.Textbox(label="Prompt", value="beautiful anime girl, masterpiece")
    steps = gr.Slider(10, 30, 20, label="Steps")
    btn = gr.Button("Generate with GPU", variant="primary")
    
    image = gr.Image(label="Result")
    status = gr.Textbox(label="Status")
    
    btn.click(generate_gpu, [prompt, steps], [image, status])

demo.launch(server_name="127.0.0.1", server_port=7862, inbrowser=True)