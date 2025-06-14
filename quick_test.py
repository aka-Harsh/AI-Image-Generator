import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import os

def generate_quick(prompt, steps=10):
    try:
        print(f"Loading model...")
        pipe = StableDiffusionPipeline.from_single_file(
            "./models/revAnimated_v2Rebirth.safetensors",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        print(f"Generating with {steps} steps...")
        
        # Quick generation with progress
        def progress_callback(step, timestep, latents):
            print(f"Step {step+1}/{steps}")
        
        image = pipe(
            prompt, 
            num_inference_steps=steps,
            callback=progress_callback,
            callback_steps=1
        ).images[0]
        
        return image, f"✅ Generated in {steps} steps!"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Quick Test Generator")
    
    prompt = gr.Textbox(label="Prompt", value="beautiful anime girl")
    steps = gr.Slider(5, 30, 10, label="Steps (lower = faster)")
    btn = gr.Button("Generate", variant="primary")
    
    image = gr.Image(label="Result")
    status = gr.Textbox(label="Status")
    
    btn.click(generate_quick, [prompt, steps], [image, status])

demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)