import gradio as gr
import numpy as np
import requests
from PIL import Image
import io
import time
import re

# Define available models with descriptions
FLUX_MODELS = {
    "FLUX.1 Dev (General)": {
        "id": "black-forest-labs/FLUX.1-dev",
        "description": "General purpose image generation model 🎨"
    },
    "FLUX.1 Schnell (Fast)": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "description": "Optimized for faster generation ⚡"
    },
    "Outfit Generator": {
        "id": "tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator",
        "description": "Specialized in generating fashion outfits 👗"
    },
    "Logo Design": {
        "id": "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
        "description": "Professional logo design generation 🎯"
    },
    "Children's Sketches": {
        "id": "Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch",
        "description": "Simple children's style drawings ✏️"
    },
    "Anti-Blur": {
        "id": "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur",
        "description": "Enhanced clarity and sharpness 🔍"
    },
    "HDR Realism": {
        "id": "prithivMLmods/Flux.1-Dev-LoRA-HDR-Realism",
        "description": "High dynamic range realistic images 📸"
    },
    "Lehenga Design": {
        "id": "tryonlabs/FLUX.1-dev-LoRA-Lehenga-Generator",
        "description": "Indian traditional dress designs 👘"
    },
    "Modern Anime": {
        "id": "alfredplpl/flux.1-dev-modern-anime-lora",
        "description": "Modern anime style artwork 🎌"
    },
    "Anatomy": {
        "id": "rorito/testSCG-Anatomy-Flux1",
        "description": "Medical and anatomical illustrations 🔬"
    },
    "Live 3D": {
        "id": "Shakker-Labs/FLUX.1-dev-LoRA-live-3D",
        "description": "3D-style image generation 🎮"
    },
    "Japanese Style": {
        "id": "UmeAiRT/FLUX.1-dev-LoRA-Ume_J1900",
        "description": "Japanese art style generation 🗾"
    }
}

def enhance_prompt(prompt, model_name, enhancement_level):
    """Enhance the prompt with additional details and styling cues based on selected model"""
    # Basic prompt improvements
    enhancements = [
        "highly detailed",
        "professional",
        "masterpiece",
        "8k resolution",
        "stunning",
        "intricate details"
    ]
    
    # Model-specific enhancements
    model_specific = {
        "Logo Design": ["minimalist", "vectorial", "professional branding"],
        "HDR Realism": ["photorealistic", "high dynamic range", "sharp details"],
        "Modern Anime": ["anime style", "vibrant colors", "detailed shading"],
        "Live 3D": ["3D render", "volumetric lighting", "depth of field"],
        "Children's Sketches": ["simple lines", "cute", "playful"],
        "Outfit Generator": ["fashion forward", "trendy", "stylish"],
        "Japanese Style": ["traditional japanese", "elegant", "zen style"]
    }
    
    # Add model-specific enhancements
    if model_name in model_specific:
        for enhancement in model_specific[model_name][:enhancement_level]:
            if enhancement.lower() not in prompt.lower():
                prompt += f", {enhancement}"
    
    # Add general quality enhancements
    for enhancement in enhancements[:enhancement_level]:
        if enhancement.lower() not in prompt.lower():
            prompt += f", {enhancement}"
    
    # Clean up prompt
    prompt = re.sub(r'\s+', ' ', prompt)  # Remove extra spaces
    prompt = prompt.strip(' ,.') + '.'  # Ensure proper ending
    
    return prompt

def generate_image(prompt, model_name, enhance_enabled, enhancement_level, safety_enabled, adult_content, progress=gr.Progress()):
    try:
        progress(0, desc="Starting image generation... 🚀")
        
        # Enhance prompt if enabled
        if enhance_enabled:
            original_prompt = prompt
            prompt = enhance_prompt(prompt, model_name, enhancement_level)
            progress(0.2, desc=f"Enhanced prompt: {prompt} ✨")
        
        # Get model ID from selection
        model_id = FLUX_MODELS[model_name]["id"]
        
        # Define the API endpoint and headers
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {
            "Authorization": "Bearer hf-xxxxxxx",
            "Content-Type": "application/json",
        }
        
        progress(0.4, desc=f"Sending request to {model_name}... 📡")
        start_time = time.time()
        
        # Send the request with the prompt
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        
        if response.status_code != 200:
            return None, None, f"❌ Error: API returned status code {response.status_code}"
        
        progress(0.7, desc="Processing image... 🎨")
        
        # Convert the response content to an image
        image = Image.open(io.BytesIO(response.content))
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        progress(1.0, desc="Done! ✨")
        
        # Return the enhanced prompt if it was modified
        final_prompt = prompt if enhance_enabled else None
        return image, final_prompt, f"✅ Image generated successfully with {model_name} in {processing_time} seconds! ✨"
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

# Example prompts for inspiration
example_prompts = [
    ["A magical forest with glowing mushrooms and fairy lights"],
    ["A cyberpunk city at night with neon signs"],
    ["A cute robot playing with butterflies in a meadow"],
    ["A majestic dragon soaring through storm clouds"],
]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 AI Image Generation with FLUX.1 Models ✨")
    gr.Markdown("Choose a model and enter a prompt to generate amazing images! 🚀")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=list(FLUX_MODELS.keys()),
                value="FLUX.1 Dev (General)",
                label="🤖 Select Model",
                info="Choose the best model for your needs"
            )
            
            # Model description
            model_info = gr.Markdown()
            
            with gr.Column():
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="✍️ Original Prompt",
                        placeholder="Describe your imagination here...",
                        lines=3,
                        scale=1
                    )
                
                with gr.Row():
                    with gr.Column(scale=4):
                        model_dropdown = gr.Dropdown(
                            choices=list(FLUX_MODELS.keys()),
                            value="FLUX.1 Schnell (Fast)",
                            label="Model"
                        )
                    with gr.Column(scale=1):
                        enhance_checkbox = gr.Checkbox(
                            value=True,
                            label="Enhance Prompt",
                            info="Automatically enhance your prompt with additional details"
                        )
                
                with gr.Row():
                    enhancement_slider = gr.Slider(
                        label="Enhancement Level",
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        info="1=Minimal, 2=Moderate, 3=Detailed, 4=Very Detailed, 5=Maximum Detail"
                    )
                    
                with gr.Row():
                    enhanced_prompt_output = gr.Textbox(
                        label="Enhanced Prompt",
                        interactive=False,
                        lines=3
                    )
                    copy_btn = gr.Button("📋 Copy", scale=0)
            
            # Safety Controls
            with gr.Row():
                safety_checkbox = gr.Checkbox(
                    label="Enable Safety Filter",
                    value=True,
                    info="Filter out explicit content"
                )
                adult_checkbox = gr.Checkbox(
                    label="Allow Adult Content (18+)",
                    value=False,
                    info="Warning: May generate adult content"
                )
            
            # Add generate/clear buttons
            with gr.Row():
                generate_btn = gr.Button("🎨 Generate", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)
            
            # Add examples section
            gr.Examples(
                examples=example_prompts,
                inputs=prompt_input,
                label="📚 Example Prompts"
            )
        
        with gr.Column(scale=3):
            # Add image output and status
            image_output = gr.Image(label="🖼️ Generated Image")
            status_output = gr.Markdown("✨ Ready to generate! Enter a prompt and click Generate.")
    
    # Event handlers
    def clear_outputs():
        return None, "", "✨ Ready to generate! Enter a prompt and click Generate."
    
    def update_model_info(model_name):
        return f"ℹ️ **{model_name}**\n\n{FLUX_MODELS[model_name]['description']}"
    
    def copy_to_prompt(enhanced_prompt):
        return gr.update(value=enhanced_prompt) if enhanced_prompt else gr.update()
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, model_dropdown, enhance_checkbox, enhancement_slider, safety_checkbox, adult_checkbox],
        outputs=[image_output, enhanced_prompt_output, status_output],
    )
    
    clear_btn.click(
        fn=clear_outputs,
        outputs=[image_output, enhanced_prompt_output, status_output],
    )
    
    copy_btn.click(
        fn=copy_to_prompt,
        inputs=[enhanced_prompt_output],
        outputs=[prompt_input],
    )
    
    # Update model info when selection changes
    model_dropdown.change(
        fn=update_model_info,
        inputs=[model_dropdown],
        outputs=[model_info],
    )
    
    # Add prompt enhancement preview
    def preview_enhancement(prompt, model_name, enhance_enabled, enhancement_level):
        if not prompt or not enhance_enabled:
            return ""
        enhanced = enhance_prompt(prompt, model_name, enhancement_level)
        return enhanced if enhanced != prompt else prompt
    
    # Update enhanced prompt in real-time
    prompt_input.change(
        fn=preview_enhancement,
        inputs=[prompt_input, model_dropdown, enhance_checkbox, enhancement_slider],
        outputs=[enhanced_prompt_output],
    )
    
    # Also update when model or enhancement setting changes
    model_dropdown.change(
        fn=preview_enhancement,
        inputs=[prompt_input, model_dropdown, enhance_checkbox, enhancement_slider],
        outputs=[enhanced_prompt_output],
    )
    
    enhance_checkbox.change(
        fn=preview_enhancement,
        inputs=[prompt_input, model_dropdown, enhance_checkbox, enhancement_slider],
        outputs=[enhanced_prompt_output],
    )
    
    enhancement_slider.change(
        fn=preview_enhancement,
        inputs=[prompt_input, model_dropdown, enhance_checkbox, enhancement_slider],
        outputs=[enhanced_prompt_output],
    )

# Launch the interface
demo.launch(show_error=True)