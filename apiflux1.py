import gradio as gr
import numpy as np
import requests
from PIL import Image
import io
import time
import re
import asyncio
import aiohttp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import Optional, Tuple, Dict, List
import hashlib
from dataclasses import dataclass
from queue import PriorityQueue
import concurrent.futures
from collections import deque
import random

# API Token Management
class TokenManager:
    def __init__(self):
        self.tokens = [
            "theres hf token xxxxx",
            "theres hf token xxxxx",
            "theres hf token xxxxx",
        ]
        self.token_usage = {token: {'last_used': 0, 'requests': 0} for token in self.tokens}
        self.lock = threading.Lock()
        self.current_token_index = 0
    
    def get_next_token(self) -> str:
        """Get the next available token using round-robin with rate limiting"""
        with self.lock:
            current_time = time.time()
            
            # Try each token in order
            for _ in range(len(self.tokens)):
                token = self.tokens[self.current_token_index]
                token_info = self.token_usage[token]
                
                # Reset request count if more than 60 seconds have passed
                if current_time - token_info['last_used'] > 60:
                    token_info['requests'] = 0
                
                # Use token if under rate limit
                if token_info['requests'] < 3:
                    token_info['last_used'] = current_time
                    token_info['requests'] += 1
                    self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
                    return token
                
                # Try next token
                self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
            
            # If all tokens are rate limited, find the one closest to reset
            oldest_used = min(info['last_used'] for info in self.token_usage.values())
            wait_time = 60 - (current_time - oldest_used)
            if wait_time > 0:
                time.sleep(min(wait_time, 5))  # Wait at most 5 seconds
            
            # Reset token with oldest usage
            for token, info in self.token_usage.items():
                if info['last_used'] == oldest_used:
                    info['requests'] = 1
                    info['last_used'] = time.time()
                    return token
    
    def mark_token_error(self, token: str):
        """Mark a token as having an error, temporarily increasing its request count"""
        with self.lock:
            if token in self.token_usage:
                self.token_usage[token]['requests'] = 3  # Max out requests to force rotation

token_manager = TokenManager()

@dataclass
class ModelConfig:
    id: str
    name: str
    description: str
    timeout: float = 30.0
    priority: int = 1
    safety_checker: bool = False
    adult_content: bool = True
    # Quality settings
    image_size: tuple = (1024, 1024)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    scheduler: str = "DPMSolverMultistep"
    num_images_per_prompt: int = 1
    strength: float = 0.75
    seed: int = -1  # -1 for random

FLUX_MODELS = {
    "FLUX.1 Schnell (Fast)": ModelConfig(
        id="stabilityai/stable-diffusion-xl-base-1.0",
        name="FLUX.1 Schnell",
        description="Fast general-purpose model",
        timeout=20.0,
        priority=1,
        image_size=(1024, 1024),
        num_inference_steps=30,
        guidance_scale=7.0,
        scheduler="DPMSolverMultistep"
    ),
    "FLUX.1 Dev (General)": ModelConfig(
        id="runwayml/stable-diffusion-v1-5",
        name="FLUX.1 Dev",
        description="High-quality general purpose model",
        timeout=30.0,
        priority=2,
        image_size=(1024, 1024),
        num_inference_steps=50,
        guidance_scale=7.5,
        scheduler="DPMSolverMultistep"
    ),
    "Outfit Generator": ModelConfig(
        id="tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator",
        name="Outfit Generator",
        description="Fashion outfit generation",
        image_size=(1024, 1024),
        num_inference_steps=45,
        guidance_scale=8.0,
        negative_prompt="low quality, worst quality, bad anatomy, bad hands, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad quality, deformed"
    ),
    "Logo Design": ModelConfig(
        id="Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
        name="Logo Design",
        description="Professional logo design",
        image_size=(1024, 1024),
        num_inference_steps=40,
        guidance_scale=7.0,
        negative_prompt="blurry, bad text, low quality"
    ),
    "Children's Sketches": ModelConfig(
        id="Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch",
        name="Children's Sketches",
        description="Simple children's drawings",
        image_size=(1024, 1024),
        num_inference_steps=35,
        guidance_scale=6.5,
        negative_prompt="complex, detailed, photorealistic"
    ),
    "Anti-Blur": ModelConfig(
        id="Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur",
        name="Anti-Blur",
        description="Ultra-sharp image generation",
        image_size=(1024, 1024),
        num_inference_steps=50,
        guidance_scale=8.5,
        negative_prompt="blur, blurry, low quality, worst quality"
    ),
    "HDR Realism": ModelConfig(
        id="prithivMLmods/Flux.1-Dev-LoRA-HDR-Realism",
        name="HDR Realism",
        description="Photorealistic HDR images",
        image_size=(1024, 1024),
        num_inference_steps=50,
        guidance_scale=9.0,
        negative_prompt="cartoon, anime, illustration, painting, drawing, low quality"
    ),
    "Lehenga Design": ModelConfig(
        id="tryonlabs/FLUX.1-dev-LoRA-Lehenga-Generator",
        name="Lehenga Design",
        description="Indian traditional dress designs",
        image_size=(1024, 1024),
        num_inference_steps=45,
        guidance_scale=8.0,
        negative_prompt="low quality, worst quality, bad fabric, bad design"
    ),
    "Modern Anime": ModelConfig(
        id="alfredplpl/flux.1-dev-modern-anime-lora",
        name="Modern Anime",
        description="Modern anime style art",
        image_size=(1024, 1024),
        num_inference_steps=40,
        guidance_scale=7.0,
        negative_prompt="realistic, photo, photograph, 3d, low quality"
    ),
    "Anatomy": ModelConfig(
        id="rorito/testSCG-Anatomy-Flux1",
        name="Anatomy",
        description="Medical illustrations",
        image_size=(1024, 1024),
        num_inference_steps=50,
        guidance_scale=8.0,
        negative_prompt="artistic, stylized, cartoon, low quality, inaccurate"
    ),
    "Live 3D": ModelConfig(
        id="Shakker-Labs/FLUX.1-dev-LoRA-live-3D",
        name="Live 3D",
        description="3D-style rendering",
        image_size=(1024, 1024),
        num_inference_steps=45,
        guidance_scale=8.0,
        negative_prompt="2d, flat, illustration, drawing, low quality"
    ),
    "Japanese Style": ModelConfig(
        id="UmeAiRT/FLUX.1-dev-LoRA-Ume_J1900",
        name="Japanese Style",
        description="Traditional Japanese art",
        image_size=(1024, 1024),
        num_inference_steps=45,
        guidance_scale=7.5,
        negative_prompt="modern, western, contemporary, low quality"
    )
}

class RequestQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.processing = set()
        self.lock = threading.Lock()
    
    def add_request(self, priority: int, model_name: str, prompt: str):
        with self.lock:
            self.queue.put((priority, (model_name, prompt)))
    
    def get_next_request(self) -> Optional[Tuple[str, str]]:
        with self.lock:
            if not self.queue.empty():
                _, request = self.queue.get()
                self.processing.add(request)
                return request
            return None
    
    def complete_request(self, request: Tuple[str, str]):
        with self.lock:
            self.processing.discard(request)

request_queue = RequestQueue()

async def process_request_queue(session: aiohttp.ClientSession):
    while True:
        request = request_queue.get_next_request()
        if request:
            model_name, prompt = request
            try:
                await generate_image_async(prompt, model_name, session)
            finally:
                request_queue.complete_request(request)
        await asyncio.sleep(0.1)

async def generate_with_fallback(prompt: str, model_name: str, session: aiohttp.ClientSession, tried_models: set = None, safety_checker: bool = True, adult_content: bool = False) -> Tuple[Optional[Image.Image], float, str]:
    """Try generating with fallback models if primary fails"""
    if tried_models is None:
        tried_models = set()
    
    if model_name in tried_models:
        return None, 0, "All fallback models failed"
    
    tried_models.add(model_name)
    model_config = FLUX_MODELS[model_name]
    
    # Try primary model
    image, gen_time, status = await generate_image_async(prompt, model_name, session, safety_checker, adult_content)
    if image is not None:
        return image, gen_time, status
    
    # Try fallback models
    for fallback_model in model_config.fallback_models:
        if fallback_model not in tried_models:
            image, gen_time, status = await generate_with_fallback(prompt, fallback_model, session, tried_models, safety_checker, adult_content)
            if image is not None:
                return image, gen_time, f"Generated with fallback model {fallback_model}"
    
    return None, 0, "All models failed"

async def generate_image_async(prompt: str, model_name: str, session: aiohttp.ClientSession, safety_checker: bool = False, adult_content: bool = True) -> Tuple[Optional[Image.Image], float, str]:
    """Asynchronously generate image using the API with token rotation and quality settings"""
    model_config = FLUX_MODELS[model_name]
    start_time = time.time()
    
    # Add quality-focused negative prompt
    base_negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad quality, deformed"
    
    # Try up to 3 different tokens
    for attempt in range(3):
        token = token_manager.get_next_token()
        url = f"https://api-inference.huggingface.co/models/{model_config.id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        # Quality-focused parameters
        params = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": base_negative_prompt + ", " + model_config.negative_prompt,
                "num_inference_steps": 50,  # Higher steps for better quality
                "guidance_scale": 7.5,  # Balanced guidance
                "width": 1024,  # High resolution
                "height": 1024,
                "eta": 0.0,  # Reduces noise
                "denoising_strength": 1.0,  # Full denoising
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }
        
        try:
            async with session.post(url, headers=headers, json=params, timeout=model_config.timeout) as response:
                if response.status == 503:
                    return None, time.time() - start_time, "Model is loading, please try again in a few moments"
                
                response_data = await response.read()
                
                if response.status != 200:
                    try:
                        error_msg = response_data.decode('utf-8')
                        if "Max requests" in error_msg:
                            token_manager.mark_token_error(token)
                            if attempt < 2:
                                continue
                        return None, time.time() - start_time, f"API Error: {error_msg}"
                    except:
                        return None, time.time() - start_time, f"API Error: Status {response.status}"
                
                try:
                    image = Image.open(io.BytesIO(response_data))
                    return image, time.time() - start_time, "success"
                except Exception as e:
                    return None, time.time() - start_time, f"Invalid image data: {str(e)}"
                    
        except asyncio.TimeoutError:
            if attempt < 2:
                continue
            return None, time.time() - start_time, f"Request timed out after {model_config.timeout} seconds"
        except aiohttp.ClientError as e:
            if attempt < 2:
                continue
            return None, time.time() - start_time, f"Network error: {str(e)}"
        except Exception as e:
            if attempt < 2:
                continue
            return None, time.time() - start_time, f"Unexpected error: {str(e)}"
    
    return None, time.time() - start_time, "All API tokens are rate limited. Please wait a moment and try again."

@lru_cache(maxsize=100)
def enhance_prompt(prompt: str, model_name: str, enhancement_level: int = 2) -> str:
    """
    Enhance the prompt with additional details based on the model and desired enhancement level
    enhancement_level: 1=minimal, 2=moderate, 3=detailed, 4=very detailed, 5=maximum detail
    """
    model_config = FLUX_MODELS[model_name]
    
    # Base quality terms for all enhancement levels
    quality_terms = {
        1: ["high quality", "detailed"],
        2: ["high quality", "detailed", "sharp focus", "intricate"],
        3: ["masterpiece", "high quality", "detailed", "sharp focus", "intricate", "professional", "artstation"],
        4: ["masterpiece", "high quality", "detailed", "sharp focus", "intricate", "professional", "artstation", "4k", "8k", "HDR"],
        5: ["masterpiece", "best quality", "ultra detailed", "sharp focus", "intricate", "professional", "artstation", "4k", "8k", "HDR", "ray tracing", "studio quality"]
    }
    
    # Model-specific enhancements
    model_specific = {
        "FLUX.1 Schnell (Fast)": {
            1: ["vibrant"],
            2: ["vibrant", "dynamic"],
            3: ["vibrant", "dynamic", "cinematic"],
            4: ["vibrant", "dynamic", "cinematic", "dramatic lighting"],
            5: ["vibrant", "dynamic", "cinematic", "dramatic lighting", "volumetric lighting", "ray traced"]
        },
        "HDR Realism": {
            1: ["photorealistic"],
            2: ["photorealistic", "hyperrealistic"],
            3: ["photorealistic", "hyperrealistic", "octane render"],
            4: ["photorealistic", "hyperrealistic", "octane render", "8k uhd"],
            5: ["photorealistic", "hyperrealistic", "octane render", "8k uhd", "volumetric lighting", "ray traced"]
        },
        "Modern Anime": {
            1: ["anime style"],
            2: ["detailed anime style", "high quality anime"],
            3: ["detailed anime style", "high quality anime", "studio ghibli"],
            4: ["detailed anime style", "high quality anime", "studio ghibli", "anime key visual"],
            5: ["detailed anime style", "high quality anime", "studio ghibli", "anime key visual", "professional anime artwork"]
        },
        "Logo Design": {
            1: ["professional logo"],
            2: ["professional logo", "minimalist"],
            3: ["professional logo", "minimalist", "vector art"],
            4: ["professional logo", "minimalist", "vector art", "corporate branding"],
            5: ["professional logo", "minimalist", "vector art", "corporate branding", "award winning design"]
        }
    }
    
    # Style-specific details
    style_details = {
        1: [],  # Minimal style details
        2: ["detailed texture"],  # Basic style details
        3: ["detailed texture", "professional lighting"],  # Moderate details
        4: ["detailed texture", "professional lighting", "ambient occlusion"],  # High details
        5: ["detailed texture", "professional lighting", "ambient occlusion", "subsurface scattering", "global illumination"]  # Maximum details
    }
    
    # Get base quality terms
    enhanced = quality_terms[enhancement_level]
    
    # Add model-specific terms if available
    if model_name in model_specific:
        enhanced.extend(model_specific[model_name][enhancement_level])
    
    # Add style details based on level
    enhanced.extend(style_details[enhancement_level])
    
    # Add base prompt
    enhanced = [prompt] + enhanced
    
    # Add composition terms for higher levels
    if enhancement_level >= 3:
        enhanced.extend(["perfect composition", "award winning"])
    if enhancement_level >= 4:
        enhanced.extend(["masterpiece", "trending on artstation"])
    if enhancement_level == 5:
        enhanced.extend(["perfect lighting", "perfect shadows", "perfect reflections"])
    
    # Join all terms
    return ", ".join(enhanced)

def generate_image(prompt: str, model_name: str, enhance_enabled: bool = True, enhancement_level: int = 2, safety_checker: bool = False, adult_content: bool = True, progress=gr.Progress()) -> Tuple[Optional[Image.Image], str, str]:
    """Generate image with quality settings and enhanced prompt"""
    try:
        progress(0, desc="Starting image generation... ")
        
        if enhance_enabled:
            prompt = enhance_prompt(prompt, model_name, enhancement_level)
            progress(0.2, desc=f"Enhanced prompt (Level {enhancement_level}): {prompt} ")
        
        # Add content warnings to prompt if needed
        if adult_content:
            prompt = f"{prompt} (Warning: May contain adult content)"
        
        progress(0.4, desc=f"Generating image with {model_name}... ")
        
        # Create async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def generate():
            async with aiohttp.ClientSession() as session:
                # Try primary model first
                image, gen_time, status = await generate_image_async(prompt, model_name, session, safety_checker, adult_content)
                
                # If primary model fails, try FLUX.1 Schnell as fallback
                if image is None and model_name != "FLUX.1 Schnell (Fast)":
                    progress(0.6, desc="Trying faster model as fallback... ")
                    image, gen_time, status = await generate_image_async(prompt, "FLUX.1 Schnell (Fast)", session, safety_checker, adult_content)
                    if image is not None:
                        status = "Generated with fallback model (FLUX.1 Schnell)"
                
                if image is None:
                    return None, prompt, f" {status}"
                
                progress(1.0, desc="Done! ")
                return image, prompt, f" {status} in {gen_time:.2f}s "
        
        return loop.run_until_complete(generate())
    
    except Exception as e:
        return None, prompt, f" Error: {str(e)}"

# Example prompts for inspiration
example_prompts = [
    ["A magical forest with glowing mushrooms and fairy lights"],
    ["A cyberpunk city at night with neon signs"],
    ["A cute robot playing with butterflies in a meadow"],
    ["A majestic dragon soaring through storm clouds"],
]

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  AI Image Generation with FLUX.1 Models ")
    gr.Markdown("Choose a model and enter a prompt to generate amazing images! ")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Model selection
            model_dropdown = gr.Dropdown(
                choices=list(FLUX_MODELS.keys()),
                value="FLUX.1 Schnell (Fast)",  # Default to fastest model
                label=" Select Model",
                info="Choose the best model for your needs"
            )
            
            # Model description
            model_info = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    # Original prompt input
                    prompt_input = gr.Textbox(
                        label=" Original Prompt",
                        placeholder="Describe your imagination here...",
                        lines=3,
                        scale=1
                    )
                with gr.Column():
                    # Enhanced prompt display
                    enhanced_prompt_output = gr.Textbox(
                        label=" Enhanced Prompt",
                        placeholder="Enhanced prompt will appear here...",
                        lines=3,
                        scale=1,
                        interactive=False
                    )
            
            # Add prompt enhancement toggle
            with gr.Row():
                enhance_checkbox = gr.Checkbox(
                    label=" Auto-Enhance Prompt",
                    value=True,
                    info="Automatically enhance your prompt with additional details"
                )
                copy_btn = gr.Button(" Copy Enhanced", scale=0)
                enhancement_slider = gr.Slider(
                    label="Enhancement Level",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="Adjust the level of enhancement for your prompt"
                )
            
            # Add safety settings
            with gr.Row():
                safety_checkbox = gr.Checkbox(
                    label=" Enable Safety Checker",
                    value=True,
                    info="Enable safety checker to filter out explicit content"
                )
                adult_checkbox = gr.Checkbox(
                    label=" Allow Adult Content",
                    value=False,
                    info="Allow adult content in generated images"
                )
            
            # Add generate/clear buttons
            with gr.Row():
                generate_btn = gr.Button(" Generate", variant="primary", scale=2)
                clear_btn = gr.Button(" Clear", scale=1)
            
            # Add examples section
            gr.Examples(
                examples=example_prompts,
                inputs=prompt_input,
                label=" Example Prompts"
            )
        
        with gr.Column(scale=3):
            # Add image output and status
            image_output = gr.Image(label=" Generated Image")
            status_output = gr.Markdown(" Ready to generate! Enter a prompt and click Generate.")
    
    # Event handlers
    def clear_outputs():
        return None, "", " Ready to generate! Enter a prompt and click Generate."
    
    def update_model_info(model_name):
        return f" **{model_name}**\n\n{FLUX_MODELS[model_name].description}\nTypical response time: {FLUX_MODELS[model_name].timeout} seconds"
    
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