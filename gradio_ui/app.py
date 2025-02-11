import gradio as gr

from diffusers import StableDiffusionPipeline

import torch

from PIL import Image

from utils.constants import MODEL_PATH



# Load the model

model_path = MODEL_PATH

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

pipe.to("cuda")



# Fixed negative prompt

fixed_negative_prompt = "weird colors, low quality, jpeg artifacts, lowres, grainy, deformed structures, blurry, opaque, low contrast, distorted details, details are low"



# Function to generate images based on input text

def generate_image(prompt):

    image = pipe(prompt=prompt, negative_prompt=fixed_negative_prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

    return image



# Create a Gradio interface with a submit button

iface = gr.Interface(

    fn=generate_image,

    inputs="text",

    outputs=gr.Image(),  # Initial placeholder for the image,

    title="RemoteDiff224 Image Generator",

    description="Generate images based on input prompts with a fixed negative prompt.",

)



# Launch the Gradio interface

iface.launch(share=True)


