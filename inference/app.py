"""Gradio interface for Remote Sensing Image Generation"""
import gradio as gr
from pathlib import Path
from model import RemoteInferenceModel
from utils.constants import UI_CONFIG
from utils.logger import setup_logger

logger = setup_logger(
    "gradio_app",
    Path("logs/gradio_app.log")
)

def create_interface():
    """Create and configure Gradio interface."""
    try:
        # Initialize model
        model = RemoteInferenceModel()
        
        # Create interface
        interface = gr.Interface(
            fn=model.generate_image,
            inputs="text",
            outputs=gr.Image(),
            title=UI_CONFIG["title"],
            description=UI_CONFIG["description"],
        )
        
        return interface
        
    except Exception as e:
        logger.error(f"Error creating interface: {e}")
        raise

def main():
    """Launch the Gradio interface."""
    try:
        interface = create_interface()
        interface.launch(share=True, debug=True)
        
    except Exception as e:
        logger.error(f"Error launching interface: {e}")
        raise

if __name__ == "__main__":
    main() 