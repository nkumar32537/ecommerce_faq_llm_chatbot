# app.py

import gradio as gr
from typing import Union
from transformers import pipeline
from chatbot import get_faq_response
import os, shutil, platform

# Load ASR pipeline (Whisper). Requires ffmpeg installed on the system.
# If this fails on your machine, comment out the ASR-related lines and use text input only.
try:
    speech_to_text = pipeline("automatic-speech-recognition", model="openai/whisper-base")
except Exception as e:
    speech_to_text = None
    print(f"[WARN] Could not load ASR model: {e}. Voice input will be disabled.")

def combined_input_handler(audio_file: Union[str, None], text_input: str) -> str:
    query = None
    if audio_file and speech_to_text is not None:
        try:
            result = speech_to_text(audio_file)
            query = result.get("text", "").strip()
        except Exception as e:
            return f"Sorry, I couldn't transcribe your audio ({e}). Please type your question."
    if not query:
        query = (text_input or "").strip()
    if not query:
        return "Please ask a question using voice or text."
    return get_faq_response(query)

with gr.Blocks(title="E-commerce FAQ Chatbot") as demo:
    gr.Markdown("""# E-commerce FAQ Chatbot
    Created By: Nishant Kumar, Course: MSAI-631 - Artificial Intelligence for Human-Computer Interaction 
Ask anything about **orders**, **shipping**, **returns**, **payments**, and more. Use **voice** or **text** below.
""")
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Ask by voice")
        text_in = gr.Textbox(label=" Or type your question", placeholder="e.g., How do I track my order?")
    submit_btn = gr.Button("Ask")
    output = gr.Markdown()

    submit_btn.click(fn=combined_input_handler, inputs=[audio_in, text_in], outputs=output)
    text_in.submit(fn=combined_input_handler, inputs=[audio_in, text_in], outputs=output)

if __name__ == "__main__":
    # --- force-add FFmpeg to PATH on Windows (edit the path below to match your install) ---
    if platform.system() == "Windows":
        ffmpeg_dir = r"C:\ffmpeg-7.1.1-essentials_build\bin"  # <-- change if you installed elsewhere
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + ";" + os.environ.get("PATH", "")
        # quick log so you can see what Python resolves
        print("FFmpeg resolved to:", shutil.which("ffmpeg"))

    demo.launch()
