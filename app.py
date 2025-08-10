# app.py

import gradio as gr
from typing import Union
from transformers import pipeline
from chatbot import get_faq_response

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
    gr.Markdown("""# 🛍️ E-commerce FAQ Chatbot
Ask anything about **orders**, **shipping**, **returns**, **payments**, and more.  
Use **voice** or **text** below.
""")
    with gr.Row():
        audio_in = gr.Audio(source="microphone", type="filepath", label="🎤 Ask by voice")
        text_in = gr.Textbox(label="⌨️ Or type your question", placeholder="e.g., How do I track my order?")
    submit_btn = gr.Button("Ask")
    output = gr.Markdown()

    submit_btn.click(fn=combined_input_handler, inputs=[audio_in, text_in], outputs=output)
    text_in.submit(fn=combined_input_handler, inputs=[audio_in, text_in], outputs=output)

if __name__ == "__main__":
    demo.launch()
