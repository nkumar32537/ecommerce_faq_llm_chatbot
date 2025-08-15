# E-commerce FAQ Chatbot (RAG + LLM Fallback)

## Nishant Kumar
### University of the Cumberlands
### MSAI-631- B01 Artificial Intelligence for Human-Computer Interaction 


## Overview
A lightweight Python project that answers e-commerce FAQs with a retrieval-first approach
(FAISS + MiniLM embeddings) and falls back to a small chat-tuned LLM when retrieval
confidence is low. Includes optional voice input via Whisper ASR.

## Features
- **RAG retrieval** over your FAQ dataset using `all-MiniLM-L6-v2` + FAISS
- **Generative fallback** with `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (HF Transformers)
- **Gradio UI** with both **microphone** and **textbox** inputs
- Runs locally on CPU/GPU (CPU works; GPU recommended for faster LLM/ASR)

## Project Structure
```
ecommerce_faq_chatbot/
├── app.py                     # Gradio UI
├── chatbot.py                 # Core logic (RAG + LLM fallback)
├── utils.py                   # Dataset loading, embeddings, FAISS index
├── config.py                  # Thresholds, model name
├── Ecommerce_FAQ_Chatbot_dataset.json  # FAQ dataset
├── requirements.txt
└── README.md
```

## Quickstart
1. **Create/activate a virtual environment** (optional but recommended).  
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Voice input) Install ffmpeg** on your system (required by Whisper). 
   
   a) On macOS:
   ```bash
   brew install ffmpeg
   ```
   b) On Ubuntu/Debian:
   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```

   c)On Window:
 
   Download FFmpeg from the official site:
   https://ffmpeg.org/download.html → Windows builds by gyan.dev

   Extract the ZIP to a folder (e.g., C:\ffmpeg).
   Add the bin folder to your PATH (Environment Variables).

   Under System variables → select Path → Edit → New → add:

   C:\ffmpeg\bin

4. **Run the app**:
   ```bash
   python app.py
   ```
5. Open the local URL shown by Gradio.

## Notes
- We need to install ffmpeg to make voice or audio-based functionality work. In my case, it is C:\ffmpeg-7.1.1-essentials_build
- We can swap the fallback model in `config.py` (e.g., upgrade to a larger chat model).
- If ASR model download fails or we don't have ffmpeg, voice input will be disabled and text input will continue to work.
- The similarity threshold is configurable in `config.py`.

## Dataset
The included dataset file `Ecommerce_FAQ_Chatbot_dataset.json` is expected to have the structure:
```json
{
  "questions": [
    {"question": "How can I create an account?", "answer": "..."}
  ]
}
```
