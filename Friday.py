import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import threading
import tempfile
import time
import re
import json
from datetime import datetime
from collections import deque
import yfinance as yf  # New import for stock data

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM

# New imports for gTTS TTS playback
from gtts import gTTS
from playsound import playsound

##############################################################################
# Configuration
##############################################################################

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.025
SILENCE_DURATION = 1.0
BUFFER_DURATION = 5
MAX_BUFFER_SAMPLES = SAMPLE_RATE * BUFFER_DURATION
MAX_HISTORY_LENGTH = 6

audio_buffer = np.array([], dtype=np.int16)
buffer_lock = threading.Lock()
processing = False

conversation_history = deque(maxlen=MAX_HISTORY_LENGTH)

# Load local LLM (e.g. Llama v3)
llm = OllamaLLM(model="llama3")

# Load Whisper model
whisper_model = whisper.load_model("base", device="cpu")

##############################################################################
# Define Your Functions
##############################################################################

def tell_time():
    """Return a string containing the current time."""
    return datetime.now().strftime("It's %I:%M %p right now.")

def shut_down():
    """Shut down the assistant."""
    speak("Shutting down. Goodbye.")
    update_history("shutdown", "Assistant shutting down.")
    exit(0)

def get_stock_price(ticker):
    """
    Get the last share price for the given stock ticker using Yahoo Finance.
    It retrieves the historical data for the past day and extracts the last closing price.
    Returns a string like: "The last price for AAPL is 175.23 dollars."
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d")
        if df.empty:
            return f"Sorry, I could not retrieve the price for {ticker.upper()}."
        last_price = df['Close'].iloc[-1]
        return f"The last price for {ticker.upper()} is {last_price} dollars."
    except Exception as e:
        return f"An error occurred while retrieving the price for {ticker.upper()}: {e}"

##############################################################################
# Audio Handling
##############################################################################

def audio_callback(indata, frames, time_info, status_flags):
    global audio_buffer
    with buffer_lock:
        audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))
        if len(audio_buffer) > MAX_BUFFER_SAMPLES:
            audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]

def start_audio_stream():
    """Continuously open the audio stream. On error, retry after a short pause."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        sd.sleep(10**9)

def record_until_silence():
    global audio_buffer
    recorded = np.array([], dtype=np.int16)
    last_sound_time = time.time()
    
    while True:
        time.sleep(0.1)
        with buffer_lock:
            chunk = audio_buffer.copy()
            audio_buffer = np.array([], dtype=np.int16)
        if len(chunk) > 0:
            recorded = np.concatenate((recorded, chunk))
            rms = np.sqrt(np.mean(chunk**2))
            if rms > SILENCE_THRESHOLD:
                last_sound_time = time.time()
        if time.time() - last_sound_time >= SILENCE_DURATION:
            break
        if time.time() - last_sound_time > 10:
            break
    return recorded

def transcribe_audio(audio_data):
    if len(audio_data) == 0:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, SAMPLE_RATE, audio_data)
        result = whisper_model.transcribe(tmpfile.name)
        return result['text'].strip()

##############################################################################
# LLM + Function Orchestration
##############################################################################

def update_history(user_text, assistant_text):
    conversation_history.append({
        'user': user_text,
        'assistant': assistant_text,
        'timestamp': datetime.now().isoformat()
    })

def decide_function_or_none(user_query):
    """
    Ask the LLM (via a direct prompt) if we need a function call.
    Available functions:
      1) tell_time(): returns the current time.
      2) shut_down(): shuts down the assistant.
      3) get_stock_price(ticker): returns the last share price for the given stock ticker.
    
    If the user is asking for the current time, respond with JSON:
      { "use_function": "tell_time" }
    If the user is asking to shut down the assistant, respond with JSON:
      { "use_function": "shut_down" }
    If the user is asking for a stock price, respond with JSON:
      { "use_function": "get_stock_price", "arguments": { "ticker": "<ticker>" } }
    Otherwise respond with JSON:
      { "use_function": "none" }
    
    User Query: {user_query}
    """
    decision_prompt = f"""
You are an AI assistant with the following functions:
1) tell_time(): returns the current time.
2) shut_down(): shuts down the assistant.
3) get_stock_price(ticker): returns the last share price for the given stock ticker.

If the user is asking for the current time, respond with JSON:
{{ "use_function": "tell_time" }}
If the user is asking to shut down the assistant, respond with JSON:
{{ "use_function": "shut_down" }}
If the user is asking for a stock price, respond with JSON:
{{ "use_function": "get_stock_price", "arguments": {{ "ticker": "<ticker>" }} }}
Otherwise respond with JSON:
{{ "use_function": "none" }}

User Query: {user_query}
"""
    raw_output = llm(decision_prompt)
    # Default result.
    result = {"use_function": "none", "arguments": {}}
    try:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            decision_json = json.loads(match.group())
            result["use_function"] = decision_json.get("use_function", "none").strip()
            result["arguments"] = decision_json.get("arguments", {})
    except Exception:
        pass
    return result

def handle_query(user_query):
    """
    1) Decide if we need a function call.
    2) If the decision is "tell_time", call tell_time().
    3) If the decision is "shut_down", call shut_down().
    4) If the decision is "get_stock_price", call get_stock_price() with the provided ticker.
    5) Otherwise, perform a normal LLM chat.
    """
    decision = decide_function_or_none(user_query)
    function_choice = decision["use_function"]
    
    if function_choice == "tell_time":
        return tell_time()
    elif function_choice == "shut_down":
        shut_down()  # This will exit the program.
    elif function_choice == "get_stock_price":
        arguments = decision.get("arguments", {})
        ticker = arguments.get("ticker")
        if ticker is None:
            return "No ticker provided."
        return get_stock_price(ticker)
    else:
        normal_prompt = PromptTemplate(
            input_variables=["history", "query"],
            template=(
                "The conversation so far:\n{history}\n\n"
                "User just asked: {query}\n\n"
                "Provide a helpful, accurate response:"
            )
        )
        chain = LLMChain(llm=llm, prompt=normal_prompt)
        
        history_text = "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in conversation_history
        )
        
        response = chain.run({"history": history_text, "query": user_query})
        return response.strip()

def speak(text):
    """
    Use gTTS to synthesize speech and playsound to play it.
    This provides a more natural voice than the macOS 'say' command.
    """
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            temp_filename = f.name
        tts.save(temp_filename)
        from playsound import playsound
        playsound(temp_filename)
        os.remove(temp_filename)
    except Exception as e:
        print(f"Error in TTS: {e}")
    return text

##############################################################################
# Main Application
##############################################################################

def main():
    global processing
    
    print("Assistant active. Say 'Jarvis' to wake me up.")
    speak("Online and ready for your commands.")
    
    # Start the audio capture in a daemon thread.
    threading.Thread(target=start_audio_stream, daemon=True).start()
    
    while True:
        if processing:
            time.sleep(0.1)
            continue
        
        with buffer_lock:
            current_buffer = audio_buffer.copy()
        
        if len(current_buffer) > 0:
            transcript = transcribe_audio(current_buffer).lower()
            
            if "jarvis" in transcript:
                processing = True
                try:
                    command_audio = record_until_silence()
                    full_command = transcribe_audio(command_audio).strip()
                    
                    if "jarvis" in full_command.lower():
                        full_command = re.sub(r'(?i)\bjarvis\b', '', full_command).strip()
                    
                    print(f"[User] {full_command}")
                    
                    answer = handle_query(full_command)
                    print(f"[Assistant] {answer}")
                    speak(answer)
                    
                    update_history(full_command, answer)
                finally:
                    processing = False
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()