# Friday Assistant: LLM-Driven Voice Assistant Framework


## Overview

**Friday Assistant** is a voice-activated assistant that leverages a local Large Language Model (LLM) via Ollama to intelligently decide whether to handle user requests with a built-in Python function or to generate a conversational response. The assistant listens for a wake word ("friday") and supports functions such as:
- **Telling the current time**
- **Retrieving the latest stock price for a given ticker**

The core innovation lies in its **LLM Function Orchestration Framework**: when a user speaks, the system transcribes the audio, sends the query to a local LLM (using Ollama), and the LLM determines if a specific function should be executed. If so, the corresponding Python function is called (e.g., `tell_time()` or `get_stock_price()`) with any necessary arguments; otherwise, a general conversational response is generated.

---

## Features

- **Voice Activation:** Listens continuously for the wake word "friday".
- **Speech-to-Text:** Uses OpenAI's Whisper model to transcribe user speech.
- **Local LLM Integration:** Employs a local LLM (via Ollama) to decide function invocation.
- **Function Orchestration:** Supports direct function calls to:
  - Tell the current time.
  - Retrieve stock prices using Yahoo Finance.
  - Shut down the assistant.
- **Conversational History:** Maintains a brief conversation history to provide context in responses.
- **Text-to-Speech:** Uses gTTS (Google Text-to-Speech) for audio feedback.

---

## Architecture & Framework

### LLM Function Orchestration
At the heart of Friday Assistant is a framework that allows the local LLM to decide if a function call is necessary:
- **Decision Making:** The function `decide_function_or_none(user_query)` constructs a prompt that describes the available Python functions. The LLM then responds in JSON format, indicating which function to invoke (if any) along with the required arguments.
- **Dynamic Function Execution:** Based on the LLM’s decision, the assistant can directly call functions like:
  - `tell_time()`: Returns the current time.
  - `get_stock_price(ticker)`: Retrieves the last closing stock price for a given ticker symbol.
  - `shut_down()`: Ends the session.
- **Fallback to Conversation:** If no specific function is needed, the assistant resorts to a general conversational LLM prompt, ensuring a smooth and natural user experience.

### Audio Processing Pipeline
- **Audio Capture:** Uses the `sounddevice` library to continuously capture audio data.
- **Transcription:** Implements Whisper to transcribe spoken language into text.
- **Command Segmentation:** Detects periods of silence to determine when a complete command has been spoken.

### Additional Components
- **Stock Data Integration:** Uses the `yfinance` library to fetch real-time stock data.
- **Speech Synthesis:** Employs gTTS and playsound to convert text responses into spoken audio.

---

## Installation & Setup

### Prerequisites
- **Python 3.7+**
- System libraries and dependencies for audio processing (e.g., PortAudio for `sounddevice`).
- Access to the local LLM via Ollama (ensure that Ollama and the desired model such as `llama3` are installed and properly configured).

### Required Python Libraries
Install the required packages using pip:
```bash
pip install whisper sounddevice scipy numpy yfinance langchain langchain_ollama gTTS playsound
