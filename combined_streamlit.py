import streamlit as st

st.set_page_config(
        page_title="Voice Synthesis Studio",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
)

import torch
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Configuration and Constants
EMOTION_DESCRIPTIONS = {
    "whisper": "Thomas speaks quietly in a whisper.",
    "enunciation": "Thomas speaks with clear and articulate enunciation.",
    "sad": "Thomas speaks with a sad tone.",
    "default": "Thomas speaks with a neutral tone with no particular emotion in normal speed.",
    "laughing": "Thomas speaks with laughter in his voice.",
    "confused": "Thomas speaks in a confued tone with confusion in his voice.",
    "happy": "Thomas expresses joyfully with happiness in his speech.",
    "emphasis": "Thomas expresses importance through his speech with emphasis."
}

MODEL_CONFIGS = {
    "Indian Accent": {
        "path": "En1gma02/Parler-TTS-Mini-v0.1-Indian-Accent-Kaggle",
        "icon": "🗣️",
        "description": """
            #### About This Model
            - Specialized for Indian English accents
            - Natural speech patterns
            - High clarity and pronunciation
            - Best for general purpose text-to-speech
        """,
        "default_text": "Welcome to the Voice Synthesis Studio!",
        "default_description": "Akshansh delivers his words quite expressively, in a very confined sounding environment with clear audio quality. He speaks fast."
    },
    "Hindi": {
        "path": "En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle",
        "icon": "🇮🇳",
        "description": """
            #### About This Model
            - Native Hindi language support
            - Natural Indian voice
            - Devanagari script compatible
            - Optimized for Hindi pronunciation
        """,
        "default_text": "भारत में कई प्रकार की भाषाएँ बोली जाती हैं।",
        "default_description": "Male Hindi Speaker, clear and natural delivery, ambient studio environment."
    },
    "Emotions": {
        "path": "En1gma02/Parler-TTS-Mini-v1-English-Emotions",
        "icon": "🎭",
        "description": """
            #### About This Model
            - Multiple emotion styles
            - Dynamic voice modulation
            - Natural emotional transitions
            - Enhanced expression control
        """,
        "default_text": "Let me express this with the perfect emotion!",
        "default_description": EMOTION_DESCRIPTIONS["default"]
    }
}
'''
# Enhanced CSS styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }

    /* Header styling */
    .stMarkdown h1 {
        color: #1a237e;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Subheader styling */
    .stMarkdown h2 {
        color: #1976d2;
        font-size: 1.8rem;
        margin-top: 2rem;
        border-bottom: 2px solid #bbdefb;
        padding-bottom: 0.5rem;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background: linear-gradient(45deg, #1976d2, #2196f3);
        color: white;
        border-radius: 10px;
        padding: 0.8rem;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        background: linear-gradient(45deg, #1565c0, #1976d2);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }

    /* Input area styling */
    .stTextArea>div>div>textarea {
        background-color: white;
        border-radius: 10px;
        border: 2px solid #e3f2fd;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }

    .stTextArea>div>div>textarea:focus {
        border-color: #2196f3;
        box-shadow: 0 0 0 2px rgba(33,150,243,0.2);
    }

    /* Card styling */
    .css-1y4p8pa {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #e3f2fd;
        padding: 2rem 1rem;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #2196f3;
    }

    /* Alert styling */
    .stAlert {
        background-color: #e3f2fd;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Navigation menu styling */
    .nav-link {
        padding: 0.8rem 1rem;
        background-color: white;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
        text-decoration: none;
        color: #1976d2;
        display: block;
    }

    .nav-link:hover {
        background-color: #bbdefb;
        transform: translateX(5px);
    }

    /* Status indicator styling */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-active {
        background-color: #4caf50;
        box-shadow: 0 0 5px #4caf50;
    }

    .status-inactive {
        background-color: #f44336;
        box-shadow: 0 0 5px #f44336;
    }

    /* History section styling */
    .history-item {
        padding: 0.8rem;
        background-color: white;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    </style>
""", unsafe_allow_html=True)
'''
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'model': None,
        'tokenizer': None,
        'current_model_type': 'Indian Accent',
        'history': [],
        'current_tab': 'Generate',
        'selected_emotion': 'default',
        'generation_count': 0,
        'last_generated_audio': None,
        'model_loading_state': 'not_loaded'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_models(model_type):
    """Enhanced model loading with better error handling and progress tracking"""
    try:
        st.session_state.model_loading_state = 'loading'
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Stage 1: Initialize
        status_text.text("Initializing model loading...")
        progress_bar.progress(20)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_path = MODEL_CONFIGS[model_type]["path"]

        # Stage 2: Load Model
        status_text.text("Loading model architecture...")
        progress_bar.progress(40)
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)

        # Stage 3: Load Tokenizer
        status_text.text("Loading tokenizer...")
        progress_bar.progress(60)
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

        # Stage 4: Finalize
        status_text.text("Finalizing setup...")
        progress_bar.progress(80)

        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_loading_state = 'loaded'

        # Complete
        progress_bar.progress(100)
        status_text.text("Model loaded successfully!")
        return True

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loading_state = 'error'
        return False
    finally:
        progress_bar.empty()
        status_text.empty()


def generate_audio(text, description, model, tokenizer, model_type):
    """Generate audio with progress tracking"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Preprocessing
        status_text.text("Preprocessing text...")
        progress_bar.progress(20)
        if model_type == "Hindi":
            text = transliterate(text, DEVANAGARI, ITRANS)

        # Tokenization
        status_text.text("Tokenizing input...")
        progress_bar.progress(40)
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        # Generation
        status_text.text("Generating audio...")
        progress_bar.progress(60)
        with torch.no_grad():
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

        # Post-processing
        status_text.text("Processing audio...")
        progress_bar.progress(80)
        audio_array = generation.cpu().numpy().squeeze()

        return audio_array

    finally:
        progress_bar.empty()
        status_text.empty()


def render_sidebar():
    """Render the sidebar with enhanced UI"""
    with st.sidebar:
        st.title(f"🎙️ Voice Studio")

        # Model Selection Section
        st.markdown("### Select Model")
        for model_type, config in MODEL_CONFIGS.items():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### {config['icon']}")
            with col2:
                if st.button(
                        model_type,
                        key=f"model_{model_type}",
                        use_container_width=True,
                        type="primary" if st.session_state.current_model_type == model_type else "secondary"
                ):
                    st.session_state.current_model_type = model_type

        st.markdown("---")

        # Model Status
        st.markdown("### Model Status")
        status_col1, status_col2 = st.columns([1, 4])
        with status_col1:
            if st.session_state.model_loading_state == 'loaded':
                st.markdown("🟢")
            elif st.session_state.model_loading_state == 'loading':
                st.markdown("🟡")
            else:
                st.markdown("🔴")
        with status_col2:
            if st.button("Load Model", use_container_width=True):
                load_models(st.session_state.current_model_type)

        # Model Info
        with st.expander("ℹ️ Model Information"):
            st.markdown(MODEL_CONFIGS[st.session_state.current_model_type]["description"])


def render_main_content():
    """Render the main content area"""
    st.title(f"{MODEL_CONFIGS[st.session_state.current_model_type]['icon']} Voice Synthesis Studio")

    # Input Section
    st.markdown("### Input Configuration")

    # Create tabs for different input sections
    input_tab, voice_tab = st.tabs(["📝 Text Input", "🎤 Voice Settings"])

    with input_tab:
        input_text = st.text_area(
            "Enter your text",
            value=MODEL_CONFIGS[st.session_state.current_model_type]["default_text"],
            height=150
        )

    with voice_tab:
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.session_state.current_model_type == "Emotions":
                selected_emotion = st.selectbox(
                    "Select Emotion",
                    list(EMOTION_DESCRIPTIONS.keys()),
                    index=list(EMOTION_DESCRIPTIONS.keys()).index(st.session_state.selected_emotion)
                )
                st.session_state.selected_emotion = selected_emotion
                default_description = EMOTION_DESCRIPTIONS[selected_emotion]
            else:
                default_description = MODEL_CONFIGS[st.session_state.current_model_type]["default_description"]

        with col2:
            voice_description = st.text_area(
                "Voice Description",
                value=default_description,
                height=150
            )

    # Generation Section
    st.markdown("### Generate Audio")
    generate_col1, generate_col2 = st.columns([2, 1])

    with generate_col1:
        if st.button("🎵 Generate Voice", use_container_width=True, type="primary"):
            if st.session_state.model is None:
                st.warning("⚠️ Please load the model first!")
            else:
                try:
                    audio_array = generate_audio(
                        input_text,
                        voice_description,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.current_model_type
                    )

                    # Save audio
                    output_path = f'generated_audio_{st.session_state.generation_count}.wav'
                    sf.write(output_path, audio_array.astype(np.float32), st.session_state.model.config.sampling_rate)

                    # Update state
                    st.session_state.last_generated_audio = output_path
                    st.session_state.generation_count += 1

                    # Add to history
                    st.session_state.history.append({
                        'text': input_text,
                        'model_type': st.session_state.current_model_type,
                        'emotion': st.session_state.selected_emotion if st.session_state.current_model_type == "Emotions" else None,
                        'file_path': output_path,
                        'timestamp': datetime.now()
                    })

                    st.success("✅ Generation successful!")

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")

    # Display last generated audio
    if st.session_state.last_generated_audio and os.path.exists(st.session_state.last_generated_audio):
        st.markdown("### 🎧 Generated Audio")
        audio_col, download_col = st.columns([3, 1])

        with audio_col:
            st.audio(st.session_state.last_generated_audio)

        with download_col:
            with open(st.session_state.last_generated_audio, 'rb') as f:
                st.download_button(
                    "📥 Download",
                    data=f,
                    file_name=f"voice_{st.session_state.current_model_type.lower()}_{st.session_state.generation_count}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )


def render_history():
    """Render the history page"""
    st.title("📜 Generation History")

    if not st.session_state.history:
        st.info("Your generation history will appear here!")
        return

    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Generation #{len(st.session_state.history) - idx}", expanded=idx == 0):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                    **Model:** {item['model_type']} {MODEL_CONFIGS[item['model_type']]['icon']}  
                    **Time:** {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  
                    **Text:** {item['text'][:100]}...  
                    {f"**Emotion:** {item['emotion']}" if item['emotion'] else ""}
                """)

            with col2:
                if os.path.exists(item['file_path']):
                    st.audio(item['file_path'])
                    with open(item['file_path'], 'rb') as f:
                        st.download_button(
                            "📥 Download",
                            data=f,
                            file_name=f"voice_history_{item['model_type'].lower()}_{idx}.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )


def main():
    """Main application function"""

    init_session_state()

    # Navigation
    render_sidebar()

    # Main content
    if st.session_state.current_tab == 'Generate':
        render_main_content()
    else:
        render_history()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Created with ❤️ by team DataKraft Corps</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
   main()
