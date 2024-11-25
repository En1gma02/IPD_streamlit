import streamlit as st
import torch
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os
import pandas as pd

# Define functions
def load_models(language):
    """Load the TTS model and tokenizer based on selected language"""
    with st.spinner("⚙️ Loading models... Please wait..."):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if language == "Hindi":
            model_path = "En1gma02/Parler-TTS-Mini-v0.1-Indian-Male-Accent-Hindi-Kaggle"
        else:
            model_path = "En1gma02/Parler-TTS-Mini-v0.1-Indian-Accent-Kaggle"

        model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.success("✅ Model loaded successfully!")

def generate_audio(text, description, model, tokenizer, language):
    """Generate audio from input text based on selected language"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if language == "Hindi":
        text = transliterate(text, DEVANAGARI, ITRANS)

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    return generation.cpu().numpy().squeeze()

# Page configuration
st.set_page_config(
    page_title="Voice Synthesis Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'current_language' not in st.session_state:
    st.session_state.current_language = 'English'
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Generate'

# Enhanced sidebar with better navigation
with st.sidebar:

    st.title("🎙️ Voice Synthesis Studio")

    # Model status indicator
    st.markdown("### System Status")
    if st.session_state.model is not None:
        st.markdown("""
            <div class="status-indicator status-active"></div>
            <span style="color: #4caf50; font-weight: 600;">Model Active</span>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-indicator status-inactive"></div>
            <span style="color: #f44336; font-weight: 600;">Model Not Loaded</span>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Enhanced navigation
    st.markdown("### 📌 Navigation")
    tabs = {
        'Generate': '🎵 Text to Speech',
        'History': '📜 Generation History',
    }

    for tab, label in tabs.items():
        if st.sidebar.button(label, key=f"nav_{tab}"):
            st.session_state.current_tab = tab

    st.markdown("---")

    # Language selection with enhanced UI
    st.markdown("### 🌐 Language Settings")
    selected_language = st.radio(
        "",
        ["English", "Hindi"],
        key="language_selector",
        on_change=lambda: setattr(st.session_state, 'current_language', st.session_state.language_selector)
    )

    if st.button("🔄 Load Selected Model", key="load_model"):
        load_models(selected_language)

    # Enhanced model information
    with st.expander("📚 Model Information"):
        if selected_language == "English":
            st.markdown("""
                #### Model Specifications
                - 🤖 **Name:** Indian accent soeaker based model finetuned on Parler-TTS-Mini
                - 📊 **Version:** 0.1
                - 🎯 **Type:** Indian Male Accent (English)
                - 🔊 **Specialization:** Natural English Speech with accents
            """)
        else:
            st.markdown("""
                #### Model Specifications
                - 🤖 **Name:** Hindi Male Speaker based model finetuned on Parler-TTS-Mini
                - 📊 **Version:** 0.1
                - 🎯 **Type:** Indian Male Accent (Hindi)
                - 🔊 **Specialization:** Devanagari Script Support
                - 🚀 **Performance:** Optimized for real-time generation
            """)

# Main content area with tab-based navigation
if st.session_state.current_tab == 'Generate':
    st.title("🎙️ Voice Synthesis Studio")

    # Create two columns for input with better spacing
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Input Text")
        if selected_language == "Hindi":
            default_text = "भारत में कई प्रकार की भाषाएँ बोली जाती हैं।"
            placeholder = "Enter Hindi text in Devanagari script..."
        else:
            default_text = "Welcome to the Voice Synthesis Studio!"
            placeholder = "Enter English text here..."

        input_text = st.text_area(
            "",
            value=default_text,
            height=200,
            placeholder=placeholder,
            key="input_text"
        )

    with col2:
        st.markdown("### 🎤 Voice Configuration")
        if selected_language == "Hindi":
            default_description = "Male Hindi Speaker, clear and natural delivery, ambient studio environment."
        else:
            default_description = "Akshansh delivers his words quite expressively, in a very confined sounding environment with clear audio quality. He speaks fast."

        voice_description = st.text_area(
            "",
            value=default_description,
            height=200,
            key="voice_description"
        )

    # Generate button and audio output
    if st.button("🎵 Generate Voice", key="generate_button"):
        if st.session_state.model is None:
            st.warning("⚠️ Please load the model first using the sidebar button!")
        else:
            try:
                with st.spinner("🎵 Crafting your audio... Please wait..."):
                    audio_array = generate_audio(
                        input_text,
                        voice_description,
                        st.session_state.model,
                        st.session_state.tokenizer,
                        selected_language
                    )

                    audio_array_float32 = audio_array.astype(np.float32)
                    output_path = f'generated_audio_{len(st.session_state.history)}.wav'
                    sf.write(output_path, audio_array_float32, st.session_state.model.config.sampling_rate)

                    # Add to history
                    st.session_state.history.append({
                        'text': input_text,
                        'language': selected_language,
                        'file_path': output_path,
                        'timestamp': pd.Timestamp.now()
                    })

                    # Display audio player and download button
                    st.markdown("### 🎧 Generated Audio")
                    audio_col, download_col = st.columns([3, 1])

                    with audio_col:
                        st.audio(output_path)

                    with download_col:
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Audio",
                                data=f,
                                file_name=f"voice_synthesis_{selected_language.lower()}.wav",
                                mime="audio/wav"
                            )

                    st.success("✅ Audio generated successfully!")

            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")

elif st.session_state.current_tab == 'History':
    st.title("📜 Generation History")

    if not st.session_state.history:
        st.info("No generations yet. Start by creating some audio!")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"""
                    <div class="history-item">
                        <h4>Generation #{len(st.session_state.history) - idx}</h4>
                        <p><strong>Language:</strong> {item['language']}</p>
                        <p><strong>Text:</strong> {item['text'][:100]}...</p>
                        <p><strong>Time:</strong> {item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                """, unsafe_allow_html=True)

                if os.path.exists(item['file_path']):
                    st.audio(item['file_path'])

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created with ❤️ by team DataKraft Corps using Streamlit</p>
    </div>
""", unsafe_allow_html=True)