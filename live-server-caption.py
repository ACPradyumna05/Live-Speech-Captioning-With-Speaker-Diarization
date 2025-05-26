from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate
import numpy as np
import sounddevice as sd
import whisper
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from sklearn.cluster import KMeans
import datetime
import queue
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Audio processing configuration
SAMPLING_RATE = 16000
CHUNK_DURATION = 10
CHANNELS = 1

# Model configuration
MODEL_CONFIGS = {
    'whisper_base': {
        'name': 'Whisper Base',
        'description': 'Fast, lightweight model for real-time transcription'
    },
    'whisper_large': {
        'name': 'Whisper Large v2',
        'description': 'High accuracy model with slower processing'
    },
    'wav2vec2': {
        'name': 'Wav2Vec2 Large',
        'description': 'Facebook\'s transformer-based model'
    }
}

# Global variables for models and processing
current_model_type = 'whisper_base'
models = {}
encoder = None
audio_queue = queue.Queue()
is_recording = False
audio_thread = None

def initialize_models():
    """Initialize all available models"""
    global models, encoder
    
    print("Loading voice encoder...")
    encoder = VoiceEncoder()
    print("Voice encoder loaded!")
    
    # Initialize with base model by default
    load_model('whisper_base')

def load_model(model_type):
    """Load a specific transcription model"""
    global models, current_model_type
    
    if model_type in models:
        current_model_type = model_type
        print(f"✅ Switched to {MODEL_CONFIGS[model_type]['name']}")
        return True
    
    print(f"📥 Loading {MODEL_CONFIGS[model_type]['name']}...")
    
    try:
        if model_type == 'whisper_base':
            models[model_type] = whisper.load_model("base")
        elif model_type == 'whisper_large':
            models[model_type] = whisper.load_model("large-v2")
        elif model_type == 'wav2vec2':
            tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").eval()
            models[model_type] = {'model': model, 'tokenizer': tokenizer}
        
        current_model_type = model_type
        print(f"✅ {MODEL_CONFIGS[model_type]['name']} loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {MODEL_CONFIGS[model_type]['name']}: {e}")
        return False

def transcribe_audio(audio_float, model_type):
    """Transcribe audio using the specified model"""
    try:
        if model_type.startswith('whisper'):
            model = models[model_type]
            result = model.transcribe(audio_float, language="en", fp16=False)
            return result["text"].strip()
            
        elif model_type == 'wav2vec2':
            model_data = models[model_type]
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            input_values = tokenizer(audio_float, return_tensors="pt", padding="longest").input_values
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = tokenizer.batch_decode(predicted_ids)[0]
            return transcription.strip()
            
    except Exception as e:
        print(f"❌ Transcription error with {model_type}: {e}")
        return ""

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio stream"""
    if status:
        print(f"⚠️ Audio status: {status}")
    if is_recording:
        audio_queue.put(indata.copy().flatten())

def process_audio():
    """Process audio chunks and emit captions via WebSocket"""
    global is_recording
    
    stream = sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=int(SAMPLING_RATE * CHUNK_DURATION),
    )
    
    try:
        with stream:
            print(f"🎤 Audio stream started with {MODEL_CONFIGS[current_model_type]['name']}")
            while is_recording:
                try:
                    # Get audio chunk with timeout to allow checking is_recording
                    audio_chunk = audio_queue.get(timeout=1.0)
                    
                    # Preprocess audio
                    audio_float = audio_chunk.astype(np.float32)
                    if np.max(np.abs(audio_float)) > 0:
                        audio_float /= np.max(np.abs(audio_float))
                    else:
                        continue  # Skip silent chunks
                    
                    # Transcribe audio using current model
                    text = transcribe_audio(audio_float, current_model_type)
                    
                    if not text:  # Skip empty transcriptions
                        continue
                    
                    # Voice embedding and speaker diarization
                    wav = preprocess_wav(audio_float)
                    _, embeddings, slices = encoder.embed_utterance(wav, return_partials=True, rate=16)
                    
                    if len(embeddings) == 0:
                        continue
                    
                    # Determine number of speakers (max 4, min 1)
                    n_speakers = min(4, max(1, len(embeddings) // 3))
                    if n_speakers > 1:
                        kmeans = KMeans(n_clusters=n_speakers, random_state=0).fit(embeddings)
                        labels = kmeans.labels_
                    else:
                        labels = [0] * len(embeddings)
                    
                    # Create speaker segments
                    words = text.split()
                    if len(words) == 0:
                        continue
                        
                    avg_words_per_segment = max(10, len(words) // len(labels))
                    word_idx = 0
                    segment_captions = []
                    
                    for i, label in enumerate(labels):
                        speaker = f"Speaker {label + 1}"
                        end_idx = min(word_idx + avg_words_per_segment, len(words))
                        caption_words = words[word_idx:end_idx]
                        word_idx = end_idx
                        
                        if caption_words:
                            caption_text = f"[{speaker}]: {' '.join(caption_words)}"
                            segment_captions.append(caption_text)
                    
                    # Emit captions via WebSocket
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    for caption in segment_captions:
                        socketio.emit('caption', {
                            'timestamp': timestamp,
                            'text': caption,
                            'model': MODEL_CONFIGS[current_model_type]['name']
                        })
                        print(f"🕒 {timestamp} - {caption}")
                    
                except queue.Empty:
                    continue  # Timeout occurred, check if still recording
                except Exception as e:
                    print(f"⚠️ Error processing audio: {e}")
                    continue
                    
    except Exception as e:
        print(f"⚠️ Audio stream error: {e}")
    finally:
        print("🛑 Audio stream stopped")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('front.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to server'})
    # Send available models and current selection
    emit('models_info', {
        'available_models': MODEL_CONFIGS,
        'current_model': current_model_type,
        'loaded_models': list(models.keys())
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_recording')
def handle_start_recording():
    """Start audio recording and processing"""
    global is_recording, audio_thread
    
    if not current_model_type in models:
        emit('status', {'message': 'Model not loaded. Please select a model first.'})
        return
    
    if not is_recording:
        is_recording = True
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()
        emit('status', {
            'message': f'Recording started with {MODEL_CONFIGS[current_model_type]["name"]}'
        })
        print(f"🎤 Recording started with {MODEL_CONFIGS[current_model_type]['name']}")
    else:
        emit('status', {'message': 'Already recording'})

@socketio.on('stop_recording')
def handle_stop_recording():
    """Stop audio recording and processing"""
    global is_recording
    
    if is_recording:
        is_recording = False
        emit('status', {'message': 'Recording stopped'})
        print("🛑 Recording stopped")
    else:
        emit('status', {'message': 'Not currently recording'})

@socketio.on('change_model')
def handle_change_model(data):
    """Change the transcription model"""
    global is_recording
    
    model_type = data.get('model_type')
    
    if model_type not in MODEL_CONFIGS:
        emit('status', {'message': f'Invalid model type: {model_type}'})
        return
    
    if is_recording:
        emit('status', {'message': 'Cannot change model while recording. Please stop recording first.'})
        return
    
    # Show loading status
    emit('status', {'message': f'Loading {MODEL_CONFIGS[model_type]["name"]}...'})
    emit('model_loading', {'model_type': model_type, 'loading': True})
    
    # Load model in a separate thread to avoid blocking
    def load_model_thread():
        success = load_model(model_type)
        if success:
            socketio.emit('status', {
                'message': f'Successfully loaded {MODEL_CONFIGS[model_type]["name"]}'
            })
            socketio.emit('model_changed', {
                'model_type': model_type,
                'model_name': MODEL_CONFIGS[model_type]['name']
            })
        else:
            socketio.emit('status', {
                'message': f'Failed to load {MODEL_CONFIGS[model_type]["name"]}'
            })
        
        socketio.emit('model_loading', {'model_type': model_type, 'loading': False})
        socketio.emit('models_info', {
            'available_models': MODEL_CONFIGS,
            'current_model': current_model_type,
            'loaded_models': list(models.keys())
        })
    
    threading.Thread(target=load_model_thread, daemon=True).start()

@socketio.on('get_models_info')
def handle_get_models_info():
    """Get information about available models"""
    emit('models_info', {
        'available_models': MODEL_CONFIGS,
        'current_model': current_model_type,
        'loaded_models': list(models.keys())
    })

if __name__ == '__main__':
    print("🚀 Starting Live Speaker-Diarized Captioning Server...")
    print("📱 Access the web interface at: http://localhost:5000")
    
    # Initialize models
    initialize_models()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)