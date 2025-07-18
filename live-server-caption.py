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


from sklearn.metrics import silhouette_score

# Globals (if not already defined)
last_kmeans_time = 0
KMEANS_INTERVAL = 10
speaker_labels = []
speaker_centroids = {}
speaker_counter = 0
SLIDING_WINDOW_DURATION = 30  # seconds
sliding_audio_buffer = np.zeros((SAMPLING_RATE * SLIDING_WINDOW_DURATION,), dtype=np.float32)

def estimate_speakers(embeddings, max_speakers=5):
    from sklearn.metrics import silhouette_score
    best_k, best_score = 1, -1
    for k in range(2, max_speakers + 1):
        try:
            km = KMeans(n_clusters=k).fit(embeddings)
            score = silhouette_score(embeddings, km.labels_)
            if score > best_score:
                best_k = k
                best_score = score
        except:
            continue
    return best_k



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
            result = model.transcribe(audio_float, language="en", word_timestamps=True, fp16=False)
            return result["text"].strip(), result.get("segments", [])
            
        elif model_type == 'wav2vec2':
            model_data = models[model_type]
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            input_values = tokenizer(audio_float, return_tensors="pt", padding="longest").input_values
            with torch.no_grad():
                logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = tokenizer.batch_decode(predicted_ids)[0]
            return transcription.strip(),[]
            
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
    global is_recording, sliding_audio_buffer, last_kmeans_time, speaker_centroids, speaker_counter
    
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
                    audio_chunk = audio_queue.get(timeout=1.0)
                    audio_float = audio_chunk.astype(np.float32)
                    if np.max(np.abs(audio_float)) > 0:
                        audio_float /= (np.max(np.abs(audio_float)) + 1e-6)
                    else:
                        continue

                    # Update the sliding buffer
                    buffer_len = len(sliding_audio_buffer)
                    new_len = len(audio_float)
                    sliding_audio_buffer = np.roll(sliding_audio_buffer, -new_len)
                    sliding_audio_buffer[-new_len:] = audio_float

                    text, segments = transcribe_audio(audio_float, current_model_type)
                    if not text or not segments:
                        continue

                    # Refresh embeddings every few seconds using sliding buffer
                    current_time = time.time()
                    if current_time - last_kmeans_time > KMEANS_INTERVAL:
                        wav = preprocess_wav(sliding_audio_buffer)
                        _, embeddings, slices = encoder.embed_utterance(wav, return_partials=True, rate=16)
                        
                        if len(embeddings) == 0:
                            continue

                        n_speakers = estimate_speakers(embeddings)
                        speaker_labels = KMeans(n_clusters=n_speakers).fit_predict(embeddings)
                        speaker_embeddings = embeddings
                        slice_segments = slices

                        # Track consistent speakers
                        new_centroids = [
                            np.mean(speaker_embeddings[speaker_labels == i], axis=0)
                            for i in range(n_speakers)
                        ]

                        speaker_id_map = {}
                        for i, new_c in enumerate(new_centroids):
                            best_match = None
                            best_similarity = -1
                            for old_id, old_c in speaker_centroids.items():
                                sim = np.dot(new_c, old_c) / (np.linalg.norm(new_c) * np.linalg.norm(old_c))
                                if sim > best_similarity:
                                    best_similarity = sim
                                    best_match = old_id
                            if best_similarity > 0.75:
                                speaker_id_map[i] = best_match
                                speaker_centroids[best_match] = new_c
                            else:
                                speaker_counter += 1
                                speaker_id_map[i] = speaker_counter
                                speaker_centroids[speaker_counter] = new_c

                        last_kmeans_time = current_time

                    # Assign each word to speaker
                    speaker_segments = {}  # maps speaker_id to word list

                    for seg in segments:
                        for word in seg.get("words", []):
                            word_start, word_end = word["start"], word["end"]
                            best_overlap, best_label = 0, 0

                            for i, sl in enumerate(slice_segments):
                                slice_start = sl.start / sampling_rate
                                slice_end = sl.stop / sampling_rate
                                overlap = max(0, min(word_end, slice_end) - max(word_start, slice_start))
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_label = speaker_labels[i]

                            speaker_id = speaker_id_map.get(best_label, 0)
                            if speaker_id not in speaker_segments:
                                speaker_segments[speaker_id] = []
                            speaker_segments[speaker_id].append(word["word"])

                    elapsed_time = round(time.time() - recording_start_time, 1)
                    graph_data.append({
                        "time": elapsed_time,
                        "counts": {f"Speaker {sid}": len(words) for sid, words in speaker_segments.items()}
                    })
                    socketio.emit("graph_update", {
                        "time": elapsed_time,
                        "counts": graph_data[-1]["counts"]
                    })

                    # Emit speaker-wise caption blocks
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    for speaker_id, words in speaker_segments.items():
                        caption_text = f"[Speaker {speaker_id}]: {' '.join(words)}"
                        socketio.emit('caption', {
                            'timestamp': timestamp,
                            'text': caption_text,
                            'model': MODEL_CONFIGS[current_model_type]['name']
                        })
                        print(f"🕒 {timestamp} - {caption_text}")

                    # Emit graph data
                    speaker_counts = {
                        f"Speaker {sid}": len(words)
                        for sid, words in speaker_segments.items()
                    }
                    socketio.emit("speaker_stats", speaker_counts)


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

graph_data = []
recording_start_time = None


@socketio.on('start_recording')
def handle_start_recording():
    """Start audio recording and processing"""
    global is_recording, audio_thread
    global graph_data, recording_start_time
    graph_data = []
    recording_start_time = time.time()

    
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