import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO
import time
import threading
import pygame
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import queue

# Custom CSS for better UI
st.set_page_config(
    page_title="Helmet Pose Detection System", 
    page_icon="🛵",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f4e79;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class HelmetPoseNet(nn.Module):
    def __init__(self):
        super(HelmetPoseNet, self).__init__()
        base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(960, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  # Output: yaw, pitch, roll

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_models():
    """Load YOLO and ResNet models with caching"""
    try:
        device = torch.device('cpu')
        
        # Check if model files exist
        yolo_path = 'yolo/v11n/epoch50/best.pt'
        pose_path = 'best_mobilenetv3_finetuned.pth'
        
        if not os.path.exists(yolo_path):
            st.error(f"YOLO model file '{yolo_path}' tidak ditemukan!")
            return None, None, None, None
        
        if not os.path.exists(pose_path):
            st.error(f"Pose model file '{pose_path}' tidak ditemukan!")
            return None, None, None, None
        
        # Load YOLO model
        yolo_model = YOLO(yolo_path)
        
        # Load pose estimation model
        pose_model = HelmetPoseNet().to(device)
        pose_model.load_state_dict(torch.load(pose_path, map_location=device))
        pose_model.eval()
        
        # Define transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        return yolo_model, pose_model, device, transform
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def predict_pose(crop, pose_model, img_transform, device):
    """Predict helmet pose from cropped image using Euler angles"""
    try:
        if crop is None or crop.size == 0:
            return 0.0, 0.0, 0.0, "unknown"
        
        # Convert BGR to RGB (important for proper color processing)
        image_rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Transform the crop
        input_tensor = img_transform(image_rgb_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pose_outputs = pose_model(input_tensor)
            pred_yaw, pred_pitch, pred_roll = pose_outputs.squeeze().tolist()
        
        # Determine pose category based on angles
        pose_category = classify_pose_from_angles(pred_yaw, pred_pitch, pred_roll)
        
        return pred_yaw, pred_pitch, pred_roll, pose_category
    
    except Exception as e:
        print(f"Error in pose prediction: {str(e)}")  # Use print for debugging in video transformer
        return 0.0, 0.0, 0.0, "error"

def classify_pose_from_angles(yaw, pitch, roll):
    """Classify pose category from Euler angles"""
    # Convert to degrees for easier interpretation
    yaw_deg = yaw
    pitch_deg = pitch
    roll_deg = roll
    
    # Define thresholds (adjust based on your model's training)
    pitch_threshold = 15  # degrees
    yaw_threshold = 20    # degrees
    
    # Classify based on pitch (up/down head movement)
    if pitch_deg > pitch_threshold:
        return "looking_up"
    elif pitch_deg < -pitch_threshold:
        return "looking_down"
    elif abs(yaw_deg) > yaw_threshold:
        return "looking_sideways"
    else:
        return "looking_straight"

def get_pose_color(pose_category):
    """Get color for pose visualization"""
    color_map = {
        "looking_straight": (0, 255, 0),    # Green - Safe
        "looking_up": (0, 165, 255),        # Orange - Moderate risk
        "looking_down": (0, 0, 255),        # Red - High risk (phone/distracted)
        "looking_sideways": (0, 255, 255),  # Yellow - Moderate risk
        "unknown": (128, 128, 128),         # Gray
        "error": (255, 0, 255)              # Magenta
    }
    return color_map.get(pose_category, (128, 128, 128))

def initialize_alarm_system():
    """Initialize pygame mixer for alarm sounds"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        return True
    except Exception as e:
        print(f"Warning: Could not initialize alarm system: {str(e)}")
        return False

def play_alarm_beep():
    """Play a simple alarm beep sound"""
    try:
        # Generate a simple beep sound
        duration = 0.3  # seconds
        sample_rate = 22050
        
        # Generate sine wave for beep
        import numpy as np
        frames = int(duration * sample_rate)
        arr = np.zeros(frames)
        
        # Create beep at 800Hz
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * 800 * i / sample_rate) * 0.3
        
        # Convert to pygame sound
        arr = (arr * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(arr)
        sound.play()
        
    except Exception as e:
        print(f"Could not play alarm: {str(e)}")

def check_and_trigger_alarm(pose_category, alarm_enabled=True):
    """Check pose and trigger alarm if distracted"""
    if not alarm_enabled:
        return False
        
    distracted_poses = ['looking_up', 'looking_down', 'looking_sideways']
    
    if pose_category in distracted_poses:
        # Run alarm in separate thread to avoid blocking
        alarm_thread = threading.Thread(target=play_alarm_beep, daemon=True)
        alarm_thread.start()
        return True
    return False

def process_frame(frame, yolo_model, pose_model, img_transform, device):
    """Process a single frame for detection and pose estimation"""
    try:
        # YOLO detection with verbose=False for cleaner output
        yolo_results = yolo_model(frame, verbose=False)
        
        # Initialize statistics
        total_detections = 0
        looking_straight = 0
        distracted = 0
        
        # Process detections - get boxes from first result
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy() if yolo_results[0].boxes is not None else []
        
        if len(boxes) == 0:
            return frame
        
        # Process each detected helmet
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Get confidence if available
            if hasattr(yolo_results[0].boxes, 'conf') and yolo_results[0].boxes.conf is not None:
                conf = yolo_results[0].boxes.conf[i].cpu().numpy()
            else:
                conf = 1.0  # Default confidence
            
            if conf > 0.5:  # Confidence threshold
                total_detections += 1
                
                # Crop the detected helmet region
                helmet_crop = frame[y1:y2, x1:x2]
                
                if helmet_crop.size > 0:
                    # Predict pose using Euler angles
                    yaw, pitch, roll, pose_category = predict_pose(helmet_crop, pose_model, img_transform, device)
                    # alarm_triggered = check_and_trigger_alarm(pose_category, alarm_enabled)

                    # Update statistics
                    if pose_category == 'looking_straight':
                        looking_straight += 1
                    else:
                        distracted += 1
                    
                    # Store latest angles in session state for display
                    if 'latest_angles' not in st.session_state:
                        st.session_state.latest_angles = {}
                    st.session_state.latest_angles = {
                        'yaw': np.degrees(yaw),
                        'pitch': np.degrees(pitch),
                        'roll': np.degrees(roll)
                    }
                    
                    # Get color based on pose
                    color = get_pose_color(pose_category)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Prepare labels with angle information
                    helmet_label = f"Helmet: {conf:.2f}"
                    pose_label = f"Pose: {pose_category.replace('_', ' ').title()}"
                    angle_label = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
                    
                    # Calculate text background size
                    label_height = 80
                    label_width = max(
                        cv2.getTextSize(helmet_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0],
                        cv2.getTextSize(pose_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0],
                        cv2.getTextSize(angle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                    ) + 20
                    
                    # Draw background rectangle for better text visibility
                    cv2.rectangle(frame, (x1, y1-label_height), (x1+label_width, y1), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1-label_height), (x1+label_width, y1), color, 2)
                    
                    # Draw text labels
                    cv2.putText(frame, helmet_label, (x1+5, y1-55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, pose_label, (x1+5, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, angle_label, (x1+5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Add risk indicator
                    risk_level = get_risk_level(pose_category)
                    risk_color = get_risk_color(risk_level)
                    cv2.circle(frame, (x2-20, y1+20), 10, risk_color, -1)
                    cv2.putText(frame, risk_level[0], (x2-25, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store statistics in session state
        if 'detection_stats' not in st.session_state:
            st.session_state.detection_stats = {'total': 0, 'straight': 0, 'distracted': 0}
        
        st.session_state.detection_stats['total'] += total_detections
        st.session_state.detection_stats['straight'] += looking_straight
        st.session_state.detection_stats['distracted'] += distracted
        
        return frame
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")  # Use print for debugging
        return frame

def get_risk_level(pose_category):
    """Get risk level based on pose category"""
    risk_map = {
        "looking_straight": "LOW",
        "looking_up": "MED",
        "looking_down": "HIGH",  # Usually checking phone
        "looking_sideways": "MED",
        "unknown": "UNK",
        "error": "ERR"
    }
    return risk_map.get(pose_category, "UNK")

def get_risk_color(risk_level):
    """Get color for risk level indicator"""
    color_map = {
        "LOW": (0, 255, 0),      # Green
        "MED": (0, 165, 255),    # Orange
        "HIGH": (0, 0, 255),     # Red
        "UNK": (128, 128, 128),  # Gray
        "ERR": (255, 0, 255)     # Magenta
    }
    return color_map.get(risk_level, (128, 128, 128))

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time webcam processing"""
    
    def __init__(self):
        self.yolo_model = None
        self.pose_model = None
        self.img_transform = None  # Renamed from 'transform' to avoid conflict
        self.device = None
        self.frame_count = 0
        
    def load_models(self, yolo_model, pose_model, img_transform, device):
        """Load models into the transformer"""
        self.yolo_model = yolo_model
        self.pose_model = pose_model
        self.img_transform = img_transform  # Use the renamed attribute
        self.device = device
    
    def transform(self, frame):
        """Transform each frame from the webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        if self.yolo_model is not None and self.pose_model is not None:
            # Process frame
            processed_img = process_frame(img, self.yolo_model, self.pose_model, self.img_transform, self.device)
            self.frame_count += 1
            
            # Add frame counter and FPS info
            fps_text = f"Frame: {self.frame_count} | Processing: Active"
            cv2.putText(processed_img, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add title overlay
            cv2.putText(processed_img, "Helmet Pose Detection - Live", (10, processed_img.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            return processed_img
        else:
            # Return original frame if models not loaded
            cv2.putText(img, "Loading models...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img

# Global video transformer instance
video_transformer = VideoTransformer()

def process_video(video_path, yolo_model, pose_model, img_transform, device, progress_bar, status_text):
    """Process uploaded video file with web-compatible output"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure even dimensions for H.264 compatibility
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        
        # Create temporary output file with web-compatible settings
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Use H.264 codec for better web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        # Alternative codecs to try if avc1 doesn't work:
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if VideoWriter is properly initialized
        if not out.isOpened():
            st.error("Failed to initialize video writer. Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                st.error("Failed to initialize video writer with alternative codec.")
                return None
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if dimensions were adjusted
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Process frame
            processed_frame = process_frame(frame, yolo_model, pose_model, img_transform, device)
            
            # Ensure processed frame has correct dimensions
            if processed_frame.shape[1] != width or processed_frame.shape[0] != height:
                processed_frame = cv2.resize(processed_frame, (width, height))
            
            out.write(processed_frame)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error("Output video file is empty or corrupted.")
            return None
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        return None

# Alternative solution using base64 encoding for video display
def display_video_with_base64(video_path):
    """Alternative method to display video using base64 encoding"""
    try:
        import base64
        
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode()
        
        video_html = f"""
        <video width="100%" height="400" controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        
        st.markdown(video_html, unsafe_allow_html=True)
        return True
        
    except Exception as e:
        st.error(f"Error displaying video with base64: {str(e)}")
        return False

def main():
    # Header
    st.markdown('<div class="main-header">Helmet Pose Detection System </div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">HELMET POSE ESTIMATION UNTUK DETEKSI VISUAL DISTRACTED DRIVING MENGGUNAKAN YOLOV8 DAN RESNET18</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        yolo_model, pose_model, device, img_transform = load_models()
    
    # alarm_available = initialize_alarm_system()
    # if alarm_available:
    #     st.sidebar.success("🔊 Alarm system ready")
    # else:
    #     st.sidebar.warning("⚠️ Alarm system unavailable")
    
    if yolo_model is None or pose_model is None:
        st.markdown('<div class="status-box error-box">❌ Failed to load models. Please check if model files exist.</div>', unsafe_allow_html=True)
        st.stop()
    
    st.markdown('<div class="status-box success-box">✅ Models loaded successfully!</div>', unsafe_allow_html=True)
    
    # Sidebar for options
    st.sidebar.title("🎛️ Detection Options")
    
    detection_mode = st.sidebar.selectbox(
        "Choose detection mode:",
        ["📹 Live Webcam", "📁 Upload Video"]
    )

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("### 🚨 Alarm Settings")

    # alarm_enabled = st.sidebar.checkbox("Enable Distraction Alarm", value=True)

    # if alarm_enabled:
    #     st.sidebar.info("🔊 Alarm will sound for:\n- Looking Down\n- Looking Up\n- Looking Sideways")
    # else:
    #     st.sidebar.info("🔇 Alarm disabled")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info("""
    **YOLO Model**: Helmet Detection
    **ResNet Model**: Pose Estimation
    **Poses**: Looking Down, Straight, Up
    **Device**: CPU
    """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if detection_mode == "📹 Live Webcam":
            st.subheader("🔴 Live Webcam Detection")
            
            # Initialize statistics
            if 'detection_stats' not in st.session_state:
                st.session_state.detection_stats = {'total': 0, 'straight': 0, 'distracted': 0}
            
            # Reset statistics button
            if st.button("🔄 Reset Statistics"):
                st.session_state.detection_stats = {'total': 0, 'straight': 0, 'distracted': 0}
                st.success("Statistics reset!")
            
            # Load models into video transformer
            video_transformer.load_models(yolo_model, pose_model, img_transform, device)
            
            # WebRTC configuration for better connectivity
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            st.info("📌 Click 'START' to begin webcam detection. Make sure to allow camera access when prompted.")
            
            # Start webcam stream
            ctx = webrtc_streamer(
                key="helmet-detection",
                video_transformer_factory=lambda: video_transformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 640, "ideal": 1280, "max": 1920},
                        "height": {"min": 480, "ideal": 720, "max": 1080},
                        "frameRate": {"min": 15, "ideal": 30, "max": 60}
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            # Display status
            if ctx.state.playing:
                st.success("🟢 Webcam is running - Detecting helmets and poses in real-time!")
                
                # Live statistics display
                stats_placeholder = st.empty()
                
                # Update statistics display every few seconds
                if st.session_state.detection_stats['total'] > 0:
                    with stats_placeholder.container():
                        st.markdown("### 📊 Live Detection Statistics")
                        col1_stats, col2_stats, col3_stats = st.columns(3)
                        
                        with col1_stats:
                            st.metric("Total Detections", st.session_state.detection_stats['total'])
                        with col2_stats:
                            st.metric("Looking Straight", st.session_state.detection_stats['straight'])
                        with col3_stats:
                            st.metric("Distracted", st.session_state.detection_stats['distracted'])
                        
                        # Calculate percentages
                        if st.session_state.detection_stats['total'] > 0:
                            straight_pct = (st.session_state.detection_stats['straight'] / st.session_state.detection_stats['total']) * 100
                            distracted_pct = (st.session_state.detection_stats['distracted'] / st.session_state.detection_stats['total']) * 100
                            
                            st.progress(straight_pct / 100)
                            st.caption(f"Safe Driving: {straight_pct:.1f}% | Distracted: {distracted_pct:.1f}%")
            
            elif ctx.state.signalling:
                st.warning("🟡 Connecting to webcam...")
            else:
                st.info("🔵 Click START to begin webcam detection")
                
                # Instructions
                with st.expander("📋 Webcam Setup Instructions"):
                    st.markdown("""
                    1. **Click the START button** above
                    2. **Allow camera access** when prompted by your browser
                    3. **Position yourself** so your helmet/head is visible
                    4. **The system will detect** helmets and estimate head poses in real-time
                    5. **Green boxes** = Normal driving position
                    6. **Red boxes** = Distracted/dangerous position
                    """)
        
        elif detection_mode == "📁 Upload Video":
            st.subheader("📤 Video Upload Detection")
        
            uploaded_file = st.file_uploader(
                "Choose a video file", 
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for helmet pose detection"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_video_path = tmp_file.name
                
                # Display video info
                st.success(f"✅ Video uploaded: {uploaded_file.name}")
                
                # Show original video
                with st.expander("📹 Original Video Preview", expanded=False):
                    st.video(temp_video_path)
                
                # Process video button
                if st.button("🚀 Process Video"):
                    with st.spinner("Processing video..."):
                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Process the video
                        output_path = process_video(
                            temp_video_path, 
                            yolo_model, 
                            pose_model, 
                            img_transform, 
                            device,
                            progress_bar,
                            status_text
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if output_path and os.path.exists(output_path):
                            st.success("✅ Video processing completed!")
                            
                            # Check file size
                            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                            st.info(f"Processed video size: {file_size:.2f} MB")
                            
                            # Display processed video with error handling
                            st.subheader("🎬 Processed Video Result")
                            
                            try:
                                # Method 1: Direct video display
                                st.video(output_path)
                                
                            except Exception as video_error:
                                st.error(f"Error displaying video: {str(video_error)}")
                                st.info("The video was processed successfully but cannot be displayed in the browser. You can still download it.")
                            
                            # Alternative: Convert to web-compatible format using ffmpeg (if available)
                            # This requires ffmpeg to be installed
                            web_compatible_path = None
                            try:
                                import subprocess
                                web_compatible_path = tempfile.NamedTemporaryFile(delete=False, suffix='_web.mp4').name
                                
                                # Convert to web-compatible format
                                cmd = [
                                    'ffmpeg', '-i', output_path,
                                    '-c:v', 'libx264',
                                    '-preset', 'fast',
                                    '-crf', '23',
                                    '-c:a', 'aac',
                                    '-movflags', '+faststart',
                                    '-y', web_compatible_path
                                ]
                                
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                
                                if result.returncode == 0 and os.path.exists(web_compatible_path):
                                    st.info("🔄 Created web-compatible version")
                                    st.video(web_compatible_path)
                                
                            except (ImportError, FileNotFoundError, subprocess.SubprocessError):
                                st.warning("FFmpeg not available for format conversion. Using original processed video.")
                            except Exception as ffmpeg_error:
                                st.warning(f"FFmpeg conversion failed: {str(ffmpeg_error)}")
                            
                            # Provide download link
                            with open(output_path, 'rb') as file:
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=file.read(),
                                    file_name=f"processed_{uploaded_file.name}",
                                    mime="video/mp4",
                                    help="Download the processed video file"
                                )
                            
                            # Video statistics
                            try:
                                cap = cv2.VideoCapture(output_path)
                                if cap.isOpened():
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                    duration = frame_count / fps if fps > 0 else 0
                                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    cap.release()
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Duration", f"{duration:.1f}s")
                                    with col2:
                                        st.metric("FPS", f"{fps:.1f}")
                                    with col3:
                                        st.metric("Resolution", f"{width}x{height}")
                                    with col4:
                                        st.metric("Frames", f"{int(frame_count)}")
                            except Exception as stats_error:
                                st.warning(f"Could not read video statistics: {str(stats_error)}")
                            
                            # Cleanup
                            if web_compatible_path and os.path.exists(web_compatible_path):
                                os.unlink(web_compatible_path)
                            os.unlink(output_path)
                            
                        else:
                            st.error("❌ Error processing video - output file not created")
                
                # Cleanup temporary input file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
    
    with col2:
        st.subheader("ℹ️ Information")
        
        st.markdown("""
        ### 🎯 Detection Classes
        - **Looking Straight**: Normal driving posture (LOW risk)
        - **Looking Down**: Checking phone/dashboard (HIGH risk)
        - **Looking Up**: Checking mirrors/signs (MED risk)
        - **Looking Sideways**: Side distraction (MED risk)
        
        ### 🔧 How it works:
        1. **YOLO** detects helmets in the frame
        2. **ResNet18** estimates head pose using Euler angles
        3. **Yaw, Pitch, Roll** angles determine pose category
        4. **Risk levels** are assigned and color-coded
        
        ### 📐 Angle Information:
        - **Yaw**: Left/Right head rotation
        - **Pitch**: Up/Down head tilt
        - **Roll**: Head lean angle
        
        ### 📋 Instructions:
        - Choose your input method from sidebar
        - For webcam: Click start and allow camera access
        - For video: Upload file and click process
        - Watch for risk indicators (L/M/H circles)
        - Download processed results
        """)
        
        # Real-time angle display for webcam
        if detection_mode == "📹 Live Webcam" and 'latest_angles' in st.session_state:
            st.markdown("### 📐 Latest Detected Angles")
            angles = st.session_state.latest_angles
            col1_angle, col2_angle, col3_angle = st.columns(3)
            
            with col1_angle:
                st.metric("Yaw", f"{angles['yaw']:.1f}°", help="Left/Right rotation")
            with col2_angle:
                st.metric("Pitch", f"{angles['pitch']:.1f}°", help="Up/Down tilt")
            with col3_angle:
                st.metric("Roll", f"{angles['roll']:.1f}°", help="Head lean")
        
        # Statistics (live updates during webcam)
        if detection_mode == "📹 Live Webcam":
            st.markdown("### 📈 Real-time Statistics")
            if 'detection_stats' in st.session_state:
                stats = st.session_state.detection_stats
                
                # Display current session stats
                st.metric("Session Detections", stats['total'])
                st.metric("Normal Posture", stats['straight'])
                st.metric("Distracted Posture", stats['distracted'])
                
                # Safety indicator
                if stats['total'] > 0:
                    safety_score = (stats['straight'] / stats['total']) * 100
                    if safety_score >= 80:
                        st.success(f"🟢 Safety Score: {safety_score:.1f}%")
                    elif safety_score >= 60:
                        st.warning(f"🟡 Safety Score: {safety_score:.1f}%")
                    else:
                        st.error(f"🔴 Safety Score: {safety_score:.1f}%")
            else:
                st.info("Start webcam to see live statistics")
        
        elif st.checkbox("Show Detection Statistics"):
            st.markdown("### 📈 Statistics")
            st.metric("Detections", "0")
            st.metric("Looking Straight", "0")
            st.metric("Distracted", "0")

if __name__ == "__main__":
    main()