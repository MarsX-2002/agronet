import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AgroNet")

# Custom CSS for a professional look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e9ecef;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        border-bottom: 2px solid #28a745;
    }
    /* Fix Chatbot Visibility */
    .stChatMessage {
        background-color: #343a40;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage p {
        color: #ffffff !important;
    }
    /* User message specific style if needed, but above handles general contrast */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- AI Advisor Setup ---
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("GEMINI_API_KEY not found in .env. AI features will be disabled.")

# --- Helper Functions ---
@st.cache_resource
def load_model(model_path):
    """Loads and caches the YOLO model to avoid reloading on every rerun."""
    return YOLO(model_path)

from PIL import Image
import numpy as np

def get_ai_response(messages, context_data, context_image=None):
    """Conversational AI with context and optional image."""
    if not API_KEY:
        return "AI Advisor is unavailable (Missing API Key)."
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Construct context from detection data
        context_str = f"Current Session Stats: {context_data}" if context_data else "No active detection data."
        
        # Build prompt from history
        chat_history = []
        # Add system context
        system_instruction = f"You are an agricultural safety expert. Context: {context_str}. Analyze the provided image (if any) and stats. Answer concise and helpful."
        chat_history.append(system_instruction)
        
        input_payload = [system_instruction]
        
        # Add Image Context (if available)
        # context_image tells us general scene, but context_data might have ID-specific images
        # We can try to find relevant images if the user asks about a specific ID.
        # For now, let's attach the LATEST image if available.
        if context_image is not None:
             input_payload.append(context_image)
        
        input_payload.append(f"User Question: {messages[-1]['content']}")

        response = model.generate_content(input_payload)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def process_stream(product_name, model_path, defect_classes):
    """
    Main processing pipeline for a specific product tab.
    """
    st.header(f"{product_name} Inspection Station")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file not found: `{model_path}`")
        st.info("Please run the `train_models.ipynb` notebook to train the models first.")
        return

    # Load Model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # File Uploader
    video_file = st.file_uploader(f"Upload {product_name} Belt Feed", type=['mp4', 'mov', 'avi'], key=f"d_{product_name}")
    
    if video_file:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # UI Layout
        col1, col2 = st.columns([3, 1])
        with col1:
            st_frame = st.empty()
        
        with col2:
            st.markdown("### Control Panel")
            stop_button = st.button("🔴 Stop Stream", key=f"stop_{product_name}")
            metrics_placeholder = st.empty()
            
            # Setup Video Writer
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Confidence Control
            conf_threshold = st.slider("Detection Confidence", min_value=0.05, max_value=1.0, value=0.15, key=f"conf_{product_name}")
            
            # Use temp file for output
            tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v is widely supported for .mp4 containers
            out_writer = cv2.VideoWriter(tfile_out.name, fourcc, fps, (width, height))

        # Session State Keys for Context
        ss_log_key = f"log_{product_name}"
        ss_img_key = f"img_{product_name}"          # Latest Image (for generic context)
        ss_id_frames_key = f"id_frames_{product_name}" # Map: ID -> {image, frame_num}
        
        if ss_log_key not in st.session_state:
            st.session_state[ss_log_key] = {}
        if ss_id_frames_key not in st.session_state:
            st.session_state[ss_id_frames_key] = {}
        
        # Initializing Chat History per product tab if not exists
        if f"messages_{product_name}" not in st.session_state:
            st.session_state[f"messages_{product_name}"] = []

        with col2:
            st.markdown("### 🤖 AI Assistant")
            
            # Chat Container
            chat_container = st.container(height=400)
            
            # Display History
            with chat_container:
                for msg in st.session_state[f"messages_{product_name}"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
            
            prompt = st.chat_input("Ask about the batch...", key=f"chat_{product_name}")
            if prompt:
                # Add user message
                st.session_state[f"messages_{product_name}"].append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Prepare Context
                    current_log = st.session_state.get(ss_log_key, {})
                    last_img = st.session_state.get(ss_img_key, None)
                    
                    # Get AI Response
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing data & image..."):
                            response = get_ai_response(
                                st.session_state[f"messages_{product_name}"], 
                                current_log,
                                last_img
                            )
                            st.write(response)
                
                # Add assistant message
                st.session_state[f"messages_{product_name}"].append({"role": "assistant", "content": response})
                st.rerun()

        defect_log_local = {} # Local tracker for verification

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run Inference
            # persist=True enables tracking (ID assignment)
            results = model.track(frame, persist=True, verbose=False, conf=conf_threshold)
            
            # Plot results on frame
            res_plotted = results[0].plot()
            
            # Process Detections
            current_counts = {}
            # Process Detections
            current_counts = {}
            
            for result in results:
                # Extract IDs if available
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [None] * len(result.boxes)

                for box, track_id in zip(result.boxes, track_ids):
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    # Log if it's a defect
                    if cls_name in defect_classes and track_id is not None:
                        # Initialize set for this class if needed
                        if cls_name not in defect_log_local:
                            defect_log_local[cls_name] = set()
                        
                        # Add tracking ID
                        defect_log_local[cls_name].add(int(track_id))
                        
                        # Update Session State Context
                        # We accumulate ID sets in session state
                        # Convert to list for session state storage (sets aren't always serialization friendly if we expand to intricate states, but fine usually)
                        # Actually, let's update a serializable dict for the AI context
                        
                        if ss_log_key not in st.session_state:
                             st.session_state[ss_log_key] = {}
                        
                        # We need to retrieve the existing set/list from session, convert, add, save back
                        # Simplified: Just update the session state with the current local set for this run
                        # This avoids complexity of merging across reruns if we just care about "this video's batch"
                        st.session_state[ss_log_key] = {k: list(v) for k, v in defect_log_local.items()}
                        
                        # Save Frame for Visual Context (Capture the most recent defect or maybe specific one?)
                        # Saving the LATEST frame for now.
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        st.session_state[ss_img_key] = pil_img
                        
                        # Save ID-Specific Frame (First capture preferred to avoid blur, or update?)
                        # Let's verify if we already have this ID stored in session state
                        current_id_frames = st.session_state.get(ss_id_frames_key, {})
                        
                        # If ID not captured yet, store it
                        # If ID not captured yet, store it
                        if int(track_id) not in current_id_frames:
                            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            
                            # CROP THE DEFECT (ROI)
                            # Get box coordinates (x1, y1, x2, y2)
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = xyxy
                            
                            # Ensure bounds
                            h, w, _ = frame.shape
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            
                            # Crop and Convert
                            if x2 > x1 and y2 > y1:
                                crop_bgr = frame[y1:y2, x1:x2]
                                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                                pil_crop = Image.fromarray(crop_rgb)
                                
                                current_id_frames[int(track_id)] = {
                                    "image": pil_crop, 
                                    "frame": frame_num,
                                    "class": cls_name
                                }
                                st.session_state[ss_id_frames_key] = current_id_frames
                    
                    current_counts[cls_name] = current_counts.get(cls_name, 0) + 1

            # Update Frame
            # Update Frame
            st_frame.image(res_plotted, channels="BGR", use_container_width=True)
            
            # Write to output video
            out_writer.write(res_plotted)
            
            # Update Live Metrics
            # We move metrics to overlay or below video to save sidebar for chat
            # using col1 overlay or just small text
            pass 

        cap.release()
        out_writer.release()
        
        # Post-Processing Report
        with col2:
            st.success("Analysis Complete")
            st.subheader("Batch Summary")
            
            # Display nicely formatted stats with Gallery
            if ss_id_frames_key in st.session_state:
                id_frames = st.session_state[ss_id_frames_key]
            else:
                id_frames = {}

            for cls_name, ids in defect_log_local.items():
                st.write(f"**{cls_name}**: {len(ids)} items")
                with st.expander(f"See Gallery for {cls_name}"):
                    # Create a gallery of images
                    sorted_ids = sorted(list(ids))
                    cols = st.columns(3) # Grid layout
                    for idx, track_id in enumerate(sorted_ids):
                        with cols[idx % 3]:
                            st.caption(f"ID: {track_id}")
                            if track_id in id_frames:
                                img_data = id_frames[track_id]
                                st.image(img_data["image"], caption=f"Frame: {img_data['frame']}", use_container_width=True)
                            else:
                                st.write("No Image")
            
            # Download Button
            with open(tfile_out.name, 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_{product_name}.mp4",
                    mime='video/mp4'
                )
            
            if not defect_log_local:
                st.info("No defects detected in this batch.")

# --- Main Layout ---
st.title("🚜 AgroNet")
st.markdown("Automated Quality Control & Safety System")

# Function to clear GPU memory between tabs logic is implicit as Streamlit reruns script
# but we segregate via tabs.

tab_potato, tab_carrot, tab_lemon = st.tabs([
    "🥔 Potato Station", 
    "🥕 Carrot Station", 
    "🍋 Lemon Station"
])

# Define Configuration per Tab
# Note: Paths assume the standard YOLOv8/11 training output structure
# If 'project' was 'runs/detect' and 'name' was 'potato_v11', weights are in 'runs/detect/potato_v11/weights/best.pt'

with tab_potato:
    process_stream(
        product_name="Potato",
        model_path="detect/potato_v11/weights/best.pt",
        defect_classes=["Potato_Damaged", "Potato_Fungal", "Potato_Sprouted"]
    )

with tab_carrot:
    process_stream(
        product_name="Carrot",
        model_path="detect/carrot_v11/weights/best.pt",
        defect_classes=["Carrot_Damaged"] # Healthy is not a defect
    )

with tab_lemon:
    process_stream(
        product_name="Lemon",
        model_path="detect/lemon_v11/weights/best.pt",
        defect_classes=["Lemon_Damaged", "Lemon_Unripe"] # Assuming Unripe is an issue for premium sorting
    )

# Sidebar Removed as per user request
