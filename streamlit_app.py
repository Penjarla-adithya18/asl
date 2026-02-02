import streamlit as st
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="ASL Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Title and description
st.title("🤟 ASL Sign Language Recognition")
st.markdown("Upload an image of an ASL hand sign to recognize the letter")

# Flask API URL - Using Render.com for cloud deployment
FLASK_API_URL = "https://asl-10.onrender.com/predict"

# Initialize session state for sentence
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Capture or Upload Image")
    
    # Create tabs for camera and upload options
    tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload"])
    
    image = None
    
    with tab1:
        # Camera input
        camera_photo = st.camera_input("Take a picture of ASL hand sign")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.image(image, caption="Captured Image", use_column_width=True)
    
    with tab2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image of ASL hand sign",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of an ASL hand sign (A-Z)"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button (shown if any image is available)
    if image is not None:
        if st.button("🔍 Recognize Sign", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Prepare image for API request
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    # Send request to Flask API (with longer timeout for Render free tier)
                    files = {'image': ('image.png', img_byte_arr, 'image/png')}
                    response = requests.post(FLASK_API_URL, files=files, timeout=150)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Check if there's an error (no hand detected)
                        if 'error' in result:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")
                        elif 'letter' in result:
                            predicted_letter = result.get('letter', 'Unknown')
                            confidence = result.get('confidence', 0.0) / 100  # Convert to decimal
                            
                            # Display results in col2
                            with col2:
                                st.subheader("✅ Recognition Result")
                                
                                # Display letter in large font
                                st.markdown(f"<h1 style='text-align: center; color: #4CAF50; font-size: 100px;'>{predicted_letter}</h1>", 
                                          unsafe_allow_html=True)
                                
                                # Display confidence with progress bar
                                st.metric("Confidence", f"{confidence:.2%}")
                                st.progress(confidence)
                                
                                # Add to sentence button
                                col_add, col_space, col_clear = st.columns(3)
                                
                                with col_add:
                                    if st.button("➕ Add to Sentence", use_container_width=True):
                                        st.session_state.sentence += predicted_letter
                                        st.rerun()
                                
                                with col_space:
                                    if st.button("␣ Add Space", use_container_width=True):
                                        st.session_state.sentence += " "
                                        st.rerun()
                                
                                with col_clear:
                                    if st.button("🗑️ Clear", use_container_width=True):
                                        st.session_state.sentence = ""
                                        st.rerun()
                        else:
                            st.error("❌ Unexpected response format")
                    else:
                        st.error(f"❌ Server error: {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timeout - Flask server is taking too long to respond")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection error - Make sure Flask server is running on " + FLASK_API_URL)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Display sentence
if st.session_state.sentence:
    st.divider()
    st.subheader("📝 Built Sentence")
    st.markdown(f"<h2 style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>{st.session_state.sentence}</h2>", 
                unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    ### How to Use:
    1. **Upload** an image of an ASL hand sign
    2. Click **Recognize Sign** button
    3. View the predicted letter and confidence
    4. **Add to Sentence** to build words
    5. Use **Clear** to start over
    
    ### Tips:
    - Use clear, well-lit images
    - Ensure hand is visible and centered
    - Try different angles if recognition fails
    - Confidence > 60% is usually accurate
    
    ### Supported:
    - Letters: A-Z (26 signs)
    - Model: MobileNetV2 Transfer Learning
    - Accuracy: 94% validation
    """)
    
    st.divider()
    
    # API status check
    st.subheader("🔌 API Status")
    try:
        health_response = requests.get("http://10.151.213.150:5000/health", timeout=5)
        if health_response.status_code == 200:
            data = health_response.json()
            st.success("✅ Flask API is running")
            st.json(data)
        else:
            st.error("❌ Flask API error")
    except:
        st.error("❌ Flask API not reachable")
        st.info(f"Make sure Flask is running on:\n`{FLASK_API_URL}`")
