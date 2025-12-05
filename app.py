# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# from PIL import Image

# # Load model
# model = tf.keras.models.load_model("plant_disease_model.h5")

# # Class names (अपने dataset के अनुसार बदलें)
# class_names =['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# st.title("🌱 Plant Disease Detection App")

# # Image upload
# uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).resize((256,256))
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess
#     img_array = np.array(image)/255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     # Prediction
#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = round(100 * np.max(prediction), 2)

#     st.write(f"### ✅ Predicted Class: {predicted_class}")
#     st.write(f"### 🔎 Confidence: {confidence}%")

# # Performance Evaluation Section
# st.header("📊 Model Performance (Validation & Test)")

# # Dummy values (replace with your confusion matrix results)
# val_total, val_correct, val_wrong, val_acc = 100, 85, 15, 0.85
# test_total, test_correct, test_wrong, test_acc = 120, 110, 10, 0.91

# labels = ['Total Samples','Correct Predictions','Wrong Predictions','Accuracy (%)']
# val_values = [val_total, val_correct, val_wrong, val_acc*100]
# test_values = [test_total, test_correct, test_wrong, test_acc*100]

# x = np.arange(len(labels))
# width = 0.35

# fig, ax = plt.subplots(figsize=(10,6))
# rects1 = ax.bar(x - width/2, val_values, width, label='Validation', color='orange')
# rects2 = ax.bar(x + width/2, test_values, width, label='Test', color='green')

# ax.set_ylabel('Count / Percentage')
# ax.set_title('Validation vs Test Performance Comparison')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45)
# ax.legend()

# # Annotate bars
# def annotate_bars(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.2f}',
#                     xy=(rect.get_x() + rect.get_width()/2, height),
#                     xytext=(0,3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# annotate_bars(rects1)
# annotate_bars(rects2)

# st.pyplot(fig)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
# matplotlib is imported but not used, so it's kept as in original

# --- 1. Configuration and Model Loading ---

# Function to get the current theme mode
def get_current_theme():
    """Tries to infer the current Streamlit theme mode."""
    return st.get_option("theme.base")

# Page Configuration - Changed initial_sidebar_state to 'auto' or 'expanded'
st.set_page_config(
    page_title="Professional Potato Disease Analyzer",
    page_icon="🥔",
    layout="wide",
    # *** CHANGE MADE HERE ***
    initial_sidebar_state="expanded" # Changed from 'collapsed' to make the theme switcher visible
)

# --- Global Configuration ---
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = (256, 256) # Model input size

# Load model and cache it
@st.cache_resource
def load_model():
    """Loads and caches the model."""
    try:
        # Ensure 'plant_disease_model.h5' is in the same directory
        model = tf.keras.models.load_model("plant_disease_model.h5")
        st.sidebar.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        # Added a global warning if model fails to load, in case sidebar is hidden
        st.error(f"❌ Application Error: Could not load the required machine learning model. Check console for details: {e}")
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

MODEL = load_model()

# --- 2. Helper Functions (No changes needed) ---

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepares the image for model input."""
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_recommendation(disease: str) -> str:
    """Provides specific nursery recommendations based on the disease."""
    if 'early_blight' in disease.lower():
        return "Immediately apply fungicides such as Chlorothalonil. Control irrigation and remove infected leaves to limit spore spread."
    elif 'late_blight' in disease.lower():
        return "The disease spreads rapidly. Use potent systemic fungicides containing Mefenoxam or Azoxystrobin. Destroy all infected plants immediately."
    else: # Healthy
        return "The plant is perfectly healthy! Maintain regular monitoring, balanced nutrition, and ensure proper airflow."

def predict_and_display(uploaded_file):
    """Performs prediction and returns results after a file is uploaded."""
    progress_bar = st.progress(0, text="Starting image processing...")
    time.sleep(0.5)

    try:
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)

        progress_bar.progress(30, text="Running model prediction...")
        time.sleep(0.5)

        if MODEL is None:
            st.error("Model not available. Please check the 'plant_disease_model.h5' file.")
            progress_bar.empty()
            return

        prediction = MODEL.predict(img_array)
        predicted_index = np.argmax(prediction)

        progress_bar.progress(70, text="Analyzing results...")
        time.sleep(0.5)

        predicted_class_full = CLASS_NAMES[predicted_index]
        confidence = round(100 * np.max(prediction), 2)

        disease_name = predicted_class_full.split('___')[-1].replace('_', ' ').title()

        progress_bar.progress(100, text="Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()

        return image, disease_name, confidence

    except Exception as e:
        st.error(f"A serious error occurred during analysis: {e}")
        progress_bar.empty()
        return None, None, None

# --- 3. Main UI Layout Function ---

def main_dashboard():
    """Renders the Streamlit main dashboard UI, now with an outer border."""
    
    # *** START BORDER DIV ***
    # Use st.markdown to open a custom div with border styling
    # The 'id="main-app-container"' is used by the CSS injected below to apply the border.
    st.markdown('<div id="main-app-container">', unsafe_allow_html=True)
    
    # --- Header ---
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🥔 Nursery Potato Disease Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Keep your potato plants healthy. Upload a leaf image and get an instant diagnosis.</p>", unsafe_allow_html=True)

    # --- Day/Night Status in Sidebar ---
    current_theme = get_current_theme()
    theme_icon = "🌙 Night Mode" if current_theme == "dark" else "☀️ Day Mode"
    st.sidebar.markdown(f"### {theme_icon}")
    st.sidebar.caption("Change the theme using the 'Settings' menu above or in the sidebar.")
    # --- END ADDITION ---

    st.markdown("---")

    # --- Input Function ---
    uploaded_file = st.file_uploader("Please upload a potato leaf image (JPG/PNG)", type=["jpg", "png", "jpeg"], help="Use a clear, well-lit image for best results.")

    if uploaded_file is None:
        # Initial screen: Centered input
        st.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)
        
        # *** END BORDER DIV (for the case where no file is uploaded) ***
        st.markdown('</div>', unsafe_allow_html=True) 
        return

    # --- Split Screen After Upload ---
    col_input, col_output = st.columns([1, 1], gap="large")

    # --- Left Column: Input and Image ---
    with col_input:
        st.subheader("🖼️ Uploaded Leaf Image")

        # Perform prediction
        image, disease_name, confidence = predict_and_display(uploaded_file)

        if image:
            # Display image with reduced size (max width set to 350px for better fit)
            st.image(image, caption="Image for Analysis", use_column_width='auto', width=450)
            st.markdown("---")
            st.subheader("Additional Tools")
            st.button("Analyze New Image", help="Resets the uploader.", key='new_analysis')


    # --- Right Column: Output and Analysis ---
    if disease_name and confidence is not None:

        with col_output:
            st.subheader("🔬 Disease Diagnosis and Recommendation")

            # --- Result Card ---
            if 'healthy' in disease_name.lower():
                icon = "✅"
                color = "#4CAF50" # Green
                st.balloons()
            else:
                icon = "🚨"
                color = "#FF9800" # Orange/Yellow

            # Note: Background color is fixed here, may look odd in dark mode.
            st.markdown(f"""
            <div style='border: 2px solid {color}; padding: 15px; border-radius: 10px; background-color: #f9f9f9;'>
                <h3 style='color: {color}; margin-top: 0;'>{icon} Diagnosis: {disease_name.upper()}</h3>
                <p><strong>Confidence Score:</strong></p>
                <div style='font-size: 24px; font-weight: bold; color: {color};'>{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence), text=f"Model Confidence: {confidence:.2f}%")
            st.markdown("---")

            # --- Recommendation Card ---
            recommendation = get_recommendation(disease_name)

            st.subheader("📝 Nursery Action Plan")
            st.info(f"""
            **Immediate Action Required:** {recommendation}
            """)

            st.markdown("---")

            # --- Model Performance Metrics ---
            st.subheader("📊 Model Performance Summary")

            # Dummy values - Replace with your actual training metrics
            val_acc = 0.88
            test_acc = 0.92

            colA, colB = st.columns(2)
            colA.metric("Validation Accuracy", f"{val_acc*100:.2f}%")
            colB.metric("Test Accuracy", f"{test_acc*100:.2f}%", delta=f"{test_acc*100 - val_acc*100:.2f}%")

            st.caption("Note: Model performance metrics are based on pre-training data.")
    
    # *** END BORDER DIV ***
    st.markdown('</div>', unsafe_allow_html=True)


# --- Run App ---
if __name__ == "__main__":
    # Inject CSS for initial centering and better styling AND THE BORDER
    st.markdown("""
        <style>
        /* CSS to apply the border to the main container */
        #main-app-container {
            border: 3px solid #66b3ff; /* Blue border line */
            border-radius: 15px; /* Rounded corners for a modern look */
            padding: 30px; /* Space inside the border */
            margin-top: 20px; /* Space above the bordered box */
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        /* Centering the initial file uploader */
        .stFileUploader {
            padding-top: 50px;
            margin: auto;
            width: 80%;
            text-align: center;
        }
        /* Styling for a professional look */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)
    main_dashboard()