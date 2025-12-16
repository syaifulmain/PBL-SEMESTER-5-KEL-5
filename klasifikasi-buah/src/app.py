"""
DUAL MODEL SYSTEM: Fast Mode vs Accurate Mode
- Fast Mode: Tanpa background removal, fitur lebih sedikit (~0.1s/image)
- Accurate Mode: Dengan background removal, fitur lengkap (~2s/image)
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import json
import os
from skimage.feature import graycomatrix, graycoprops, hog
from PIL import Image
import time

# ==========================================
# 1. KONFIGURASI & PATH
# ==========================================
st.set_page_config(
    page_title="Klasifikasi Buah - Dual Mode", 
    page_icon="🍎",
    layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(current_dir, 'Output_Artifacts')

# PERBAIKAN: Dua set model
MODELS = {
    'fast': {
        'name': '⚡ Fast Mode',
        'model_path': os.path.join(ARTIFACTS_PATH, 'svm_fast_model.pkl'),
        'scaler_path': os.path.join(ARTIFACTS_PATH, 'scaler_fast.pkl'),
        'label_encoder_path': os.path.join(ARTIFACTS_PATH, 'label_encoder_fast.pkl'),
        'description': 'Cepat (~0.1s) - Tanpa background removal',
        'icon': '⚡',
        'color': 'blue'
    },
    'accurate': {
        'name': '🎯 Accurate Mode',
        'model_path': os.path.join(ARTIFACTS_PATH, 'svm_accurate_model.pkl'),
        'scaler_path': os.path.join(ARTIFACTS_PATH, 'scaler_accurate.pkl'),
        'label_encoder_path': os.path.join(ARTIFACTS_PATH, 'label_encoder_accurate.pkl'),
        'description': 'Akurat (~2s) - Dengan background removal',
        'icon': '🎯',
        'color': 'green'
    }
}

# ==========================================
# 2. PREPROCESSING - FAST MODE
# ==========================================

def preprocess_image_fast(pil_image, target_size=(100, 100)):
    """
    FAST MODE: Tanpa background removal
    """
    try:
        img_array = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        return img_rgb, img_gray
    except Exception as e:
        st.error(f"Error fast preprocessing: {e}")
        return None, None


def extract_color_features_fast(image_rgb):
    """FAST: Bins lebih kecil [6,6,6] = 216 fitur"""
    try:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [6, 6, 6], 
                           [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist
    except:
        return np.zeros(216)


def extract_texture_features_fast(image_gray):
    """FAST: 2 sudut saja, 4 fitur"""
    try:
        if image_gray.dtype != np.uint8:
            image_gray = (image_gray * 255).astype(np.uint8)
        
        img_quantized = (image_gray // 8).astype(np.uint8)
        glcm = graycomatrix(img_quantized, distances=[1], 
                           angles=[0, np.pi/2], 
                           levels=32, symmetric=True, normed=True)
        
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        
        return np.array([contrast, homogeneity, energy, correlation])
    except:
        return np.zeros(4)


def extract_shape_features_fast(image_gray):
    """FAST: pixels_per_cell=(25,25) untuk fitur lebih sedikit"""
    try:
        if image_gray.dtype != np.float32:
            image_gray = image_gray.astype('float32') / 255.0
        
        features = hog(image_gray, orientations=8, 
                      pixels_per_cell=(25, 25),
                      cells_per_block=(2, 2),
                      transform_sqrt=True, block_norm="L2-Hys")
        return features
    except:
        return np.zeros(72)  # Approximate size


def extract_features_fast(pil_image):
    """Pipeline lengkap FAST mode"""
    start_time = time.time()
    
    img_rgb, img_gray = preprocess_image_fast(pil_image)
    if img_rgb is None:
        return None, None, 0
    
    color = extract_color_features_fast(img_rgb)
    texture = extract_texture_features_fast(img_gray)
    shape = extract_shape_features_fast(img_gray)
    
    features = np.hstack([color, texture, shape])
    elapsed = time.time() - start_time
    
    return features, img_rgb, elapsed


# ==========================================
# 3. PREPROCESSING - ACCURATE MODE
# ==========================================

def remove_background_accurate(image):
    """
    ACCURATE MODE: Dengan background removal
    """
    try:
        if image is None or image.size == 0:
            return image
        
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        
        _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(s_channel)
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            result = cv2.bitwise_and(image, image, mask=clean_mask)
            return result
        return image
    except:
        return image


def preprocess_image_accurate(pil_image, target_size=(100, 100)):
    """
    ACCURATE MODE: Dengan background removal
    """
    try:
        img_array = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
        
        # Background removal
        img_segmented = remove_background_accurate(img_resized)
        
        img_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
        
        return img_rgb, img_gray
    except Exception as e:
        st.error(f"Error accurate preprocessing: {e}")
        return None, None


def extract_color_features_accurate(image_rgb):
    """ACCURATE: Bins [8,8,8] = 512 fitur dengan masking"""
    try:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        if np.sum(mask) == 0:
            mask = None
        
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8], 
                           [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist
    except:
        return np.zeros(512)


def extract_texture_features_accurate(image_gray):
    """ACCURATE: 3 sudut, 5 fitur"""
    try:
        if image_gray.dtype != np.uint8:
            image_gray = (image_gray * 255).astype(np.uint8)
        
        if np.std(image_gray) < 10:
            image_gray = cv2.equalizeHist(image_gray)
        
        img_quantized = (image_gray // 8).astype(np.uint8)
        glcm = graycomatrix(img_quantized, distances=[1], 
                           angles=[0, np.pi/4, np.pi/2], 
                           levels=32, symmetric=True, normed=True)
        
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        
        return np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    except:
        return np.zeros(5)


def extract_shape_features_accurate(image_gray):
    """ACCURATE: pixels_per_cell=(16,16) untuk fitur lebih detail"""
    try:
        if image_gray.dtype != np.float32:
            image_gray = image_gray.astype('float32') / 255.0
        
        features = hog(image_gray, orientations=9, 
                      pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2),
                      transform_sqrt=True, block_norm="L2-Hys")
        return features
    except:
        return np.zeros(324)


def extract_features_accurate(pil_image):
    """Pipeline lengkap ACCURATE mode"""
    start_time = time.time()
    
    img_rgb, img_gray = preprocess_image_accurate(pil_image)
    if img_rgb is None:
        return None, None, 0
    
    color = extract_color_features_accurate(img_rgb)
    texture = extract_texture_features_accurate(img_gray)
    shape = extract_shape_features_accurate(img_gray)
    
    features = np.hstack([color, texture, shape])
    elapsed = time.time() - start_time
    
    return features, img_rgb, elapsed


# ==========================================
# 4. LOAD MODELS
# ==========================================

@st.cache_resource
def load_model_resources(mode):
    """Load model berdasarkan mode yang dipilih"""
    try:
        config = MODELS[mode]
        
        if not os.path.exists(config['model_path']):
            return None, None, None, f"Model {mode} tidak ditemukan"
        
        model = joblib.load(config['model_path'])
        scaler = joblib.load(config['scaler_path'])
        
        if os.path.exists(config['label_encoder_path']):
            le = joblib.load(config['label_encoder_path'])
            class_names = le.classes_.tolist()
        else:
            return None, None, None, f"Label encoder {mode} tidak ditemukan"
        
        return model, scaler, class_names, None
    
    except Exception as e:
        return None, None, None, str(e)


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_fruit(pil_image, mode, model, scaler, class_names):
    """Prediksi dengan mode yang dipilih"""
    
    # Extract features sesuai mode
    if mode == 'fast':
        features, img_processed, extract_time = extract_features_fast(pil_image)
    else:
        features, img_processed, extract_time = extract_features_accurate(pil_image)
    
    if features is None:
        return None
    
    # Validasi dimensi
    features = features.reshape(1, -1)
    expected = scaler.n_features_in_
    
    if features.shape[1] != expected:
        st.error(f"❌ Dimensi tidak cocok! Expected: {expected}, Got: {features.shape[1]}")
        return None
    
    # Scale & Predict
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    label = class_names[pred_idx]
    
    return {
        'label': label,
        'confidence': confidence,
        'probs': probs,
        'class_names': class_names,
        'img_processed': img_processed,
        'extract_time': extract_time,
        'features_dim': features.shape[1]
    }


# ==========================================
# 6. UI UTAMA
# ==========================================

st.title("🍎 Klasifikasi Buah - Dual Mode System")

# Mode Selection
st.markdown("### 🎛️ Pilih Mode Prediksi")

col_mode1, col_mode2 = st.columns(2)

with col_mode1:
    fast_selected = st.button(
        "⚡ Fast Mode",
        type="primary" if 'selected_mode' not in st.session_state or st.session_state.selected_mode == 'fast' else "secondary",
        use_container_width=True,
        help="Cepat (~0.1s) tanpa background removal"
    )
    if fast_selected:
        st.session_state.selected_mode = 'fast'

with col_mode2:
    accurate_selected = st.button(
        "🎯 Accurate Mode",
        type="primary" if 'selected_mode' in st.session_state and st.session_state.selected_mode == 'accurate' else "secondary",
        use_container_width=True,
        help="Akurat (~2s) dengan background removal"
    )
    if accurate_selected:
        st.session_state.selected_mode = 'accurate'

# Default mode
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = 'fast'

current_mode = st.session_state.selected_mode
mode_config = MODELS[current_mode]

# Info mode
st.info(f"{mode_config['icon']} **{mode_config['name']}**: {mode_config['description']}")

# Load model sesuai mode
model, scaler, class_names, error = load_model_resources(current_mode)

# Sidebar
with st.sidebar:
    st.header("📊 Status Sistem")
    
    # Status kedua model
    for mode_key, mode_info in MODELS.items():
        with st.expander(f"{mode_info['icon']} {mode_info['name']}"):
            if os.path.exists(mode_info['model_path']):
                st.success("✅ Model tersedia")
                if mode_key == current_mode and model:
                    st.write(f"**Kernel:** {model.kernel}")
                    st.write(f"**Kelas:** {len(class_names)}")
            else:
                st.error("❌ Model tidak ditemukan")
                st.caption(f"Path: {mode_info['model_path']}")
    
    st.markdown("---")
    
    # Comparison table
    st.markdown("### ⚖️ Perbandingan")
    st.markdown("""
    | Fitur | Fast | Accurate |
    |-------|------|----------|
    | **Kecepatan** | ~0.1s | ~2s |
    | **Background Removal** | ❌ | ✅ |
    | **Color Bins** | 6×6×6 | 8×8×8 |
    | **GLCM Angles** | 2 | 3 |
    | **HOG Cell Size** | 25×25 | 16×16 |
    | **Total Fitur** | ~292 | ~841 |
    | **Akurasi** | Good | Best |
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Rekomendasi")
    st.markdown("""
    **Fast Mode:**
    - Real-time classification
    - Gambar sudah bersih
    - Background polos
    
    **Accurate Mode:**
    - Maximum accuracy
    - Background kompleks
    - Lighting kurang ideal
    """)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Gambar Buah", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📷 Gambar Asli")
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, use_container_width=True)
        st.caption(f"Ukuran: {image_pil.size[0]}×{image_pil.size[1]} px")
    
    # Predict button
    if st.button("🔍 Analisis Buah", type="primary", use_container_width=True):
        with st.spinner(f"⏳ Processing dengan {mode_config['name']}..."):
            
            start_total = time.time()
            result = predict_fruit(image_pil, current_mode, model, scaler, class_names)
            total_time = time.time() - start_total
            
            if result:
                # Show processed image
                with col2:
                    st.write("### 🎨 Hasil Preprocessing")
                    st.image(result['img_processed'], use_container_width=True)
                    
                    # Performance metrics
                    st.metric("⏱️ Extraction Time", f"{result['extract_time']:.3f}s")
                    st.metric("🔢 Feature Dimension", result['features_dim'])
                
                # Results
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                confidence = result['confidence']
                label = result['label']
                
                # Threshold
                threshold = 0.6 if len(class_names) <= 10 else 0.5
                
                if confidence >= threshold:
                    st.success(f"### 🏷️ Prediksi: **{label}**")
                    
                    if confidence >= 0.8:
                        st.success(f"Confidence: {confidence*100:.1f}% - Sangat Yakin! ✅")
                    elif confidence >= 0.6:
                        st.info(f"Confidence: {confidence*100:.1f}% - Cukup Yakin ✓")
                    else:
                        st.warning(f"Confidence: {confidence*100:.1f}% - Kurang Yakin ⚠️")
                    
                    st.progress(float(confidence))
                else:
                    st.warning("⚠️ **Prediksi Tidak Yakin**")
                    st.info(f"Kemiripan tertinggi: **{label}** ({confidence*100:.1f}%)")
                    
                    if current_mode == 'fast':
                        st.info("💡 Coba gunakan **Accurate Mode** untuk hasil lebih baik!")
                
                # Top 3
                st.write("### 🔝 Top 3 Prediksi")
                indices = np.argsort(result['probs'])[::-1][:3]
                
                cols = st.columns(3)
                for i, idx in enumerate(indices):
                    with cols[i]:
                        st.metric(
                            label=f"#{i+1}",
                            value=result['class_names'][idx],
                            delta=f"{result['probs'][idx]*100:.1f}%"
                        )
                
                # Performance summary
                with st.expander("⚡ Performance Summary"):
                    perf_cols = st.columns(3)
                    with perf_cols[0]:
                        st.metric("Feature Extraction", f"{result['extract_time']:.3f}s")
                    with perf_cols[1]:
                        st.metric("Total Time", f"{total_time:.3f}s")
                    with perf_cols[2]:
                        st.metric("Mode", mode_config['name'])
                
                # All probabilities
                with st.expander("📋 Semua Probabilitas"):
                    import pandas as pd
                    indices_all = np.argsort(result['probs'])[::-1]
                    df = pd.DataFrame({
                        'Rank': range(1, len(indices_all)+1),
                        'Kelas': [result['class_names'][i] for i in indices_all],
                        'Probabilitas (%)': [result['probs'][i]*100 for i in indices_all]
                    })
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ Prediksi gagal!")

elif not model:
    st.warning(f"⚠️ Model {current_mode} belum tersedia!")
    st.error(error if error else "Unknown error")
    
    st.info("""
    ### 📝 Setup Instructions:
    
    **Untuk Fast Mode:**
    1. Train model dengan `extract_features_fast()`
    2. Save sebagai `svm_fast_model.pkl`, `scaler_fast.pkl`, `label_encoder_fast.pkl`
    
    **Untuk Accurate Mode:**
    1. Train model dengan `extract_features_accurate()`
    2. Save sebagai `svm_accurate_model.pkl`, `scaler_accurate.pkl`, `label_encoder_accurate.pkl`
    
    **Struktur folder:**
    ```
    Output_Artifacts/
    ├── svm_fast_model.pkl
    ├── scaler_fast.pkl
    ├── label_encoder_fast.pkl
    ├── svm_accurate_model.pkl
    ├── scaler_accurate.pkl
    └── label_encoder_accurate.pkl
    ```
    """)

# Footer
st.markdown("---")
st.caption(f"🍎 Dual Model Fruit Classification | Current Mode: {mode_config['name']}")
st.caption("Made with ❤️ using Streamlit")