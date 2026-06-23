import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="PlantGuard — Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d2010 !important;
    }
    [data-testid="stSidebar"] section {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] * {
        color: #a0c8a0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #5a9a5a !important;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #1a3a1a !important;
        border-color: #2a5a2a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #ffffff !important;
    }
    /* ── Main background ── */
    .stApp {
        background-color: #f2f6f2;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1100px;
    }
    /* ── Primary button ── */
    .stButton > button {
        background-color: #1a4a1a;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        font-weight: 500;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #0d3010;
        border: none;
        color: #ffffff;
    }
    .stButton > button:focus {
        outline: 2px solid #4a9a4a;
        border: none;
    }
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #90c890;
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] label {
        color: #0d2010 !important;
        font-weight: 500;
    }
    /* ── Divider ── */
    hr {
        border-color: #d0e8d0;
    }
    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #2d7a2d !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Unchanged prediction logic ──────────────────────────────────────────────
def model_predict(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


CLASS_NAMES = [
    'Apple — Apple Scab', 'Apple — Black Rot', 'Apple — Cedar Apple Rust', 'Apple — Healthy',
    'Blueberry — Healthy', 'Cherry — Powdery Mildew', 'Cherry — Healthy',
    'Corn — Cercospora / Gray Leaf Spot', 'Corn — Common Rust', 'Corn — Northern Leaf Blight',
    'Corn — Healthy', 'Grape — Black Rot', 'Grape — Esca (Black Measles)',
    'Grape — Leaf Blight (Isariopsis)', 'Grape — Healthy',
    'Orange — Huanglongbing (Citrus Greening)', 'Peach — Bacterial Spot', 'Peach — Healthy',
    'Pepper — Bacterial Spot', 'Pepper — Healthy',
    'Potato — Early Blight', 'Potato — Late Blight', 'Potato — Healthy',
    'Raspberry — Healthy', 'Soybean — Healthy', 'Squash — Powdery Mildew',
    'Strawberry — Leaf Scorch', 'Strawberry — Healthy',
    'Tomato — Bacterial Spot', 'Tomato — Early Blight', 'Tomato — Late Blight',
    'Tomato — Leaf Mold', 'Tomato — Septoria Leaf Spot',
    'Tomato — Spider Mites (Two-spotted)', 'Tomato — Target Spot',
    'Tomato — Yellow Leaf Curl Virus', 'Tomato — Mosaic Virus', 'Tomato — Healthy',
]

HEALTHY_TIPS = {
    'Apple': 'Ensure good air circulation and avoid overhead watering.',
    'Blueberry': 'Maintain acidic soil pH (4.5–5.5) and prune annually.',
    'Cherry': 'Prune after harvest and watch for signs of fungal issues in wet weather.',
    'Corn': 'Rotate crops yearly and remove crop debris after harvest.',
    'Grape': 'Prune vines in late winter and ensure good canopy airflow.',
    'Orange': 'Monitor for psyllid insects which spread citrus greening.',
    'Peach': 'Apply dormant oil spray in late winter to prevent pests.',
    'Pepper': 'Avoid wetting foliage and use copper-based sprays preventively.',
    'Potato': 'Use certified seed potatoes and avoid planting in waterlogged soil.',
    'Raspberry': 'Remove old canes after fruiting to prevent disease buildup.',
    'Soybean': 'Rotate with non-legume crops to reduce soil-borne pathogens.',
    'Squash': 'Use row covers early in the season to prevent powdery mildew spores.',
    'Strawberry': 'Renovate beds yearly and ensure good drainage.',
    'Tomato': 'Stake plants, mulch the base, and water at the soil level.',
}

DISEASE_TIPS = {
    'Scab': 'Apply fungicide at bud break. Remove and destroy fallen leaves.',
    'Black Rot': 'Prune infected parts, apply copper-based fungicide.',
    'Cedar Apple Rust': 'Remove nearby cedar/juniper hosts if possible. Apply myclobutanil.',
    'Powdery Mildew': 'Improve air circulation. Apply sulfur or neem oil spray.',
    'Cercospora': 'Apply foliar fungicide at first sign. Rotate crops.',
    'Common Rust': 'Use resistant varieties. Apply fungicide if severe.',
    'Northern Leaf Blight': 'Apply fungicide. Remove infected plant debris.',
    'Esca': 'No cure; prune infected wood well below visible symptoms.',
    'Leaf Blight': 'Apply copper fungicide. Avoid wetting foliage.',
    'Huanglongbing': 'No cure. Remove infected trees. Control psyllid populations.',
    'Bacterial Spot': 'Apply copper-based bactericide. Avoid overhead irrigation.',
    'Early Blight': 'Apply chlorothalonil or copper fungicide. Remove lower infected leaves.',
    'Late Blight': 'Apply fungicide immediately. Destroy severely infected plants.',
    'Leaf Mold': 'Reduce humidity in greenhouses. Apply fungicide.',
    'Septoria': 'Apply fungicide at first sign. Remove infected leaves.',
    'Spider Mites': 'Apply miticide or insecticidal soap. Increase humidity.',
    'Target Spot': 'Apply chlorothalonil fungicide. Ensure good plant spacing.',
    'Yellow Leaf Curl Virus': 'Control whitefly vectors. Remove infected plants.',
    'Mosaic Virus': 'No cure. Remove infected plants. Control aphid vectors.',
    'Leaf Scorch': 'Apply fungicide. Remove infected leaves and plant debris.',
}


def get_tip(class_name):
    crop = class_name.split(' — ')[0]
    disease = class_name.split(' — ')[1] if ' — ' in class_name else ''
    if 'Healthy' in disease:
        return HEALTHY_TIPS.get(crop, 'Keep up current care practices.')
    for key, tip in DISEASE_TIPS.items():
        if key.lower() in disease.lower():
            return tip
    return 'Consult an agronomist for targeted treatment advice.'


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:1.5rem;">
    <div style="background:#2d7a2d; border-radius:8px; width:32px; height:32px;
                display:flex; align-items:center; justify-content:center;
                font-size:13px; color:#fff; font-weight:700;">PG</div>
    <span style="color:#ffffff; font-size:1rem; font-weight:600;">PlantGuard</span>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Disease Recognition", "About"],
    label_visibility="visible"
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background:#1a3a1a; border-radius:8px; padding:0.75rem 1rem; margin-top:0.5rem;">
    <p style="color:#7abf7a; font-size:0.7rem; text-transform:uppercase;
              letter-spacing:1px; margin:0 0 6px;">Model stats</p>
    <p style="color:#fff; font-size:0.85rem; margin:2px 0;">38 disease classes</p>
    <p style="color:#fff; font-size:0.85rem; margin:2px 0;">87K training images</p>
    <p style="color:#fff; font-size:0.85rem; margin:2px 0;">128 x 128 input size</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="margin-top:2rem;">
    <p style="color:#3a6a3a; font-size:0.7rem; line-height:1.6;">
        Supports: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach,
        Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
    </p>
</div>
""", unsafe_allow_html=True)


# ── Home page ─────────────────────────────────────────────────────────────────
if app_mode == "Home":

    st.markdown("""
    <div style="background:#0d2010; border-radius:14px; padding:2.5rem 2rem;
                color:#fff; margin-bottom:1.5rem;">
        <p style="color:#5abf5a; font-size:0.75rem; text-transform:uppercase;
                  letter-spacing:2px; margin:0 0 0.5rem;">AI-powered diagnosis</p>
        <h1 style="font-size:2rem; font-weight:700; margin:0 0 0.75rem; color:#fff;">
            Plant Disease Recognition System
        </h1>
        <p style="color:#a0c8a0; font-size:1rem; max-width:580px; margin:0; line-height:1.7;">
            Upload a photo of any plant leaf and our deep learning model instantly
            identifies diseases across 38 classes — helping you protect crops and
            ensure a healthier harvest.
        </p>
    </div>
    """, unsafe_allow_html=True)

    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, title, desc in [
        (c1, "Upload your image",
         "Go to Disease Recognition and upload a clear photo of the affected plant leaf."),
        (c2, "AI analysis",
         "Our model processes the image through a trained convolutional neural network."),
        (c3, "Get your result",
         "Receive an instant diagnosis with care tips to act quickly.")
    ]:
        col.markdown(f"""
        <div style="background:#fff; border-radius:12px; padding:1.25rem;
                    border:1px solid #d0e8d0; height:100%;">
            <p style="font-weight:600; color:#0d2010; margin:0 0 6px;">{title}</p>
            <p style="font-size:0.85rem; color:#555; margin:0; line-height:1.6;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fff; border-radius:12px; padding:1.5rem;
                border:1px solid #d0e8d0; display:flex; gap:2rem; flex-wrap:wrap;">
        <div style="text-align:center; min-width:120px;">
            <p style="font-size:2rem; font-weight:700; color:#0d2010; margin:0;">87K+</p>
            <p style="font-size:0.8rem; color:#777; margin:0;">Training images</p>
        </div>
        <div style="text-align:center; min-width:120px;">
            <p style="font-size:2rem; font-weight:700; color:#0d2010; margin:0;">38</p>
            <p style="font-size:0.8rem; color:#777; margin:0;">Disease classes</p>
        </div>
        <div style="text-align:center; min-width:120px;">
            <p style="font-size:2rem; font-weight:700; color:#0d2010; margin:0;">14</p>
            <p style="font-size:0.8rem; color:#777; margin:0;">Crop types</p>
        </div>
        <div style="text-align:center; min-width:120px;">
            <p style="font-size:2rem; font-weight:700; color:#0d2010; margin:0;">&lt; 2s</p>
            <p style="font-size:0.8rem; color:#777; margin:0;">Result time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── About page ────────────────────────────────────────────────────────────────
elif app_mode == "About":

    st.markdown("""
    <div style="background:#0d2010; border-radius:14px; padding:2rem;
                color:#fff; margin-bottom:1.5rem;">
        <h1 style="font-size:1.75rem; font-weight:700; margin:0 0 0.5rem; color:#fff;">
            About this project
        </h1>
        <p style="color:#a0c8a0; margin:0;">Dataset details and model information</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Dataset overview")
    st.markdown("""
    This model was trained on the
    [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
    by Vipoooool on Kaggle. The dataset was recreated using offline augmentation from the original
    PlantVillage dataset and contains **~87,000 RGB images** of healthy and diseased crop leaves
    across **38 classes**, split 80/20 for training and validation.
    A separate test set of 33 images is provided for prediction.
    """)

    col1, col2, col3 = st.columns(3)
    for col, number, label in [
        (col1, "70,295", "Training images"),
        (col2, "17,572", "Validation images"),
        (col3, "33", "Test images"),
    ]:
        col.markdown(f"""
        <div style="background:#e8f5e8; border-radius:10px; padding:1.25rem;
                    border:1px solid #b0d8b0; text-align:center;">
            <p style="font-size:1.75rem; font-weight:700; color:#0d2010; margin:0;">{number}</p>
            <p style="font-size:0.85rem; color:#3a7a3a; margin:0;">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Supported crops")

    crops = [
        ("Apple", "4 classes"),
        ("Blueberry", "1 class"),
        ("Cherry", "2 classes"),
        ("Corn (Maize)", "4 classes"),
        ("Grape", "4 classes"),
        ("Orange", "1 class"),
        ("Peach", "2 classes"),
        ("Pepper", "2 classes"),
        ("Potato", "3 classes"),
        ("Raspberry", "1 class"),
        ("Soybean", "1 class"),
        ("Squash", "1 class"),
        ("Strawberry", "2 classes"),
        ("Tomato", "10 classes"),
    ]

    cols = st.columns(4)
    for i, (name, classes) in enumerate(crops):
        cols[i % 4].markdown(f"""
        <div style="background:#fff; border-radius:10px; padding:0.75rem 1rem;
                    border:1px solid #d0e8d0; margin-bottom:8px;">
            <span style="font-weight:500; color:#0d2010; font-size:0.9rem;">{name}</span>
            <span style="font-size:0.75rem; color:#888; display:block; margin-top:2px;">
                {classes}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ── Disease Recognition page ─────────────────────────────────────────────────
elif app_mode == "Disease Recognition":

    st.markdown("""
    <div style="background:#0d2010; border-radius:14px; padding:1.75rem 2rem;
                color:#fff; margin-bottom:1.5rem; display:flex;
                align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
        <div>
            <p style="color:#5abf5a; font-size:0.7rem; text-transform:uppercase;
                      letter-spacing:2px; margin:0 0 4px;">AI-powered</p>
            <h1 style="font-size:1.5rem; font-weight:700; margin:0; color:#fff;">
                Disease Recognition
            </h1>
            <p style="color:#a0c8a0; font-size:0.85rem; margin:6px 0 0;">
                Upload a clear photo of a plant leaf to get an instant diagnosis
            </p>
        </div>
        <div style="background:#1a3a1a; border:1px solid #2d6a2d; border-radius:10px;
                    padding:0.75rem 1.25rem; text-align:center;">
            <p style="color:#7abf7a; font-size:0.7rem; text-transform:uppercase;
                      letter-spacing:1px; margin:0;">Classes supported</p>
            <p style="color:#fff; font-size:1.5rem; font-weight:700; margin:0;">38</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        st.markdown("""
        <p style="font-weight:600; color:#0d2010; margin-bottom:0.5rem; font-size:0.95rem;">
            Upload plant image
        </p>
        """, unsafe_allow_html=True)

        test_image = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if test_image:
            st.image(test_image, caption="Uploaded image", use_column_width=True)
        else:
            st.markdown("""
            <div style="background:#fff; border:2px dashed #90c890; border-radius:12px;
                        padding:3rem 2rem; text-align:center; color:#aaa;">
                <p style="margin:0; font-size:0.9rem;">Your uploaded image will appear here</p>
                <p style="margin:4px 0 0; font-size:0.8rem; color:#bbb;">
                    JPG or PNG, max 10MB
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <p style="font-weight:600; color:#0d2010; margin-bottom:0.5rem; font-size:0.95rem;">
            Diagnosis result
        </p>
        """, unsafe_allow_html=True)

        if test_image is None:
            st.markdown("""
            <div style="background:#fff; border:1px solid #d0e8d0; border-radius:12px;
                        padding:2.5rem 1.5rem; text-align:center; color:#aaa; height:100%;">
                <p style="margin:0; font-size:0.9rem; color:#bbb;">
                    Upload an image and click Analyze to see results here
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("Analyze Disease", use_container_width=True):
                with st.spinner("Analyzing your plant image..."):
                    result_index = model_predict(test_image)
                    detected = CLASS_NAMES[result_index]
                    is_healthy = "Healthy" in detected
                    crop = detected.split(" — ")[0]
                    disease = detected.split(" — ")[1] if " — " in detected else ""
                    tip = get_tip(detected)

                if is_healthy:
                    st.markdown(f"""
                    <div style="background:#f0faf0; border:1px solid #6abf6a;
                                border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1rem;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
                            <div style="width:10px; height:10px; border-radius:50%;
                                        background:#2d7a2d;"></div>
                            <span style="font-size:0.7rem; text-transform:uppercase;
                                         letter-spacing:1px; color:#3a7a3a;">No disease detected</span>
                        </div>
                        <p style="font-size:1.1rem; font-weight:700; color:#0d2010;
                                  margin:0 0 4px;">{crop}</p>
                        <p style="font-size:0.85rem; color:#3a7a3a; margin:0;">{disease}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#fff8f0; border:1px solid #e8904a;
                                border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1rem;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;">
                            <div style="width:10px; height:10px; border-radius:50%;
                                        background:#d85a30;"></div>
                            <span style="font-size:0.7rem; text-transform:uppercase;
                                         letter-spacing:1px; color:#993c1d;">Disease detected</span>
                        </div>
                        <p style="font-size:1.1rem; font-weight:700; color:#0d2010;
                                  margin:0 0 4px;">{crop}</p>
                        <p style="font-size:0.9rem; color:#993c1d; margin:0;">{disease}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:#fff; border:1px solid #d0e8d0; border-radius:12px;
                            padding:1.25rem 1.5rem;">
                    <p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;
                              color:#5a9a5a; margin:0 0 6px;">Care recommendation</p>
                    <p style="font-size:0.9rem; color:#1a3a1a; margin:0; line-height:1.6;">
                        {tip}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="background:#fff; border:1px dashed #b0d8b0; border-radius:12px;
                            padding:2rem 1.5rem; text-align:center;">
                    <p style="font-size:0.9rem; color:#777; margin:0;">
                        Click <strong>Analyze Disease</strong> to run the model
                    </p>
                </div>
                """, unsafe_allow_html=True)
