import streamlit as st
import tensorflow as tf
import numpy as np
import string

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide"
)


# ---------------- LOAD TRAINED MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/character_recognition_v2.keras"
    )


model = load_model()


# ---------------- CHARACTER MAPPING ----------------

class_names = (
    list(string.digits)
    + list(string.ascii_uppercase)
    + list(string.ascii_lowercase)
)


# ---------------- IMAGE PREPROCESSING ----------------

def preprocess_canvas(image_data):

    image = Image.fromarray(
        image_data.astype("uint8")
    ).convert("RGB")

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)

    image = image.point(
        lambda p: 255 if p > 50 else 0
    )

    bbox = image.getbbox()

    if bbox is None:
        return None

    image = image.crop(bbox)
    image.thumbnail((20, 20))

    final_image = Image.new("L", (28, 28), 0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    final_image.paste(image, (x, y))

    image_array = np.array(
        final_image,
        dtype=np.float32
    ) / 255.0

    return image_array.reshape(1, 28, 28, 1)


# ---------------- APPLICATION INTERFACE ----------------

st.title("✍️ Handwritten Character Recognition")

st.success("AI model loaded successfully!")

st.write(
    "Draw a digit or alphabet and let our AI recognize it."
)

st.divider()

st.subheader("Draw your character")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="character_canvas",
    return_image_data=True
)


# ---------------- RECOGNITION ----------------

if st.button("🔍 Recognize Character"):

    if canvas_result.image_data is None:

        st.warning("Please draw a character first.")

    else:

        processed_image = preprocess_canvas(
            canvas_result.image_data
        )

        if processed_image is None:

            st.warning("Please draw a character first.")

        else:

            st.subheader("Image received by AI")

            st.image(
                processed_image[0, :, :, 0],
                width=200,
                clamp=True
            )

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_character = class_names[
                predicted_index
            ]

            confidence = (
                predictions[0][predicted_index] * 100
            )

            st.divider()

            st.subheader("AI Prediction")

            st.success(
                f"Recognized Character: {predicted_character}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Top 3 Predictions")

            top_3 = np.argsort(
                predictions[0]
            )[-3:][::-1]

            for index in top_3:

                character = class_names[index]

                score = predictions[0][index] * 100

                st.write(
                    f"**{character}** — {score:.2f}%"
                )