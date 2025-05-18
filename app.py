import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import random
import zipfile

def perspective_rotate(img, angle, direction='horizontal'):
    h, w = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    d = w if direction == 'horizontal' else h
    offset = d * np.tan(angle_rad)

    if direction == 'horizontal':
        src_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
        dst_pts = np.float32([
            [offset if angle > 0 else 0,0],
            [w - (0 if angle > 0 else offset),0],
            [w - (0 if angle > 0 else offset),h],
            [offset if angle > 0 else 0,h]
        ])
    else:
        src_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
        dst_pts = np.float32([
            [0, offset if angle > 0 else 0],
            [w, offset if angle > 0 else 0],
            [w, h - (0 if angle > 0 else offset)],
            [0, h - (0 if angle > 0 else offset)]
        ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rotated = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated

def random_crop(img, crop_scale=0.9):
    h, w = img.shape[:2]
    new_h, new_w = int(h * crop_scale), int(w * crop_scale)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = img[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))

def apply_color_filters(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    filters = [
        lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(1.1, 1.3)),
        lambda i: ImageEnhance.Brightness(i).enhance(random.uniform(0.7, 0.9)),
        lambda i: ImageEnhance.Contrast(i).enhance(random.uniform(1.1, 1.3)),
        lambda i: ImageEnhance.Contrast(i).enhance(random.uniform(0.7, 0.9)),
        lambda i: i.convert("RGB").point(lambda p: min(255, int(p * 1.05))),
        lambda i: i.convert("RGB").point(lambda p: max(0, int(p * 0.95))),
    ]

    random.shuffle(filters)
    for f in filters[:random.choice([1, 2])]:
        img_pil = f(img_pil)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def add_noise(img):
    row, col, ch = img.shape
    mean = 0
    var = random.uniform(0.001, 0.01)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img / 255 + gauss
    noisy = np.clip(noisy, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def augment_image(img):
    angles = list(range(-10, 11, 2))  # from -10 to +10 degrees in steps of 2
    augmented_images = []

    # 1. rotation + crop
    for angle in angles:
        rotated = perspective_rotate(img, angle, direction=random.choice(['horizontal', 'vertical']))
        cropped = random_crop(rotated)
        augmented_images.append(cropped)

    # 2. crop + filter
    for _ in angles:
        cropped = random_crop(img)
        filtered = apply_color_filters(cropped)
        augmented_images.append(filtered)

    # 3. filter + rotation
    for angle in angles:
        filtered = apply_color_filters(img)
        rotated = perspective_rotate(filtered, angle, direction=random.choice(['horizontal', 'vertical']))
        augmented_images.append(rotated)

    # 4. rotation + crop + filter
    for angle in angles:
        rotated = perspective_rotate(img, angle, direction=random.choice(['horizontal', 'vertical']))
        cropped = random_crop(rotated)
        filtered = apply_color_filters(cropped)
        augmented_images.append(filtered)

    # 5. Add noise to all above images
    noisy_images = [add_noise(im) for im in augmented_images]
    augmented_images.extend(noisy_images)

    return augmented_images

st.title("Image Augmentation Tool")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)

    if st.button("Generate Augmented Images"):
        with st.spinner("Generating images..."):
            augmented_imgs = augment_image(img)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                for idx, im in enumerate(augmented_imgs):
                    _, im_buf = cv2.imencode(".jpg", im)
                    zip_file.writestr(f"augmented_{idx+1}.jpg", im_buf.tobytes())

            st.success(f"Generated {len(augmented_imgs)} images!")
            zip_buffer.seek(0)
            st.download_button(
                label="Download augmented images as ZIP",
                data=zip_buffer,
                file_name="augmented_images.zip",
                mime="application/zip"
            )
