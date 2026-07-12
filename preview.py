import streamlit as st
import os
import math
from config import IMG_DIR, LABEL_DIR
from PIL import Image, ImageDraw



def load_image_with_boxes(img_path, label_path, class_list):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if os.path.exists(label_path):
        w, h = img.size
        with open(label_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(float(parts[0]))
            class_name = (
                class_list[class_id]
                if class_id < len(class_list)
                else f"Class {class_id}"
            )

            coords = list(map(float, parts[1:]))

            # ต้องมีอย่างน้อย 3 จุด (6 ค่า) และจำนวนค่าต้องเป็นคู่
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            points = [
                (coords[i] * w, coords[i + 1] * h)
                for i in range(0, len(coords), 2)
            ]

            draw.polygon(points, outline="red", width=2)
            draw.text(
                (points[0][0], max(0, points[0][1] - 15)),
                class_name,
                fill="red"
            )

    return img


def preview_page():
    st.header("🖼️ Preview Labeled Images")

    if not os.path.exists(IMG_DIR) or not os.path.exists(LABEL_DIR):
        st.warning("ไม่พบโฟลเดอร์รูปหรือ label")
        return

    labeled_imgs = sorted([
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not labeled_imgs:
        st.info("ยังไม่มีรูป")
        return

    if "class_list" not in st.session_state:
        st.warning("ไม่พบ class_list")
        return

    class_list = st.session_state.class_list

    selected_filter = st.selectbox(
        "🎯 เลือก Class",
        ["ทั้งหมด"] + class_list,
        key="class_filter"
    )

    # ======================
    # Filter
    # ======================
    filtered_imgs = []

    if selected_filter == "ทั้งหมด":
        filtered_imgs = labeled_imgs
    else:
        selected_id = class_list.index(selected_filter)

        for img_file in labeled_imgs:
            label_path = os.path.join(
                LABEL_DIR,
                img_file.rsplit(".", 1)[0] + ".txt"
            )

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(float(parts[0]))
                if cid == selected_id:
                    filtered_imgs.append(img_file)
                    break

    if not filtered_imgs:
        st.warning("ไม่พบรูป class นี้")
        return

    # ======================
    # Pagination
    # ======================
    images_per_page = 25
    total_pages = math.ceil(len(filtered_imgs) / images_per_page)

    page = st.number_input(
        "เลือกหน้า",
        min_value=1,
        max_value=total_pages,
        value=1,
        key="page_number"
    )

    start = (page - 1) * images_per_page
    end = start + images_per_page
    page_images = filtered_imgs[start:end]

    # ======================
    # Zoom Section
    # ======================
    if "zoom_image" in st.session_state:
        st.divider()
        st.subheader(f"🔍 {st.session_state.zoom_image}")

        full_img_path = os.path.join(
            IMG_DIR,
            st.session_state.zoom_image
        )

        label_path = os.path.join(
            LABEL_DIR,
            st.session_state.zoom_image.rsplit(".", 1)[0] + ".txt"
        )

        img = load_image_with_boxes(
            full_img_path,
            label_path,
            class_list
        )

        st.image(img, use_column_width=True)

        if st.button("❌ ปิดภาพ"):
            del st.session_state.zoom_image
            st.rerun()

        st.divider()

    # ======================
    # Grid
    # ======================
    cols = st.columns(5)

    for i, img_file in enumerate(page_images):
        with cols[i % 5]:

            img_path = os.path.join(IMG_DIR, img_file)
            label_path = os.path.join(
                LABEL_DIR,
                img_file.rsplit(".", 1)[0] + ".txt"
            )

            img = load_image_with_boxes(
                img_path,
                label_path,
                class_list
            )

            st.image(img, width=250)

            if st.button("🔍", key=f"zoom_{img_file}"):
                st.session_state.zoom_image = img_file
                st.rerun()

            # ======================
            # Delete Section
            # ======================
            delete_key = f"delete_{img_file}"
            confirm_key = f"confirm_{img_file}"

            if st.button("🗑 Delete", key=delete_key):
                st.session_state[confirm_key] = True

            if st.session_state.get(confirm_key, False):
                st.warning(f"ยืนยันการลบ {img_file}?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ ยืนยัน", key=f"yes_{img_file}"):
                        try:
                            if os.path.exists(img_path):
                                os.remove(img_path)

                            if os.path.exists(label_path):
                                os.remove(label_path)

                            st.success(f"ลบ {img_file} เรียบร้อยแล้ว")

                            st.session_state.pop(confirm_key, None)
                            st.rerun()

                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {e}")

                with col2:
                    if st.button("❌ ยกเลิก", key=f"no_{img_file}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
