def labeling_page():
    import streamlit as st
    import pandas as pd
    import os
    import hashlib
    import math
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from config import IMG_DIR, LABEL_DIR
    from helpers.image_helper import transform_image

    CLASSES_FILE = "classes.txt"
    MAX_WIDTH = 1000

    def load_classes():
        if not os.path.exists(CLASSES_FILE):
            return []
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            return [c.strip() for c in f.readlines() if c.strip()]

    def save_classes(class_list):
        with open(CLASSES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(class_list))

    st.header("🖌️ Labeling: Multi-Image Annotation")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### ⚙️ ตั้งค่า Class")

        if "class_list" not in st.session_state:
            st.session_state.class_list = load_classes()

        class_text = st.text_area(
            "รายชื่อวัตถุ (บรรทัดละชื่อ)",
            "\n".join(st.session_state.class_list),
            height=120,
        )

        new_class_list = [c.strip() for c in class_text.split("\n") if c.strip()]

        if st.button("💾 Save Classes"):
            if not new_class_list:
                st.warning("ต้องมีอย่างน้อย 1 class")
            else:
                save_classes(new_class_list)
                st.session_state.class_list = new_class_list
                st.success("บันทึก classes.txt แล้ว")
                st.rerun()

        if not new_class_list:
            st.warning("⚠ กรุณาเพิ่มอย่างน้อย 1 class")
            return

        stroke_width = st.slider("ความหนาเส้น", 1, 5, 2)

        st.divider()
        st.markdown("### 🛠️ Drawing Mode")

        drawing_mode = st.radio(
            "เครื่องมือวาด",
            ["rect", "polygon", "transform"],
            horizontal=True,
            help="rect=วาดสี่เหลี่ยม | polygon=วาดหลายเหลี่ยม (ดับเบิลคลิกเพื่อปิด) | transform=เลือก/หมุน/ย้ายกรอบที่วาดแล้ว"
        )

        if drawing_mode == "polygon":
            st.info("💡 คลิกเพื่อเพิ่มจุด **ดับเบิลคลิก** เพื่อปิดรูปทรง")

        st.divider()
        st.markdown("### 🖼️ Image Config")

        defaults = {
            "rotate_angle": 0,
            "flip_h": False,
            "flip_v": False,
            "brightness": 1.0,
            "contrast": 1.0,
            "color": 1.0,
        }

        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        rotate_angle = st.slider("🔄 Rotate", -180, 180, step=5, key="rotate_angle")
        flip_h = st.checkbox("↔️ Flip Horizontal", key="flip_h")
        flip_v = st.checkbox("↕️ Flip Vertical", key="flip_v")
        brightness = st.slider("☀️ Brightness", 0.2, 2.0, key="brightness")
        contrast = st.slider("🎚️ Contrast", 0.2, 2.0, key="contrast")
        color = st.slider("🎨 Color", 0.2, 2.0, key="color")

        def reset_image_config():
            for k, v in defaults.items():
                st.session_state[k] = v

        st.button("♻️ Reset Image Config", on_click=reset_image_config)

    with col2:
        

        # --- ส่วนที่ 1: เลือกรูปภาพ (Mac / Windows / Linux Compatible) ---

        uploaded_files = st.file_uploader(
            "📂 เลือกรูปภาพ",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("💡 เลือกรูปภาพเพื่อเริ่มต้นทำ Labeling")
            return

        file_names_only = [f.name for f in uploaded_files]

        selected_idx = st.selectbox(
            f"📄 รายการไฟล์ (พบ {len(file_names_only)} รูป)",
            range(len(file_names_only)),
            format_func=lambda i: file_names_only[i],
        )

        uploaded_file = uploaded_files[selected_idx]
        uploaded_file.seek(0)

        file_bytes = uploaded_file.getvalue()

        img_name = uploaded_file.name

        orig_image = Image.open(uploaded_file).convert("RGB")

        image = transform_image(
            orig_image,
            rotate_angle,
            flip_h,
            flip_v,
            brightness,
            contrast,
            color,
        )

        orig_w, orig_h = image.size

        # การคำนวณสเกลเพื่อแสดงผลบน Canvas
        if orig_w > MAX_WIDTH:
            scale = MAX_WIDTH / orig_w
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            display_image = image.resize((new_w, new_h))
        else:
            scale = 1.0
            new_w, new_h = orig_w, orig_h
            display_image = image

        # สร้าง Key สำหรับ Canvas เพื่อให้วาดใหม่เมื่อเปลี่ยนรูป
        hash_input = file_bytes + str(selected_idx).encode()
        canvas_hash = hashlib.md5(hash_input).hexdigest()
        canvas_key = f"canvas_{canvas_hash}"

        canvas = st_canvas(
            fill_color="rgba(0,150,255,0.2)",
            stroke_width=stroke_width,
            stroke_color="#0066CC",
            background_image=display_image,
            update_streamlit=True,
            height=new_h,
            width=new_w,
            drawing_mode=drawing_mode,
            key=canvas_key,
        )

        if canvas.json_data is not None:
            objects = pd.json_normalize(canvas.json_data["objects"])

            if not objects.empty:
                NOISE_TYPES = ("circle",)
                valid_objects = objects[~objects["type"].isin(NOISE_TYPES)].copy()

                def is_valid_rect(row):
                    if row.get("type") not in ("rect", None):
                        return True
                    w_box = row.get("width", 0) * row.get("scaleX", 1.0) / scale
                    h_box = row.get("height", 0) * row.get("scaleY", 1.0) / scale
                    return w_box >= 5 and h_box >= 5

                valid_objects = valid_objects[valid_objects.apply(is_valid_rect, axis=1)].reset_index(drop=True)

                if valid_objects.empty:
                    st.info("ยังไม่มีกรอบที่ใช้งานได้")
                else:
                    st.success(f"✅ พบ {len(valid_objects)} กรอบ")
                    st.markdown("### 🏷️ กำหนด Class ให้แต่ละกรอบ")

                    class_per_box = []
                    for i in range(len(valid_objects)):
                        selected = st.selectbox(
                            f"กรอบที่ {i+1}",
                            new_class_list,
                            key=f"class_box_{i}"
                        )
                        class_per_box.append(selected)

                    if st.button("💾 บันทึกข้อมูล (Save)", type="primary"):
                        label_name = os.path.splitext(img_name)[0] + ".txt"

                        os.makedirs(IMG_DIR, exist_ok=True)
                        os.makedirs(LABEL_DIR, exist_ok=True)

                        # บันทึกรูปภาพลงโฟลเดอร์ datasets
                        image.save(os.path.join(IMG_DIR, img_name))

                        lines = []
                        for idx, row in valid_objects.iterrows():
                            cid = new_class_list.index(class_per_box[idx])

                            # --- Rectangle → Seg format ---
                            if row.get("type") in ("rect", None) and pd.notna(row.get("left", None)):
                                w_box = row["width"] * row.get("scaleX", 1.0) / scale
                                h_box = row["height"] * row.get("scaleY", 1.0) / scale
                                angle = row.get("angle", 0)
                                rad   = math.radians(angle)
                                tl_x = row["left"] / scale
                                tl_y = row["top"] / scale
                                cx = tl_x + (w_box / 2) * math.cos(rad) - (h_box / 2) * math.sin(rad)
                                cy = tl_y + (w_box / 2) * math.sin(rad) + (h_box / 2) * math.cos(rad)
                                w2, h2 = w_box / 2, h_box / 2
                                corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
                                rotated = [(cx + dx * math.cos(rad) - dy * math.sin(rad),
                                            cy + dx * math.sin(rad) + dy * math.cos(rad)) for dx, dy in corners]
                                coords = " ".join(f"{x/orig_w:.6f} {y/orig_h:.6f}" for x, y in rotated)
                                lines.append(f"{cid} {coords}")

                            # --- Polygon → Seg format ---
                            elif "path" in row and row.get("path") is not None:
                                try:
                                    path_data = row["path"]
                                    points = [(p[1] / scale, p[2] / scale) for p in path_data if len(p) >= 3 and p[0] in ("M", "L")]
                                    if len(points) < 3: continue
                                    coords = " ".join(f"{x/orig_w:.6f} {y/orig_h:.6f}" for x, y in points)
                                    lines.append(f"{cid} {coords}")
                                except Exception: continue

                        with open(os.path.join(LABEL_DIR, label_name), "w") as f:
                            f.write("\n".join(lines))

                        st.toast("บันทึกเรียบร้อย 🎉")
