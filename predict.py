def predict_page():
    import streamlit as st
    import cv2
    import os
    import datetime
    import json
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    st.header("🔮 Predict")

    col_cfg, col_files = st.columns([1, 2])
    
    with col_cfg:
        st.markdown("### ⚙️ Model Config")
        model_path = st.text_input(
            "📂 Path Model",
            "runs/detect/my_custom_model/weights/best.pt"
        )
        conf_threshold = st.slider("🎚️ ความมั่นใจ (Confidence)", 0.0, 1.0, 0.25, 0.05)

    with col_files:
        st.markdown("### 📁 Select Images")
        
        # ---------------------------------------------------------
        # แก้ไขจาก tkinter มาใช้ st.file_uploader สำหรับ Web App 
        # ---------------------------------------------------------
        uploaded_files = st.file_uploader(
            "📂 อัปโหลดรูปภาพเพื่อทำนาย (เลือกได้หลายไฟล์)", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )

        if not uploaded_files:
            st.info("💡 กรุณาอัปโหลดรูปภาพเพื่อเริ่มการทำนาย")
            return

        file_names_only = [f.name for f in uploaded_files]

        selected_idx = st.selectbox(
            f"📄 รายการไฟล์ (พบ {len(file_names_only)} รูป)",
            range(len(file_names_only)),
            format_func=lambda i: file_names_only[i],
        )

        uploaded_file = uploaded_files[selected_idx]
        img_name = uploaded_file.name

        if st.button("🔍 เริ่มทำนาย (Predict)", type="primary", use_container_width=True):
            try:
                model = YOLO(model_path)
            except Exception:
                st.error(f"❌ ไม่พบไฟล์โมเดลที่: {model_path}")
                st.stop()

            # อ่านรูปภาพจาก Memory (Bytes) โดยตรง
            pil_img = Image.open(uploaded_file).convert("RGB")
            results = model(pil_img, conf=conf_threshold)

            res_plotted_bgr = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)

            st.divider()
            st.image(res_plotted_rgb, caption=f"ผลลัพธ์: {img_name}", width=700)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"✅ เจอวัตถุทั้งหมด {len(boxes)} ชิ้น")
            else:
                st.warning("⚠️ ไม่เจอวัตถุ (ลองลดค่า Confidence ลงดูนะครับ)")

            # ==========================================
            # บันทึก Log ไฟล์ .json
            # ==========================================
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "predict_log.json")

            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            else:
                log_data = []

            names = model.names
            detected_objects = []
            for box in boxes:
                detected_objects.append({
                    "class_id"  : int(box.cls[0]),
                    "class_name": names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "box_xyxy"  : [round(c, 1) for c in box.xyxy[0].tolist()]
                })

            record = {
                "timestamp"           : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path"          : img_name, # เปลี่ยนมาเก็บชื่อไฟล์แทน Path บนเครื่อง
                "model_path"          : model_path,
                "confidence_threshold": conf_threshold,
                "total_detected"      : len(boxes),
                "detections"          : detected_objects
            }

            log_data.append(record)

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            st.info(f"📝 บันทึก Log แล้วที่: {log_path}")