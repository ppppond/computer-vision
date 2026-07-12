def webcam_page():
    import streamlit as st
    import cv2
    from ultralytics import YOLO
    import time
    import json
    import os
    import datetime

    st.header("🎥 Webcam (ไม่ใช้ Thread)")

    LOG_INTERVAL_SEC = 1.0  # บันทึกทุก 1 วินาที

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    # ==========================================
    # Helper: บันทึก log ทันที (append-safe)
    # ==========================================
    def append_log(records: list):
        """รับ list of dict แล้ว append เข้า predict_log.json ทันที"""
        if not records:
            return
        log_dir  = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "predict_log.json")

        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = []

        log_data.extend(records)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    # ==========================================
    # UI
    # ==========================================
    col1, col2 = st.columns(2)
    with col1:
        model_path = st.text_input("📂 Path Model", "runs/detect/my_custom_model/weights/best.pt")
    with col2:
        conf_threshold = st.slider("🎚️ Confidence", 0.0, 1.0, 0.25, 0.05)

    mirror_view  = st.checkbox("🪞 Mirror View", value=True)
    camera_index = st.number_input("📷 Camera Index", min_value=0, max_value=5, value=0)

    placeholder = st.empty()

    if st.button("▶️ เปิด / ปิด กล้อง"):
        st.session_state.cam_running = not st.session_state.cam_running

    if st.session_state.cam_running:
        try:
            model = YOLO(model_path)
        except:
            st.error("❌ ไม่พบไฟล์โมเดล")
            st.session_state.cam_running = False
            st.stop()

        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("❌ เปิดกล้องไม่สำเร็จ")
            st.session_state.cam_running = False
            st.stop()

        names           = model.names
        frame_count     = 0
        class_summary   = {}
        snapshot_buffer = []      # buffer snapshot ก่อน flush
        last_log_time   = time.time()
        FLUSH_EVERY     = 5       # flush ทุก 5 snapshots เพื่อลด I/O

        while st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame   = cv2.resize(frame, (640, 480))
            results = model(frame, conf=conf_threshold)
            plotted = results[0].plot()
            boxes   = results[0].boxes

            # นับ class_summary สะสม
            for box in boxes:
                cls_name = names[int(box.cls[0])]
                class_summary[cls_name] = class_summary.get(cls_name, 0) + 1

            # สร้าง snapshot ทุก N วินาที
            now = time.time()
            if now - last_log_time >= LOG_INTERVAL_SEC:
                timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detections = []
                for box in boxes:
                    detections.append({
                        "class_name": names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "box_xyxy"  : [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    })

                snapshot_buffer.append({
                    "type"                 : "webcam",
                    "timestamp"            : timestamp_now,
                    "source"               : f"camera_{int(camera_index)}",
                    "model_path"           : model_path,
                    "confidence_threshold" : conf_threshold,
                    "frame"                : frame_count,
                    "total_detected"       : len(detections),
                    "detections"           : detections,
                })
                last_log_time = now

                # ✅ flush buffer → บันทึกทันทีทุก FLUSH_EVERY snapshots
                if len(snapshot_buffer) >= FLUSH_EVERY:
                    append_log(snapshot_buffer)
                    snapshot_buffer = []

            if mirror_view:
                plotted = cv2.flip(plotted, 1)

            frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, use_column_width=True)

            time.sleep(0.03)

        cap.release()
        placeholder.empty()

        # ✅ flush snapshot ที่เหลือค้างใน buffer (กรณีกดปิดกลางคัน)
        if snapshot_buffer:
            append_log(snapshot_buffer)
            snapshot_buffer = []

        # ✅ บันทึก summary session (ถ้ามีการ scan อย่างน้อย 1 frame)
        if frame_count > 0:
            final_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            append_log([{
                "type"                 : "webcam",
                "timestamp"            : final_timestamp,
                "source"               : f"camera_{int(camera_index)}",
                "model_path"           : model_path,
                "confidence_threshold" : conf_threshold,
                "frame"                : frame_count,
                "total_detected"       : len(detections),
                "total_boxes"          : len(detections),  # ✅ จำนวนกรอบที่เจอ
                "detections"           : detections,
            }])

            log_path = os.path.join("logs", "predict_log.json")

            st.divider()
            st.success(
                f"✅ Session เสร็จสิ้น | {frame_count} frames "
                f"| เจอวัตถุทั้งหมด {sum(class_summary.values())} ครั้ง"
            )
            if class_summary:
                st.markdown("### 📊 สรุปวัตถุที่เจอ")
                for cls, count in sorted(class_summary.items(), key=lambda x: -x[1]):
                    st.write(f"- **{cls}**: {count} ครั้ง")

            st.info(f"📝 บันทึก Log แล้วที่: {log_path}")