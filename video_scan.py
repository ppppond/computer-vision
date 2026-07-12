def video_scan_page():
    import streamlit as st
    import cv2
    import os
    import tempfile
    import yt_dlp
    import uuid
    import json
    import datetime
    import time
    from ultralytics import YOLO
    from config import BASE_DIR
 
    st.header("📹 Video Scan (Upload / YouTube)")
 
    LOG_INTERVAL_SEC = 1.0  # บันทึกทุก 1 วินาที
 
    if "current_video_path" not in st.session_state:
        st.session_state.current_video_path = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
 
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
        model_path = st.text_input(
            "📂 Path Model",
            "runs/detect/my_custom_model/weights/best.pt",
            key="vid_model_path"
        )
        conf_threshold = st.slider(
            "🎚️ Confidence",
            0.0, 1.0, 0.25, 0.05,
            key="vid_conf"
        )
 
    with col2:
        source_type = st.radio("แหล่งที่มา", ["📁 Upload File", "🔴 YouTube URL"])
 
        if source_type == "📁 Upload File":
            video_file = st.file_uploader("🎬 อัปโหลดไฟล์วิดีโอ", type=["mp4", "mov", "avi"])
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                st.session_state.current_video_path = tfile.name
                st.success("อัปโหลดสำเร็จ")
 
        else:
            youtube_url = st.text_input("🔗 YouTube URL (รองรับ Shorts)")
            if st.button("📥 โหลดวิดีโอ"):
                if youtube_url:
                    with st.spinner("กำลังดาวน์โหลดวิดีโอ..."):
                        try:
                            if (
                                st.session_state.current_video_path
                                and os.path.exists(st.session_state.current_video_path)
                            ):
                                try:
                                    os.remove(st.session_state.current_video_path)
                                except:
                                    pass
 
                            unique_name = f"yt_{uuid.uuid4().hex}.mp4"
                            ydl_opts = {
                                "format": "bv*[ext=mp4]+ba[ext=m4a]/b",
                                "outtmpl": os.path.join(BASE_DIR, unique_name),
                                "noplaylist": True,
                                "quiet": True,
                                "merge_output_format": "mp4",
                                "extractor_args": {
                                    "youtube": {"player_client": ["android"]}
                                }
                            }
 
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(youtube_url, download=True)
                                filename = ydl.prepare_filename(info)
                                if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                                    raise Exception("ไฟล์วิดีโอว่าง (0 bytes)")
                                st.session_state.current_video_path = filename
                                st.success(f"โหลดเสร็จ: {info.get('title', 'Video')}")
 
                        except Exception as e:
                            st.error(f"โหลดไม่สำเร็จ: {e}")
 
    if st.session_state.current_video_path:
        st.info(f"📂 ไฟล์ปัจจุบัน: {os.path.basename(st.session_state.current_video_path)}")
 
        if st.button("▶️ เริ่ม / หยุด Scan", type="primary"):
            st.session_state.processing = not st.session_state.processing
 
        expand_video = st.checkbox("🔍 ขยายวิดีโอใหญ่", value=False)
        target_width = 1200 if expand_video else 600
 
        if st.session_state.processing:
            try:
                model = YOLO(model_path)
                cap   = cv2.VideoCapture(st.session_state.current_video_path)
 
                if not cap.isOpened():
                    raise Exception("ไม่สามารถเปิดไฟล์วิดีโอได้")
 
                st_frame      = st.empty()
                fps           = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_count   = 0
                names         = model.names
                class_summary = {}
                snapshot_buffer = []      # buffer snapshot ก่อน flush
                last_log_time   = time.time()
                FLUSH_EVERY     = 5       # flush ทุก 5 snapshots เพื่อลด I/O
 
                while cap.isOpened() and st.session_state.processing:
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.processing = False
                        break
 
                    frame_count += 1
                    h, w  = frame.shape[:2]
                    scale = target_width / w
                    frame = cv2.resize(frame, (target_width, int(h * scale)))
 
                    results   = model(frame, conf=conf_threshold)
                    plotted   = results[0].plot()
                    frame_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    boxes     = results[0].boxes
 
                    st_frame.image(
                        frame_rgb,
                        caption=f"Frame {frame_count} | {round(frame_count / fps, 1)}s",
                        width=target_width
                    )
 
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
                            "type"                 : "video",
                            "timestamp"            : timestamp_now,
                            "source"               : os.path.basename(st.session_state.current_video_path),
                            "model_path"           : model_path,
                            "confidence_threshold" : conf_threshold,
                            "frame"                : frame_count,
                            "video_second"         : round(frame_count / fps, 2),
                            "total_detected"       : len(detections),
                            "detections"           : detections,
                        })
                        last_log_time = now
 
                        # ✅ flush buffer → บันทึกทันทีทุก FLUSH_EVERY snapshots
                        if len(snapshot_buffer) >= FLUSH_EVERY:
                            append_log(snapshot_buffer)
                            snapshot_buffer = []
 
                cap.release()
 
                # ✅ flush snapshot ที่เหลือค้างใน buffer (กรณีหยุดกลางคัน)
                if snapshot_buffer:
                    append_log(snapshot_buffer)
                    snapshot_buffer = []
 
                # ✅ บันทึก summary session (ถ้ามีการ scan อย่างน้อย 1 frame)
                if frame_count > 0:
                    snapshot_rows.append({
                        "type"                 : "video",
                        "timestamp"            : timestamp_now,
                        "source"               : os.path.basename(st.session_state.current_video_path),
                        "model_path"           : model_path,
                        "confidence_threshold" : conf_threshold,
                        "frame"                : frame_count,
                        "video_second"         : round(frame_count / fps, 2),
                        "total_detected"       : len(detections),
                        "total_boxes"          : len(detections),  # ✅ จำนวนกรอบที่เจอ
                        "detections"           : detections,
                    })
 
                # แสดงสรุป
                st.divider()
                st.success(
                    f"✅ Scan เสร็จ | {frame_count} frames "
                    f"| เจอวัตถุทั้งหมด {sum(class_summary.values())} ครั้ง"
                )
                if class_summary:
                    st.markdown("### 📊 สรุปวัตถุที่เจอ")
                    for cls, count in sorted(class_summary.items(), key=lambda x: -x[1]):
                        st.write(f"- **{cls}**: {count} ครั้ง")
 
                log_path = os.path.join("logs", "predict_log.json")
                st.info(f"📝 บันทึก Log แล้วที่: {log_path}")
 
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {e}")
                st.session_state.processing = False
 
    else:
        st.warning("👈 กรุณาอัปโหลดไฟล์หรือใส่ลิงก์ YouTube ก่อน")
 