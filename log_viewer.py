def log_viewer_page():
    import streamlit as st
    import json
    import os
    import pandas as pd
    from io import BytesIO

    st.header("📋 Log Viewer")

    log_path = "logs/predict_log.json"

    if not os.path.exists(log_path):
        st.warning("⚠️ ยังไม่มีไฟล์ log กรุณา Predict ก่อน")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    if not log_data:
        st.warning("⚠️ Log ว่างเปล่า")
        return

    # ==========================================
    # แปลงเป็น DataFrame
    # ==========================================
    rows = []

    for idx, record in enumerate(log_data):
        record_type = record.get("type", "image")

        if record_type in ("image", "video", "webcam"):
            source = (
                os.path.basename(record.get("image_path", ""))
                or record.get("source", "-")
            )
            detections = record.get("detections")

            if not detections:
                rows.append({
                    "_record_idx"   : idx,
                    "type"          : record_type,
                    "timestamp"     : record.get("timestamp", "-"),
                    "source"        : source,
                    "model"         : os.path.basename(record.get("model_path", "-")),
                    "conf_threshold": record.get("confidence_threshold"),
                    "frame"         : record.get("frame"),
                    "video_second"  : record.get("video_second"),
                    "class_name"    : "ไม่เจอวัตถุ",
                    "confidence"    : None,
                    "box_xyxy"      : None,
                })
            else:
                for det in detections:
                    rows.append({
                        "_record_idx"   : idx,
                        "type"          : record_type,
                        "timestamp"     : record.get("timestamp", "-"),
                        "source"        : source,
                        "model"         : os.path.basename(record.get("model_path", "-")),
                        "conf_threshold": record.get("confidence_threshold"),
                        "frame"         : record.get("frame"),
                        "video_second"  : record.get("video_second"),
                        "class_name"    : det.get("class_name", "unknown"),
                        "confidence"    : det.get("confidence"),
                        "box_xyxy"      : str(det.get("box_xyxy")),
                    })

        elif record_type in ("video_summary", "webcam_summary"):
            class_summary = record.get("class_summary", {})
            if not class_summary:
                rows.append({
                    "_record_idx"   : idx,
                    "type"          : record_type,
                    "timestamp"     : record.get("timestamp", "-"),
                    "source"        : record.get("source", "-"),
                    "model"         : os.path.basename(record.get("model_path", "-")),
                    "conf_threshold": record.get("confidence_threshold"),
                    "frame"         : record.get("total_frames_scanned"),
                    "video_second"  : None,
                    "class_name"    : "ไม่เจอวัตถุ",
                    "confidence"    : None,
                    "box_xyxy"      : None,
                })
            else:
                for cls_name, count in class_summary.items():
                    rows.append({
                        "_record_idx"   : idx,
                        "type"          : record_type,
                        "timestamp"     : record.get("timestamp", "-"),
                        "source"        : record.get("source", "-"),
                        "model"         : os.path.basename(record.get("model_path", "-")),
                        "conf_threshold": record.get("confidence_threshold"),
                        "frame"         : record.get("total_frames_scanned"),
                        "video_second"  : None,
                        "class_name"    : f"{cls_name} (×{count})",
                        "confidence"    : None,
                        "box_xyxy"      : None,
                    })

    df = pd.DataFrame(rows)

    # ==========================================
    # render_table
    # ==========================================
    def render_table(data: pd.DataFrame, tab_name: str):
        st.markdown("### 🔍 ค้นหา / กรอง")
        col1, col2, col3 = st.columns(3)

        with col1:
            search_source = st.text_input("📄 ชื่อไฟล์ต้นทาง", "", key=f"src_{tab_name}")
        with col2:
            all_classes    = sorted(data["class_name"].dropna().unique().tolist())
            selected_class = st.selectbox("📦 Class", ["ทั้งหมด"] + all_classes, key=f"cls_{tab_name}")
        with col3:
            min_conf = st.slider("🎚️ Confidence ขั้นต่ำ", 0.0, 1.0, 0.0, 0.05, key=f"conf_{tab_name}")

        filtered = data.copy()
        if search_source:
            filtered = filtered[filtered["source"].str.contains(search_source, case=False)]
        if selected_class != "ทั้งหมด":
            filtered = filtered[filtered["class_name"] == selected_class]
        if min_conf > 0:
            filtered = filtered[filtered["confidence"].notna() & (filtered["confidence"] >= min_conf)]

        display_cols = [c for c in filtered.columns if c != "_record_idx"]
        st.markdown(f"**พบ {len(filtered)} รายการ**")
        st.dataframe(filtered[display_cols], use_container_width=True)

        # ==========================================
        # ลบ Log
        # ==========================================
        st.markdown("### 🗑️ ลบ Log")

        unique_records = (
            filtered[["_record_idx", "type", "timestamp", "source", "frame", "video_second"]]
            .drop_duplicates(subset="_record_idx")
            .reset_index(drop=True)
        )

        if unique_records.empty:
            st.info("ไม่มีรายการให้ลบ")
        else:
            # ✅ label ละเอียด แยก snapshot วิดีโอ/webcam ออกจากกันได้
            def make_label(record_idx):
                row = unique_records[unique_records["_record_idx"] == record_idx].iloc[0]
                label = f"[{row['type']}]  {row['timestamp']}  |  {row['source']}"

                frame = row.get("frame")
                vsec  = row.get("video_second")

                try:
                    if frame is not None and pd.notna(frame):
                        label += f"  |  frame {int(frame)}"
                except (TypeError, ValueError):
                    pass

                try:
                    if vsec is not None and pd.notna(vsec):
                        label += f"  ({float(vsec):.1f}s)"
                except (TypeError, ValueError):
                    pass

                return label

            record_labels = {
                row["_record_idx"]: make_label(row["_record_idx"])
                for _, row in unique_records.iterrows()
            }

            selected_to_delete = st.multiselect(
                "เลือก record ที่ต้องการลบ",
                options=list(record_labels.keys()),
                format_func=lambda x: record_labels[x],
                key=f"del_{tab_name}"
            )

            col_del1, col_del2 = st.columns([1, 4])
            with col_del1:
                if st.button(
                    "🗑️ ลบที่เลือก",
                    key=f"btn_del_{tab_name}",
                    type="primary",
                    disabled=len(selected_to_delete) == 0
                ):
                    new_log = [r for i, r in enumerate(log_data) if i not in selected_to_delete]
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(new_log, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ ลบ {len(selected_to_delete)} record เรียบร้อย")
                    st.rerun()

            with col_del2:
                if st.button("⚠️ ลบทั้งหมดในแท็บนี้", key=f"btn_del_all_{tab_name}"):
                    all_idx_in_tab = set(filtered["_record_idx"].unique())
                    new_log = [r for i, r in enumerate(log_data) if i not in all_idx_in_tab]
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(new_log, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ ลบ {len(all_idx_in_tab)} record เรียบร้อย")
                    st.rerun()

        # ==========================================
        # Export
        # ==========================================
        st.markdown("### 📊 Export")
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_data = filtered[display_cols].to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️ Download CSV", csv_data, "predict_log.csv", "text/csv",
                use_container_width=True,
                key=f"csv_{tab_name}"
            )
        with col_ex2:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                filtered[display_cols].to_excel(writer, index=False, sheet_name="PredictLog")
            st.download_button(
                "⬇️ Download Excel", buffer.getvalue(), "predict_log.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"excel_{tab_name}"
            )

    # ==========================================
    # แท็บ
    # ==========================================
    tab_all, tab_image, tab_video, tab_webcam = st.tabs([
        "🗂️ ทั้งหมด", "🖼️ ภาพ", "📹 วิดีโอ", "🎥 Webcam"
    ])

    with tab_all:
        render_table(df, "all")

    with tab_image:
        df_image = df[df["type"] == "image"].reset_index(drop=True)
        if df_image.empty:
            st.info("ยังไม่มี log ของภาพ")
        else:
            render_table(df_image, "image")

    with tab_video:
        df_video = df[df["type"].isin(["video", "video_summary"])].reset_index(drop=True)
        if df_video.empty:
            st.info("ยังไม่มี log ของวิดีโอ")
        else:
            render_table(df_video, "video")

    with tab_webcam:
        df_webcam = df[df["type"].isin(["webcam", "webcam_summary"])].reset_index(drop=True)
        if df_webcam.empty:
            st.info("ยังไม่มี log ของ Webcam")
        else:
            render_table(df_webcam, "webcam")