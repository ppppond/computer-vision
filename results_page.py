def results_page():
    import streamlit as st
    import os
    import yaml
    import torch
    from ultralytics import YOLO
    from config import DATASET_DIR

    st.header("📊 Model Performance")

    base_dir = "runs/detect"

    # -----------------------------
    # ตรวจสอบโฟลเดอร์ runs
    # -----------------------------
    if not os.path.exists(base_dir):
        st.warning("⚠️ ยังไม่มีโฟลเดอร์ runs/detect")
        return

    run_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f))
    ]

    if not run_folders:
        st.warning("⚠️ ยังไม่มีโมเดลที่ Train")
        return

    # -----------------------------
    # เลือก Run
    # -----------------------------
    selected_run = st.selectbox("📁 เลือกโมเดล", run_folders)
    run_dir = os.path.join(base_dir, selected_run)

    weight_path = os.path.join(run_dir, "weights", "best.pt")
    ready_dir = DATASET_DIR + "_ready"
    yaml_path = os.path.join(ready_dir, "data.yaml")

    # -----------------------------
    # ตรวจสอบไฟล์
    # -----------------------------
    if not os.path.exists(weight_path):
        st.error("❌ ไม่พบ best.pt ใน run นี้")
        return

    if not os.path.exists(yaml_path):
        st.error("❌ ไม่พบ data.yaml ใน DATASET_DIR กรุณาสร้างก่อน")
        return

    st.success("✅ พบ Model และ data.yaml แล้ว")

    st.divider()

    # -----------------------------
    # Evaluate
    # -----------------------------
    if st.button("🔎 Evaluate Model (Validate ใหม่)"):

         # --------------------------------------
        # ตรวจสอบ GPU (รองรับ NVIDIA / AMD (ROCm) / Mac M-Series)
        # --------------------------------------
        device = "cpu"  # Default

        # CUDA (NVIDIA) หรือ ROCm (AMD) สำหรับ Linux ที่ติดตั้ง PyTorch ที่รองรับ ROCm
        if torch.cuda.is_available():
            # ถ้าเป็น ROCm build จะมี torch.version.hip
            if getattr(torch.version, "hip", None) is not None:
                # ROCm (AMD) - ใช้ same device indexing as CUDA (0, 1, ...)
                device = 0
                st.success("🚀 ใช้ AMD GPU (ROCm)")
            else:
                device = 0
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "CUDA GPU"
                st.success(f"🚀 ใช้ NVIDIA GPU: {gpu_name}")

        elif torch.backends.mps.is_available():
            # นี่คือส่วนสำคัญสำหรับ Mac M4 / M-Series
            device = "mps"
            st.success("🚀 ใช้ Apple Metal Performance Shaders (MPS / M-Series GPU) 🍏")

        with st.spinner("⏳ กำลังประเมินผลโมเดล..."):

            model = YOLO(weight_path)

            # โหลด yaml
            with open(yaml_path, "r") as f:
                data_yaml = yaml.safe_load(f)

            model_classes = len(model.names)
            yaml_classes = len(data_yaml["names"])

            # เช็ค class mismatch
            if model_classes != yaml_classes:
                st.error("❌ จำนวน class ใน Model ไม่ตรงกับ data.yaml")
                st.write("Model classes:", model.names)
                st.write("YAML classes:", data_yaml["names"])
                st.stop()

            metrics = model.val(
                data=yaml_path,
                split="val",
                device=device,  
                workers=2, 
                save=True,
                project=run_dir,
                name="validation_results",
                exist_ok=True
            )

        st.success("🎉 ประเมินผลเสร็จแล้ว!")

        st.divider()

        # -----------------------------
        # Extract Metrics
        # -----------------------------
        results_dict = metrics.results_dict

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        map50 = results_dict.get("metrics/mAP50(B)")
        map5095 = results_dict.get("metrics/mAP50-95(B)")
        precision = results_dict.get("metrics/precision(B)")
        recall = results_dict.get("metrics/recall(B)")

        if map50 is not None:
            col1.metric("mAP@0.5", f"{map50:.4f}")
        if map5095 is not None:
            col2.metric("mAP@0.5:0.95", f"{map5095:.4f}")
        if precision is not None:
            col3.metric("Precision", f"{precision:.4f}")
        if recall is not None:
            col4.metric("Recall", f"{recall:.4f}")

        st.divider()

        # -----------------------------
        # Raw Metrics
        # -----------------------------
        st.subheader("📋 Raw Metrics")
        st.json(results_dict)

        st.divider()

        # -----------------------------
        # Validation Graphs
        # -----------------------------
        val_dir = os.path.join(run_dir, "validation_results")

        st.subheader("📊 Validation Graphs")

        image_files = [
            "confusion_matrix.png",
            "PR_curve.png",
            "F1_curve.png",
            "results.png"
        ]

        for img in image_files:
            img_path = os.path.join(val_dir, img)
            if os.path.exists(img_path):
                st.image(img_path, caption=img)

        st.divider()

        # -----------------------------
        # Download Model
        # -----------------------------
        st.subheader("📥 Download Best Model")

        with open(weight_path, "rb") as f:
            st.download_button(
                label="Download best.pt",
                data=f,
                file_name=f"{selected_run}_best.pt"
            )
