import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import random
import streamlit as st
import torch
from ultralytics import YOLO
from helpers.dataset_helper import create_yaml
from config import DATASET_DIR

# ==========================================
# 🛠️ Data Pipeline: Flexible Dataset
# ==========================================
def prepare_flexible_dataset(
    raw_dir,
    ready_dir,
    active_classes,
    val_ratio=0.2,
    seed=42,
):
    """
    ตรวจคู่ image/label, กรอง class และแบ่ง train/validation แบบ reproducible.

    Source structure:
        raw_dir/images/<image>
        raw_dir/labels/<label>.txt

    Ready structure:
        ready_dir/images/train
        ready_dir/images/val
        ready_dir/labels/train
        ready_dir/labels/val
    """
    image_dir = os.path.join(raw_dir, "images")
    label_dir = os.path.join(raw_dir, "labels")
    image_extensions = (".jpg", ".jpeg", ".png")

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise ValueError("ไม่พบโฟลเดอร์ datasets/images หรือ datasets/labels")

    image_by_stem = {}
    for file_name in sorted(os.listdir(image_dir)):
        stem, extension = os.path.splitext(file_name)
        if extension.lower() not in image_extensions:
            continue
        if stem in image_by_stem:
            raise ValueError(f"พบรูป basename ซ้ำกัน: {stem}")
        image_by_stem[stem] = file_name

    label_by_stem = {
        os.path.splitext(file_name)[0]: file_name
        for file_name in sorted(os.listdir(label_dir))
        if file_name.lower().endswith(".txt")
    }

    missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
    orphan_labels = sorted(set(label_by_stem) - set(image_by_stem))
    if missing_labels:
        raise ValueError(f"รูปไม่มี label: {', '.join(missing_labels[:5])}")
    if orphan_labels:
        raise ValueError(f"label ไม่มีรูป: {', '.join(orphan_labels[:5])}")

    stems = sorted(image_by_stem)
    if len(stems) < 2:
        raise ValueError("ต้องมีอย่างน้อย 2 รูปเพื่อแบ่ง train/validation")

    random.Random(seed).shuffle(stems)
    val_count = max(1, round(len(stems) * val_ratio))
    val_count = min(val_count, len(stems) - 1)
    val_stems = set(stems[:val_count])
    train_stems = set(stems[val_count:])

    if os.path.exists(ready_dir):
        shutil.rmtree(ready_dir)

    for split in ("train", "val"):
        os.makedirs(os.path.join(ready_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(ready_dir, "labels", split), exist_ok=True)

    class_counts = {
        "train": {class_id: 0 for class_id in active_classes},
        "val": {class_id: 0 for class_id in active_classes},
    }

    for stem in sorted(stems):
        split = "val" if stem in val_stems else "train"
        image_name = image_by_stem[stem]
        label_name = label_by_stem[stem]

        shutil.copy2(
            os.path.join(image_dir, image_name),
            os.path.join(ready_dir, "images", split, image_name),
        )

        valid_lines = []
        with open(os.path.join(label_dir, label_name), "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    class_id = int(float(parts[0]))
                except (ValueError, IndexError) as error:
                    raise ValueError(
                        f"class ID ไม่ถูกต้องใน {label_name} บรรทัด {line_number}"
                    ) from error

                if class_id in active_classes:
                    valid_lines.append(line.strip())
                    class_counts[split][class_id] += 1

        with open(
            os.path.join(ready_dir, "labels", split, label_name),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("\n".join(valid_lines))

    return {
        "total": len(stems),
        "train": len(train_stems),
        "val": len(val_stems),
        "class_counts": class_counts,
        "train_stems": sorted(train_stems),
        "val_stems": sorted(val_stems),
    }


def train_page():
    st.header("🚀 2. Train Model")

    # -----------------------------
    # CLASS LIST
    # -----------------------------
    st.subheader("📌 Class List")

    saved_classes = []
    if os.path.exists("classes.txt"):
        with open("classes.txt", "r", encoding="utf-8") as f:
            saved_classes = [line.strip() for line in f if line.strip()]

    class_text = st.text_area(
        "ใส่ชื่อคลาส (1 บรรทัด ต่อ 1 class)",
        "\n".join(st.session_state.get("class_list", saved_classes)),
        height=120,
        placeholder="person\ncar\nmotorcycle"
    )

    classes = [c.strip() for c in class_text.split("\n") if c.strip()]

    if classes:
        st.info(f"พบ {len(classes)} classes : {classes}")
    else:
        st.warning("ยังไม่ได้ใส่ class")

    # -----------------------------
    # PREPARE DATASET + CREATE data.yaml
    # -----------------------------
    val_ratio = st.selectbox(
        "Validation Split",
        [0.1, 0.2, 0.25, 0.3],
        index=1,
        format_func=lambda value: f"{int(value * 100)}%",
    )
    split_seed = st.number_input("Split Seed", min_value=0, value=42, step=1)

    if st.button("📄 Prepare Dataset + สร้าง data.yaml"):
        if not classes:
            st.error("❌ กรุณาใส่ Class อย่างน้อย 1 class")
        else:
            try:
                ready_dir = DATASET_DIR + "_ready"
                summary = prepare_flexible_dataset(
                    DATASET_DIR,
                    ready_dir,
                    list(range(len(classes))),
                    val_ratio=val_ratio,
                    seed=int(split_seed),
                )
                yaml_path = create_yaml(ready_dir, classes)
                st.session_state["class_list"] = classes
                st.success(
                    f"✅ เตรียม Dataset สำเร็จ: Train {summary['train']} รูป | "
                    f"Validation {summary['val']} รูป"
                )
                st.code(yaml_path)
            except ValueError as error:
                st.error(f"❌ เตรียม Dataset ไม่สำเร็จ: {error}")

    st.divider()

    # -----------------------------
    # TRAIN SETTING
    # -----------------------------
    st.subheader("⚙️ Train Settings")

    epochs = st.number_input("Epochs", min_value=10, max_value=300, value=50, step=10)
    batch  = st.selectbox("Batch Size", [4, 8, 16, 32], index=0)   
    imgsz  = st.selectbox("Image Size", [416, 512, 640, 768], index=2)  
    model_type = st.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0
    )

    st.divider()

    # -----------------------------
    # START TRAIN
    # -----------------------------
    if st.button("🚀 Start Training"):
        if not classes:
            st.error("❌ ไม่พบข้อมูลคลาส กรุณาระบุชื่อคลาสก่อนเทรน")
            return
            
        ready_dir = DATASET_DIR + "_ready"
        
        with st.spinner("⏳ กำลังเตรียมข้อมูลและกรอง Label (Pre-processing)..."):
            # สร้างลิสต์ของ Index คลาสที่ต้องการเทรน เช่น ถ้ามี 2 คลาส จะได้ [0, 1]
            active_classes = list(range(len(classes)))
            
            # รัน Pipeline ตรวจคู่ไฟล์ กรอง class และแบ่ง train/validation
            try:
                summary = prepare_flexible_dataset(
                    DATASET_DIR,
                    ready_dir,
                    active_classes,
                    val_ratio=val_ratio,
                    seed=int(split_seed),
                )
            except ValueError as error:
                st.error(f"❌ เตรียม Dataset ไม่สำเร็จ: {error}")
                return
            
            # สร้าง data.yaml ตัวใหม่ ชี้ไปที่โฟลเดอร์ _ready
            yaml_path = create_yaml(ready_dir, classes)

            st.success(
                f"📦 Dataset: Train {summary['train']} รูป | "
                f"Validation {summary['val']} รูป"
            )

        # ✅ ลบ results.csv เก่าก่อน train
        results_csv = os.path.join("runs", "detect", "my_custom_model", "results.csv")
        if os.path.exists(results_csv):
            os.remove(results_csv)

        # ✅ เลือก device อัตโนมัติ
        device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = 0
            try:
                gpu_name  = torch.cuda.get_device_name(0)
                free_mem  = torch.cuda.mem_get_info(0)[0] / 1024**3
                st.success(f"🚀 ใช้ GPU: {gpu_name} | VRAM ว่าง: {free_mem:.1f} GB")
            except Exception:
                st.success("🚀 ใช้ CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            st.success("🚀 ใช้ Apple MPS")
        else:
            st.info("💻 ใช้ CPU (ไม่พบ GPU)")

        with st.spinner("⏳ กำลัง Train Model... (ไปชงกาแฟรอได้เลย)"):
            model = YOLO(model_type)
            model.train(
                data=yaml_path,        # ชี้ไปที่ yaml ของโฟลเดอร์ _ready
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                project="runs/detect",
                name="my_custom_model",
                exist_ok=True,
                workers=0,      
                cache=False,             
            )

        st.success("🎉 Train เสร็จเรียบร้อย!")
