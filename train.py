import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import streamlit as st
import torch
from ultralytics import YOLO
from helpers.dataset_helper import create_yaml
from config import DATASET_DIR

# ==========================================
# 🛠️ Data Pipeline: Flexible Dataset
# ==========================================
def prepare_flexible_dataset(raw_dir, ready_dir, active_classes):
    """
    คัดลอกรูปภาพและกรอง Label อัตโนมัติ:
    สแกนโฟลเดอร์ทั้งหมด ถ้าเจอไฟล์ .txt จะคัดมาเฉพาะบรรทัดที่ Class ID 
    อยู่ในลิสต์ active_classes ส่วนไฟล์รูปภาพจะถูกคัดลอกไปตรงๆ
    """
    if os.path.exists(ready_dir):
        shutil.rmtree(ready_dir) # ล้างโฟลเดอร์เก่าทิ้งก่อนเพื่อความสะอาด
        
    for root, dirs, files in os.walk(raw_dir):
        # สร้างโครงสร้าง Directory ให้ตรงกับต้นฉบับ (รองรับ images/train, labels/train ฯลฯ)
        rel_path = os.path.relpath(root, raw_dir)
        dest_root = os.path.join(ready_dir, rel_path)
        os.makedirs(dest_root, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            
            # กรองข้อมูลเฉพาะไฟล์ Label (ยกเว้นไฟล์ตั้งค่าเช่น classes.txt หรือ data.yaml)
            if file.endswith('.txt') and file not in ["classes.txt", "data.yaml"]:
                with open(src_file, 'r') as f:
                    lines = f.readlines()
                    
                valid_lines = []
                for line in lines:
                    if not line.strip(): continue
                    try:
                        # ตัดข้อความและเช็ค Class ID
                        class_id = int(line.split()[0])
                        if class_id in active_classes:
                            valid_lines.append(line)
                    except ValueError:
                        pass # ข้ามบรรทัดที่รูปแบบไม่ถูกต้อง
                        
                with open(dest_file, 'w') as f:
                    f.writelines(valid_lines)
            else:
                # กรณีเป็นไฟล์รูปภาพ (.jpg, .png) ให้ Copy มาไว้เลย
                shutil.copy2(src_file, dest_file)


def train_page():
    st.header("🚀 2. Train Model")

    # -----------------------------
    # CLASS LIST
    # -----------------------------
    st.subheader("📌 Class List")

    class_text = st.text_area(
        "ใส่ชื่อคลาส (1 บรรทัด ต่อ 1 class)",
        "\n".join(st.session_state.get("class_list", [])),
        height=120,
        placeholder="person\ncar\nmotorcycle"
    )

    classes = [c.strip() for c in class_text.split("\n") if c.strip()]

    if classes:
        st.info(f"พบ {len(classes)} classes : {classes}")
    else:
        st.warning("ยังไม่ได้ใส่ class")

    # -----------------------------
    # CREATE data.yaml (ปุ่มนี้เก็บไว้ตรวจสอบได้ แต่ระบบจะสร้างอัตโนมัติก่อนเทรนอีกครั้ง)
    # -----------------------------
    if st.button("📄 สร้าง data.yaml"):
        if not classes:
            st.error("❌ กรุณาใส่ Class อย่างน้อย 1 class")
        else:
            create_yaml(DATASET_DIR, classes)
            st.session_state["class_list"] = classes
            st.success("✅ สร้าง data.yaml สำเร็จ (บนโฟลเดอร์ Raw Data)")

    st.divider()

    # -----------------------------
    # TRAIN SETTING
    # -----------------------------
    st.subheader("⚙️ Train Settings")

    epochs = st.number_input("Epochs", min_value=10, max_value=300, value=30, step=10)
    batch  = st.selectbox("Batch Size", [4, 8, 16, 32], index=0)   
    imgsz  = st.selectbox("Image Size", [416, 512, 640, 768], index=0)  
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
            
            # รัน Pipeline กรองข้อมูล
            prepare_flexible_dataset(DATASET_DIR, ready_dir, active_classes)
            
            # สร้าง data.yaml ตัวใหม่ ชี้ไปที่โฟลเดอร์ _ready
            create_yaml(ready_dir, classes)
            yaml_path = os.path.join(ready_dir, "data.yaml")

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