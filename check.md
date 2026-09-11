# INTERVIEW_CONTEXT.md

## Purpose

I am preparing for a **co-op / internship technical interview**.

I have already submitted this project in my resume.

My main problem is that although I built/worked on the project, I am not yet confident that I can explain every technical detail if an interviewer asks deeper questions.

The goal is therefore **NOT to improve the resume right now**.

The goal is to:

> Reverse-engineer this project from the actual source code until I can confidently explain and defend what I wrote in my resume.

---

# 1. Project Overview

This project is a **Computer Vision Web Application** that combines multiple parts of the computer vision workflow into a single platform.

Main features:

* Image labeling / annotation
* Class creation
* Image preprocessing / editing
* Image preview
* Dataset preparation
* YOLO model training
* Model selection
* Training result visualization
* Image prediction
* Webcam prediction
* Video prediction
* Good / Bad classification or inspection result
* Prediction history
* Log Viewer

The application was developed mainly using:

* Python
* Streamlit
* YOLO / Ultralytics
* Computer Vision related libraries

Do not assume any implementation details from this document.

**Inspect the source code to determine what is actually used.**

---

# 2. User Workflow

The rough user workflow is:

```text
Start Application
      ↓
Labeling
      ↓
Create Classes
      ↓
Upload / Select Images
      ↓
Annotate Images
      ↓
Edit / Transform Images
      ↓
Preview Images
      ↓
Prepare Dataset
      ↓
Generate / Save YAML
      ↓
Select YOLO Model
      ↓
Train Model
      ↓
View Training Results
      ↓
Prediction
      ↓
 ┌─────────┬──────────┬─────────┐
 │         │          │         │
Image    Webcam     Video
 │         │          │
 └─────────┴──────────┘
            ↓
      Detection Result
            ↓
        Good / Bad
            ↓
        Save History
            ↓
        Log Viewer
```

This workflow is based on my current understanding.

Verify it against the actual project.

---

# 3. Labeling System

The user can perform image labeling inside the application.

Known / claimed functionality includes:

* Create classes
* Select annotation type
* Bounding box annotation
* Polygon annotation
* Resize annotation
* Move annotation
* Image transformation / preprocessing
* Preview edited images

I need to understand exactly how this works internally.

Please investigate:

```text
User interaction
      ↓
Annotation UI
      ↓
Coordinates
      ↓
Class ID
      ↓
Label representation
      ↓
Saved annotation
      ↓
Dataset
```

Questions I eventually need to be able to answer:

* How is a bounding box represented?
* What coordinates are stored?
* How does polygon annotation work?
* How are annotations moved?
* How are annotations resized?
* Where are annotations stored?
* What file format is used?
* How are classes represented?
* How does the annotation become YOLO training data?
* Are coordinates normalized?
* If so, where does normalization happen?

Do not answer these questions based only on general YOLO knowledge.

**Find the implementation in my source code first.**

---

# 4. Dataset and YAML

After labeling, the application prepares data for training.

The system also creates or saves a `.yaml` file.

I need to understand:

```text
Annotations
     ↓
Dataset Structure
     ↓
Train / Validation Data
     ↓
YAML
     ↓
YOLO Training
```

Investigate:

* Where the YAML file is generated
* Which function generates it
* What values are written into it
* Where class names come from
* Where dataset paths come from
* Whether train/validation splitting exists
* How images and labels are organized
* Whether the application uses YOLO's expected dataset structure

Show me the relevant files and functions.

---

# 5. Model Training

The application allows the user to select a model and train it.

Models mentioned in my resume include:

* YOLOv8n
* YOLOv8s
* YOLOv8m

I need to understand the exact implementation.

Trace:

```text
User selects model
       ↓
Model configuration
       ↓
Dataset YAML
       ↓
Training parameters
       ↓
YOLO training
       ↓
Output files
       ↓
Training results
```

Investigate:

* Where the model is selected
* How model names are mapped to model files
* Where YOLO is initialized
* How `.train()` or equivalent is called
* Which training parameters are configurable
* Epochs
* Image size
* Batch size
* Device
* CPU / GPU handling
* Output directories
* Saved weights
* Best model
* Last model

Do not assume all of these exist.

Report only what exists in the project.

---

# 6. YOLO Model Selection Claim

My resume mentions selecting different YOLOv8 variants such as:

```text
YOLOv8n
YOLOv8s
YOLOv8m
```

and refers to balancing:

```text
speed vs accuracy
```

This is an important interview risk.

I need you to determine:

1. Does the project actually compare these models?
2. Did I perform any benchmark?
3. Is inference speed measured?
4. Is accuracy measured?
5. Is training time measured?
6. Or does the application simply allow the user to select a model?

If the code does NOT support the stronger interpretation of the resume claim, clearly tell me.

Do not invent evidence.

---

# 7. Training Results Page

The application has a result page where the user can inspect model performance.

I need to know exactly what is displayed.

Investigate whether the application shows things such as:

* Precision
* Recall
* mAP
* mAP50
* mAP50-95
* Training loss
* Validation loss
* Confusion matrix
* Precision-recall curve
* F1 curve
* Training graphs
* Ultralytics generated files

Only report metrics actually used or displayed in the project.

For every metric found, explain later:

1. What it means
2. Why it matters
3. Where it comes from
4. How my application obtains it

---

# 8. Prediction System

The application supports prediction using:

```text
Image
Webcam
Video
```

Trace each workflow separately.

## Image

```text
Image
  ↓
Preprocessing
  ↓
Model
  ↓
Inference
  ↓
Detection
  ↓
Result
```

## Webcam

```text
Camera
  ↓
Video Frames
  ↓
Model Inference
  ↓
Detection
  ↓
Display
```

## Video

```text
Video File
   ↓
Frames
   ↓
Model
   ↓
Detection
   ↓
Processed Video / Display
```

Investigate:

* Which library reads images
* Which library reads webcam frames
* Which library reads videos
* Whether OpenCV is used
* Where YOLO inference happens
* Confidence threshold
* Bounding box rendering
* Class prediction
* Frame processing
* Resource cleanup
* Webcam release
* Video handling

---

# 9. Good / Bad Result

The application can indicate whether an inspected item is:

```text
Good
or
Bad
```

This needs careful investigation.

Determine exactly:

* What makes something Good?
* What makes something Bad?
* Is Good/Bad a YOLO class?
* Is it derived from detected classes?
* Is there custom business logic?
* Is there a confidence threshold?
* Is there a rule after model inference?

Trace the exact function responsible.

Do NOT assume this is binary classification unless the code proves it.

---

# 10. History Log / Log Viewer

The application contains a prediction history viewer.

The history may contain results from:

* Image
* Webcam
* Video

I need to understand the entire data flow:

```text
Prediction
    ↓
Result
    ↓
Save Log
    ↓
Storage
    ↓
Log Viewer
```

Investigate:

* What information is stored
* Timestamp
* Input type
* Image/video path
* Prediction result
* Detected classes
* Confidence
* Good/Bad status
* Model used

Only include fields that actually exist.

Also determine:

* Storage format
* JSON?
* CSV?
* Text file?
* Local directory?
* Session state?
* Database?

---

# 11. Local Storage and Database Claim

My resume mentions local-file storage and the possibility of migrating to a database for better scalability.

This may be asked during an interview.

Investigate the current storage architecture.

Explain:

```text
Current Storage
       ↓
What files are stored?
       ↓
Where?
       ↓
Which parts depend on local files?
```

Then help me understand:

* What limitations local-file storage creates
* What problems could occur with multiple users
* What concurrency problems might occur
* What data would make sense to move to a database
* What should remain as files
* How a database-backed design could look

Separate clearly:

```text
WHAT THE CURRENT PROJECT DOES

vs.

WHAT WOULD BE A FUTURE IMPROVEMENT
```

Do not describe proposed improvements as if they already exist.

---

# 12. CPU Training Bottleneck

My resume mentions CPU-only model training as a performance bottleneck.

Verify this from the project.

Investigate:

* How the training device is selected
* Whether CUDA is supported
* Whether MPS is supported
* Whether device is hardcoded
* Whether YOLO chooses automatically
* Whether the application actually forces CPU

Then explain why training deep-learning models on CPU can become a bottleneck compared with GPU acceleration.

Again, separate:

```text
Observed in project

vs.

General technical explanation
```

---

# 13. Resume Claims to Verify

These are the important ideas currently represented in my resume.

Treat each one as a **claim that must be verified against the source code**.

### Claim 1

Developed a computer vision annotation application integrating:

* Annotation
* Model training
* Prediction

### Claim 2

Implemented annotation tools including:

* Bounding box
* Polygon
* Resize
* Move / transform

### Claim 3

Supported image preprocessing and custom class management.

### Claim 4

Supported YOLOv8 model selection including variants such as:

* YOLOv8n
* YOLOv8s
* YOLOv8m

### Claim 5

Supported prediction using:

* Images
* Videos
* Webcam

### Claim 6

Developed a prediction history / log viewer.

### Claim 7

CPU-based training was identified as a performance bottleneck.

### Claim 8

Migrating from local-file storage to a database was proposed as a scalability improvement.

For each claim classify it as:

```text
A. Fully supported by the code

B. Partially supported

C. Technically true but wording may exaggerate what was done

D. Not supported by the current code

E. Cannot determine yet
```

Explain why.

---

# 14. Reverse-Engineering Strategy

Do not attempt to explain the whole project at once.

Reverse-engineer it layer by layer.

Use this structure:

```text
Resume Claim
      ↓
Feature
      ↓
User Flow
      ↓
Source File
      ↓
Function / Class
      ↓
Data Flow
      ↓
Library / API
      ↓
Technical Concept
      ↓
Why was it designed this way?
      ↓
Potential Interview Questions
```

---

# 15. Five Levels of Understanding

For every major feature, help me reach these five levels.

## Level 1 — What

What does the feature do?

Example:

```text
The labeling page allows the user to create annotations.
```

---

## Level 2 — How

How does the feature work from the user's perspective?

Example:

```text
Upload image
→ select class
→ draw bounding box
→ save annotation
```

---

## Level 3 — Implementation

How is it implemented in the source code?

Example structure:

```text
UI event
→ function
→ coordinates
→ state
→ saved file
```

---

## Level 4 — Concept

What technical concepts are behind the implementation?

Possible examples:

* Image coordinates
* Bounding boxes
* Normalization
* YOLO annotation format
* Object detection
* Confidence
* IoU
* Precision
* Recall
* mAP

Only connect concepts that are actually relevant.

---

## Level 5 — Why

Why was this implementation or technology used?

Examples:

* Why Streamlit?
* Why YOLO?
* Why multiple YOLO sizes?
* Why use a YAML dataset configuration?
* Why local files?
* Why would database storage improve the system?
* Why is CPU training slower?

---

# 16. Interview Preparation Goal

Eventually I want to be able to answer questions such as:

### Project overview

> Tell me about your project.

### Architecture

> How does your application work from labeling to prediction?

### Labeling

> How did you implement your annotation system?

### Dataset

> How did you convert your annotations into YOLO training data?

### YAML

> What is inside your YAML file and why does YOLO need it?

### YOLO

> Why did you use YOLO?

### Model variants

> Why did you support YOLOv8n, YOLOv8s and YOLOv8m?

### Training

> How does model training work in your application?

### Metrics

> How do you know whether your model performs well?

### Prediction

> How does webcam inference work?

### Video

> How is video inference different from image inference?

### Logs

> What information do you store in your prediction history?

### Performance

> What was the main performance bottleneck?

### Architecture improvement

> Why would you migrate from local files to a database?

### Problems

> What was the hardest technical problem you faced?

### Improvement

> If you rebuilt this project today, what would you change?

---

# 17. Important Rules for Codex

When analyzing this project:

### Rule 1 — Source code is the truth

Do not assume the project works according to this document.

Check the actual source code.

---

### Rule 2 — Do not modify the project yet

For now:

```text
READ
ANALYZE
TRACE
EXPLAIN
```

Do NOT refactor or rewrite the project unless I explicitly ask.

---

### Rule 3 — Do not invent implementation

If you cannot find something, say:

```text
I could not find this implementation in the current codebase.
```

Do not fill the gap using general knowledge.

---

### Rule 4 — Separate fact from theory

Always distinguish:

```text
Your project does this:
...

General technical concept:
...
```

---

### Rule 5 — Identify interview risks

If something written in the resume is stronger than what the project actually implements, flag it.

Use:

```text
⚠ Interview Risk
```

Then explain what an interviewer could ask.

---

### Rule 6 — Show where evidence comes from

Whenever possible give:

```text
File:
Function/Class:
Relevant flow:
```

Example:

```text
File: pages/training.py
Function: train_model()
Purpose: starts YOLO training using the selected model
```

Use the actual project paths and names.

---

# 18. Desired Output

After initially inspecting the repository, create a project map similar to:

```text
PROJECT
│
├── Labeling
│   ├── files
│   ├── functions
│   ├── stored data
│   └── concepts
│
├── Dataset
│   ├── files
│   ├── YAML
│   └── YOLO format
│
├── Training
│   ├── files
│   ├── model selection
│   ├── parameters
│   └── results
│
├── Prediction
│   ├── image
│   ├── webcam
│   └── video
│
└── History Log
    ├── storage
    ├── fields
    └── viewer
```

---

# 19. Start Here

First, inspect the repository.

Do **not** start teaching YOLO theory yet.

Start by answering:

## Step 1

What are the important files and folders in this project?

## Step 2

Which file is responsible for:

* Labeling
* Dataset preparation
* YAML generation
* Training
* Results
* Image prediction
* Webcam prediction
* Video prediction
* History / logs

## Step 3

Build the actual architecture and data-flow diagram based on the code.

## Step 4

Compare that architecture with the project description in this document.

## Step 5

List any inconsistencies or interview-risk claims.

Only after these steps should we begin detailed interview preparation.
