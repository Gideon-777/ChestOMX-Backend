
<h1 align="center">
  🏥 3D Medical Image Analysis - Backend 🏥
</h1>

<p align="center">
  <i>A robust, containerized backend for automated 3D medical image segmentation using deep learning.</i>
</p>

<div align="center">
  
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/Flask-Backend-orange?style=flat-square" alt="Flask API" />
  <img src="https://img.shields.io/badge/Docker-Containerized-green?style=flat-square" alt="Dockerized" />
  <img src="https://img.shields.io/badge/Firebase-Authentication-yellow?style=flat-square" alt="Firebase Authentication" />
  <img src="https://img.shields.io/badge/CUDA-Enabled-blue?style=flat-square" alt="CUDA Enabled" />
  <img src="https://img.shields.io/badge/Redis-Job%20Queue-red?style=flat-square" alt="Redis RQ" />

</div>

---

## 🌟 Overview

This backend is designed for **3D medical image segmentation** using **deep learning models**. It processes **DICOM and NIfTI scans**, applies **deep learning-based segmentation**, and provides **REST API endpoints** for seamless interaction.

Built using **Flask**, it integrates:
- **A U-Net-based deep learning model** for 3D medical image analysis.
- **Firebase Authentication** for secure user management.
- **Redis RQ** for handling large inference jobs asynchronously.
- **Docker** for GPU-accelerated execution with **CUDA support**.
- **LibreOffice & Selenium** for generating medical reports.

---

## 🚀 Features

✅ **Deep Learning-Powered 3D Image Segmentation** – Uses **3D U-Net architecture** for medical image analysis.  
✅ **Asynchronous Inference Processing** – Utilizes **Redis RQ** to manage long-running tasks.  
✅ **DICOM & NIfTI Support** – Preprocessing tools for handling medical imaging formats.  
✅ **Firebase Authentication** – Ensures secure user login and role-based access control.  
✅ **Flask REST API** – Provides endpoints for model inference, results retrieval, and user management.  
✅ **Dockerized Deployment** – Easily scalable and optimized for **GPU acceleration**.  
✅ **Custom Metrics Calculation** – Calculates segmentation metrics for evaluation.  
✅ **Medical Report Generation** – Integrates **LibreOffice & Selenium** for PDF/HTML report generation.  

---

## 🏗️ System Architecture

### 🔹 **Core Components**
| Component       | Technology Used  |
|----------------|-----------------|
| **API Framework** | Flask |
| **Deep Learning Framework** | PyTorch |
| **Containerization** | Docker (NVIDIA CUDA 11.0) |
| **Task Queue** | Redis RQ |
| **Authentication** | Firebase |
| **Database** | Firestore (Firebase) |
| **Medical Imaging** | SimpleITK, Pydicom, DICOM2NIfTI |
| **Web Automation** | Selenium, Chrome WebDriver |
| **Report Generation** | LibreOffice |

### 🔹 **Processing Pipeline**
1. **Image Upload** – Users upload **DICOM/NIfTI** scans.  
2. **Preprocessing** – Converts **DICOM to NIfTI**, normalizes voxel intensities.  
3. **Model Inference** – Segmentation using a **3D U-Net**-based deep learning model.  
4. **Postprocessing** – Enhances segmentation masks and calculates metrics.  
5. **Results Storage** – Saves segmented images & reports to Firebase or provides via API.  

---

## 🛠️ Setup & Installation

### 🔹 Prerequisites
- **Docker & NVIDIA Container Toolkit** ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- **Python 3.8+**
- **CUDA-compatible GPU**
- **Firebase Admin SDK credentials**
- **Redis for task queue management**

### 🔹 Clone the Repository
```bash
git clone https://github.com/Gideon-777/ChestOMX-Backend.git
cd ChestOMX-Backend
```

### 🔹 Build & Run with Docker
```bash
docker build -t ChestOMX-Backend .
docker run --gpus all -p 5000:5000 ChestOMX-Backend
```

### 🔹 Run Without Docker
1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up Firebase**
   - Download **Firebase Admin SDK** credentials (`serviceAccountKey.json`).
   - Place it in the project directory.
4. **Start the Flask Server**
   ```bash
   python app.py
   ```

---

## 📡 API Endpoints

### 🔐 **Authentication**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/token` | Authenticate user and retrieve JWT token |
| `POST` | `/register` | Register a new user |
| `POST` | `/activate_user` | Activate a registered user |

### 🖼 **Medical Image Processing**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/prediction/upload` | Upload a DICOM/NIfTI image for processing |
| `POST` | `/api/prediction/inference` | Start inference on a medical image |
| `GET`  | `/api/image/<img_id>/<slice_id>` | Retrieve a specific image slice |
| `GET`  | `/nii/<patient_id>/<unique_id>/<model>` | Download NIfTI file for a segmented image |
| `GET`  | `/dcm/<patient_id>/<unique_id>/<model>` | Download DICOM file for a segmented image |

### 📊 **Metrics & Results**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/prediction/calc_metrics` | Compute segmentation accuracy metrics |
| `GET`  | `/api/prediction/metrics` | Retrieve calculated metrics |
| `GET`  | `/api/models` | Retrieve available models for segmentation |

### 📄 **Reports**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/report/<patient_id>/<unique_id>` | Retrieve HTML report |
| `GET`  | `/api/report_download/<patient_id>/<unique_id>` | Download HTML report |
| `GET`  | `/api/report_pdf_download/<patient_id>/<unique_id>` | Download PDF report |

---

## 📞 Contact  
For questions or feedback, feel free to reach out via:  
✉️ **Email:** mengaraaxel@gmail.com 
🔗 **GitHub:** [Author](https://github.com/Gideon-777)  

---
