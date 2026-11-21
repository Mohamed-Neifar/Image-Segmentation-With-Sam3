# Image-Segmentation-With-Sam3

# sam3-concept-segmentation-pipeline

# 🎯 SAM 3: Segment Anything with Concepts (Local & Hybrid API)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0%20%2B%20CUDA%2012.6-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Meta AI](https://img.shields.io/badge/Model-Meta%20SAM%203-0668E1.svg?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/sam3)
[![Gradio](https://img.shields.io/badge/API-Gradio%20Client-orange.svg?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Local%20%2F%20Cloud-blueviolet.svg?style=for-the-badge)](#pipeline-architecture)

Welcome to the **SAM 3 (Segment Anything Model 3)** project repository—a Python implementation showcasing **Promptable Concept Segmentation (PCS)** developed by Meta AI[cite: 4, 18, 21].

Developed as part of the **Image and Video Processing (Master's / Data Science Project)** curriculum, this solution implements a **hybrid pipeline** combining local GPU inference (PyTorch + Triton on Windows) and remote API querying (Hugging Face / Gradio) to bypass hardware VRAM constraints.

---

## 👨‍🎓 Team & Supervision

* **Authors:** Mohamed Neifar & Mahdi Neifar[cite: 4]
* **Supervisor:** Mr. Ali Wali[cite: 4]
* **Model Studied:** Meta SAM 3 (*Segment Anything Model 3*)[cite: 4, 21]

---

## 🏛️ Pipeline Architecture

The system is engineered around a decoupled hybrid architecture that separates metadata/confidence score retrieval (via the remote server) from tensor computation and high-precision graphical rendering (locally).

```mermaid
graph TD
    subgraph User_Interface [User Input & CLI]
        IMG[Local Image .jpg/.png]
        TXT[Text Prompt e.g., 'person', 'penguin']
    end

    subgraph Hybrid_Core [Python Hybrid Engine]
        API_Client[Gradio Client API]
        Local_Model[SAM 3 Image Model - PyTorch]
        Triton_Engine[Triton Windows + MSVC C++]
    end

    subgraph Cloud_Server [Hugging Face Spaces]
        HF_SAM3[Remote Server akhaliq/sam3]
    end

    subgraph Visualization [Local Graphical Rendering]
        NumPy_Proc[Tensors & Masks Conversion]
        MatPlot[Matplotlib Alpha Blending & BBoxes]
    end

    %% Connections
    IMG & TXT --> API_Client
    IMG & TXT --> Local_Model
    
    API_Client <-->|Request / Confidence Scores| HF_SAM3
    Local_Model <-->|GPU Kernel Compilation| Triton_Engine
    
    Local_Model -->|Masks, Boxes, Scores| NumPy_Proc
    NumPy_Proc --> MatPlot
