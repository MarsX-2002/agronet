# 🚜 AgroNet: AI Quality Control System

AgroNet is a Deep Learning-powered agricultural quality control system. It detects defects in **Potatoes**, **Carrots**, and **Lemons** using YOLOv11 and provides AI-driven safety advice via Google Gemini.

## 🌟 Features
*   **Multi-Model Detection**: Dedicated YOLOv11 models for Potato, Carrot, and Lemon.
*   **Real-Time Inference**: Process video feeds with bounding box visualization.
*   **Defect Isolation**: Automatically crops and captures images of individual defective items (ROI).
*   **AI Safety Advisor**: Multimodal Chatbot (Gemini 2.0 Flash) analyzes defect logs and images to provide safety recommendations.
*   **Video Download**: Export processed video with bounding boxes.

## 🛠️ Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/MarsX-2002/agronet.git>
    cd deeplearning
    ```

2.  **Install Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Environment Variables**
    Create a `.env` file and add your Google Gemini API Key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 🧠 Models
The system uses three custom-trained YOLOv11 models located in `detect/`:
*   `detect/potato_v11/weights/best.pt`
*   `detect/carrot_v11/weights/best.pt`
*   `detect/lemon_v11/weights/best.pt`

## 📊 Datasets
Datasets were sourced from Roboflow and augmented for class imbalance. (Excluded from repo due to size).

## 📄 License
Academic Project - Level 5/6 Deep Learning Assignment.
