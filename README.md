# Chest X-Ray Captioning with Polished Radiology Reports

This is a **Streamlit-based demo** that generates radiology reports from chest X-ray images using a Vision Transformer (ViT) + GPT2 model and then **polishes the report** using a free Hugging Face LLM (FLAN-T5). The polished reports are professional, readable, and preserve all findings from the raw report.

---

## Features

- Upload a chest X-ray image (PNG/JPG).  
- Generate a **raw radiology report** using a ViT-GPT2 model.  
- Automatically **polish the report** into a 5–6 sentence professional format using a free LLM.  
- GPU-compatible for faster inference if available.  
- Fully free to run using open-source models.  

---

## Prerequisites

- Python 3.9+  
- GPU recommended but not required  
- Pretrained ViT-GPT2 model weights (`vit-gpt_model.pt`)  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/qubit19/chest-xray-captioning.git
cd chest-xray-captioning
```
2. **Install requirements:**

```bash
pip install -r requirements.txt
```
3. **Download the model weights and put it in chest-xray-captioning folder**

```bash
https://drive.google.com/file/d/12aUoPpAyb7WPQ4Wd1YiUQuTFwRelDzVB/view?usp=sharing
```
5. **Run command**

```bash
streamlit run app.py
'''
