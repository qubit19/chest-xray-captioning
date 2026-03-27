# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import GPT2Tokenizer, ViTModel, GPT2LMHeadModel, GPT2Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------- Page Config -----------------
st.set_page_config(page_title="Chest X-Ray Captioning + Polishing", layout="centered")
st.title("Chest X-Ray Captioning with Polished Reports")
st.write("Upload a chest X-ray image, generate a report, and then polish it using a free LLM!")

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# ----------------- Tokenizer and Transforms -----------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

img_size = (224, 224)
image_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ----------------- Captioning Model -----------------
class ViTGPT2Captioner(torch.nn.Module):
    def __init__(self, vit_name="google/vit-base-patch16-224", gpt2_name="gpt2"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_name)
        gpt2_config = GPT2Config.from_pretrained(gpt2_name)
        gpt2_config.add_cross_attention = True
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name, config=gpt2_config)
        self.proj = None
        if self.vit.config.hidden_size != gpt2_config.hidden_size:
            self.proj = torch.nn.Linear(self.vit.config.hidden_size, gpt2_config.hidden_size)

    def generate(self, image, max_length=100, top_p=0.9, device="cuda"):
        self.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            vit_outputs = self.vit(pixel_values=image)
            encoder_hidden_states = vit_outputs.last_hidden_state
            if self.proj:
                encoder_hidden_states = self.proj(encoder_hidden_states)

            caption = [tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id]

            for _ in range(max_length):
                input_ids = torch.tensor(caption, dtype=torch.long).unsqueeze(0).to(device)
                outputs = self.gpt2(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                probs = probs / torch.sum(probs)
                next_token = torch.multinomial(probs, 1).item()
                caption.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(caption, skip_special_tokens=True)

# ----------------- Load Captioning Model -----------------
st.info("Loading captioning model...")
caption_model = ViTGPT2Captioner().to(device)
# Load your pretrained weights if available
caption_model.load_state_dict(torch.load("vit-gpt_model.pt", map_location=device))
st.success("Captioning model loaded!")

# ----------------- LLM for Polishing -----------------
st.info("Loading polishing LLM (free Hugging Face model)...")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
st.success("Polishing LLM loaded!")

def polish_report(report: str) -> str:
    prompt = f"""
You are a professional radiology report editor. 
Your task is to **rewrite the following raw radiology findings** into a polished, professional report.

Requirements:
- Keep **all findings exactly as they appear**, do NOT omit anything.
- Use proper medical terminology and complete sentences.
- Improve readability, grammar, and clarity.
- Expand the report into 5-6 sentences without inventing new findings.

Raw findings:
{report}
"""


    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_length=256)
    polished = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return polished

# ----------------- Upload Image -----------------
uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = image_transforms(image)
    st.info("Generating report...")
    raw_report = caption_model.generate(img_tensor, device=device)
    st.success("Report generated!")

    st.subheader("Raw Radiology Report:")
    st.write(raw_report)

    st.info("Polishing report...")
    polished_report = polish_report(raw_report)
    st.success("Polished Report Ready!")
    st.subheader("Polished Radiology Report:")
    st.write(polished_report)