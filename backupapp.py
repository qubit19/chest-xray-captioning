# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import GPT2Tokenizer, ViTModel, GPT2LMHeadModel, GPT2Config
import torch.nn as nn

# ----------------- Page Config -----------------
st.set_page_config(page_title="Chest XRay Captioning", layout="centered")

st.title("Chest X-Ray Captioning Demo")
st.write("Upload a chest X-ray image, and the model will generate a radiology report!")

# ----------------- Device -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# ----------------- Tokenizer -----------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ----------------- Image Transform -----------------
img_size = (224, 224)
image_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ----------------- Model Definition -----------------
class ViTGPT2Captioner(nn.Module):
    def __init__(self, vit_name="google/vit-base-patch16-224", gpt2_name="gpt2"):
        super().__init__()
        # Encoder
        self.vit = ViTModel.from_pretrained(vit_name)
        
        # Decoder with cross-attention
        gpt2_config = GPT2Config.from_pretrained(gpt2_name)
        gpt2_config.add_cross_attention = True
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name, config=gpt2_config)

        # Optional projection
        if self.vit.config.hidden_size != gpt2_config.hidden_size:
            self.proj = nn.Linear(self.vit.config.hidden_size, gpt2_config.hidden_size)
        else:
            self.proj = None

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        vit_outputs = self.vit(pixel_values=pixel_values)
        encoder_hidden_states = vit_outputs.last_hidden_state
        if self.proj:
            encoder_hidden_states = self.proj(encoder_hidden_states)
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )
        return outputs

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

# ----------------- Load Model -----------------
st.info("Loading model... This may take a minute!")
model = ViTGPT2Captioner().to(device)
# If you have pretrained weights, load here:
model.load_state_dict(torch.load("vit-gpt_model.pt", map_location=device))
st.success("Model loaded!")

# ----------------- Upload Image -----------------
uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform image
    img_tensor = image_transforms(image)

    # Generate Caption
    st.info("Generating report...")
    caption = model.generate(img_tensor, device=device)
    st.success("Report generated!")
    st.subheader("Generated Radiology Report:")
    st.write(caption)