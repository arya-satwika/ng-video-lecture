# use_trained_gpt.py

import torch

from gpt2 import GPTLanguageModel, encode, decode

device = "cuda" if torch.cuda.is_available() else "cpu"

# load checkpoint saved by gpt.py
checkpoint_path = "gpt_shakespeare.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# rebuild the model architecture and load weights
model = GPTLanguageModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ---- change this prompt to anything using only characters in input.txt ----
prompt = "To be, or not to be"

# encode prompt and create context tensor
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    out = model.generate(context, max_new_tokens=200)

generated_text = decode(out[0].tolist())
print(generated_text)