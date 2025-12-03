import torch
from gpt import GPTLanguageModel  # or from gpt import GPTLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("gpt_shakespeare.pt", map_location=device)
config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ixs):
    return "".join([itos[i] for i in ixs])

vocab_size = len(stoi)
model = GPTLanguageModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

prompt = "RAFI: But"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)

open('more8.txt', 'w').write(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
# # Your new input (prompt)

# # Generate continuation
# with torch.no_grad():
#     out = model.generate(context, max_new_tokens=200)[0].tolist()

# print(decode(out))