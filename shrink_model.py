import torch
import os

filepath = "outputs/models/bert_epoch1_f10.4322.pt"
print(f"Original size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")

# Load only to CPU
ckpt = torch.load(filepath, map_location="cpu")

# Extract only the weights (strip the optimizer state)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    new_ckpt = ckpt["model_state_dict"]
else:
    new_ckpt = ckpt
    
# Save back a much smaller version
out_path = "outputs/models/bert_epoch1_f10.4322_shrink.pt"
torch.save(new_ckpt, out_path)
print(f"New size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")
