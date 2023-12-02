import torch
checkpoint = torch.load(
    '/local/tomlord1122/hw3_R12944044/gligen_checkpoints/diffusion_pytorch_model.bin')
print(checkpoint.keys())
