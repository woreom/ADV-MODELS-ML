import torch
from model import AutoEncoder

if __name__ == '__main__':
    model = AutoEncoder.load_from_checkpoint('./lightning_logs/version_19/checkpoints/epoch=63-step=15040.ckpt', learning_rate=0.0005)
    
    torch.save(model,'model.pt')