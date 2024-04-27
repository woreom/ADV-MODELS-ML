import torch
from torch import nn

class VQCodebook(nn.Module):
    def __init__(self, codebook_dim, num_codewords, beta=0.25):
        super(VQCodebook, self).__init__()
        self.embedding = nn.Embedding(num_codewords, codebook_dim)
        self.embedding.weight.data.uniform_(-1, 1)
        self.codebook_dim = codebook_dim
        self.num_codewords = num_codewords
        self.beta = beta
    
    def forward(self, z_e):
        batch_size, channels, w, h = z_e.shape
        if channels != self.codebook_dim:
            raise RuntimeError("VQ Codebook dimension mismatch!")
        # Reshape such that channels is last dimension, then flatten
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        z_e_flat = z_e.view(-1, self.codebook_dim)

        # calculate distances between z_e and codewords
        d = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_e_flat, self.embedding.weight.t())

        # find closest codewords
        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view(z_e.shape)

        vq_loss = torch.mean((quantized.detach() - z_e) ** 2) +  \
        self.beta * torch.mean((quantized - z_e.detach()) ** 2)

        quantized = quantized.view(batch_size, w, h, channels).permute(0, 3, 1, 2).contiguous()

        return quantized, indices, vq_loss

if __name__ == '__main__':
    x = torch.randn(64, 32, 54, 77)
    codebook = VQCodebook(32, 256)
    print(codebook(x))
        
        