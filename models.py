
import torch
import torch.nn as nn
import torch.nn.functional as F


"""The script defined the core Fourier Neural Operator (FNO) architecture, to be used for anomaly detection.

The FNO architectre aims to be 

"""

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        The 2D Spectral Convolution operation (The upper path in Figure 2).
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes1 (int): Number of Fourier modes to keep in the x-direction.
            modes2 (int): Number of Fourier modes to keep in the y-direction.
        """
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Truncated modes must be <= floor(N/2) + 1
        self.modes2 = modes2

        # Scale factor for initialization
        self.scale = (1 / (in_channels * out_channels))
        
        # Complex learnable weights for the lower Fourier modes (R in the paper)
        # We need two weight tensors for 2D because rfft2 returns a specific layout for negative frequencies
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_features, weights):
        """
        Complex multiplication of Fourier modes using Einstein summation.
        (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input_features, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Apply Fast Fourier Transform (F)
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant lower Fourier modes with learned weights R
        # Initialize an empty complex tensor to store the filtered frequencies
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply top-left (positive x frequencies)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            
        # Multiply bottom-left (negative x frequencies)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Return to physical space via Inverse Fast Fourier Transform (F^-1)
        # We specify the output size `s` to ensure exact original spatial dimensions
        x_fourier = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x_fourier

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        The complete Fourier Layer block combining the Spectral Conv and Local Conv.
        """
        super(FourierLayer, self).__init__()
        
        # The Spectral/Fourier Path (top part of the diagram)
        self.fourier_path = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        
        # The Local/Linear Path W (bottom part of the diagram, implemented as a 1x1 conv)
        self.linear_path = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Calculate upper path
        x_fourier = self.fourier_path(x)
        
        # Calculate lower path
        x_local = self.linear_path(x)
        
        # Add them together and apply the non-linear activation (sigma)
        # The paper generally uses GELU
        out = F.gelu(x_fourier + x_local)
        
        return out

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    batch_size = 8
    channels = 32
    height, width = 64, 64
    
    # Keeping 12 Fourier modes in each dimension
    modes_x, modes_y = 12, 12 
    
    # Initialize the layer
    fno_layer = FourierLayer(in_channels=channels, out_channels=channels, modes1=modes_x, modes2=modes_y)
    
    # Create a dummy spatial input (e.g., fluid dynamics data)
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = fno_layer(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")