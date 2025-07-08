import torch
import torch.nn as nn

# Simple model for demonstration
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        model = SimpleModel()
        model = nn.DataParallel(model)  # Wrap model for multi-GPU
        model = model.cuda()
    elif torch.cuda.is_available():
        print(f"Found 1 GPU: {torch.cuda.get_device_name(0)}")
        model = SimpleModel().cuda()
    else:
        print("No CUDA GPUs found.")
        exit()

    # Dummy input
    x = torch.randn(32, 10).cuda()
    y = model(x)
    print("Output shape:", y.shape)