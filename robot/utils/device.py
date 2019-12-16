device = "cuda:0"

def MO(x):
    return x.detach().cpu().numpy()

def MI(x):
    return torch.tensor(x).to(device)