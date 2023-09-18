import torch
torch.manual_seed(1234)
hidden_units = 5
net = torch.nn.Sequential(
 torch.nn.Linear(4, hidden_units),
 torch.nn.ReLU(),
 torch.nn.Linear(hidden_units, 3)
)
