import numpy as np
import torch
import torch.nn as nn


def main():
    n_steps = 10000
    lr = 1e-3
    l1_weight = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.load("data/states.npy")).to(device).float()
    u = torch.from_numpy(np.load("data/inputs.npy")).to(device).float()

    n_states = x.shape[-1]
    n_inputs = u.shape[-1]

    A = nn.Parameter(torch.zeros(n_states, n_states, device=device))
    B = nn.Parameter(torch.zeros(n_states, n_inputs, device=device))

    optimizer = torch.optim.Adam([A, B], lr=lr)
    for i in range(n_steps):
        optimizer.zero_grad()
        x_pred = x[:, :-1] @ A.T + u[:, :-1] @ B.T
        mse_loss = (x[:, 1:] - x_pred).pow(2).mean()
        l1_loss = A.abs().mean() + B.abs().mean()
        loss = mse_loss + l1_weight * l1_loss
        loss.backward()
        optimizer.step()
        if i % (n_steps // 10) == n_steps // 10 - 1:
            print(f"MSE={mse_loss.item():.2e} | L1={l1_loss.item():.2e}")

    np.save("data/A.npy", A.detach().cpu().numpy())
    np.save("data/B.npy", B.detach().cpu().numpy())


if __name__ == "__main__":
    main()
