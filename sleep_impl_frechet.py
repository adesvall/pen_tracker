import numpy as np
import torch
from numba import jit, prange
from torch.autograd import Function
import matplotlib.pyplot as plt

@jit(nopython = True)
def soft_min(vals, gamma):
    scaled_vals = np.array(vals) / (-gamma)
    max_val = np.max(scaled_vals)
    s = 0.
    for v in scaled_vals:
        s += np.exp(v - max_val)
    return - gamma * (np.log(s) + max_val)

@jit(nopython = True, parallel = True)
def compute_softfrechet(D, gamma, bandwidth):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  S = np.zeros((B, N + 2, M + 2))
  for k in prange(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):

        # Check the pruning condition
        if 0 < bandwidth < np.abs(i - j):
            continue

        softmin = soft_min([R[k, i - 1, j - 1], R[k, i - 1, j], R[k, i, j - 1]], gamma)
        S[k, i, j] = softmin
        R[k, i, j] = -soft_min([-D[k, i - 1, j - 1], -softmin], gamma)
  return R, S

@jit(nopython = True, parallel = True)
def compute_softfrechet_backward(D_, R, S, gamma, bandwidth):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E = np.zeros((B, N + 2, M + 2))
  E[:, -1, -1] = 1
  R[:, : , -1] = np.inf
  R[:, -1, :] = np.inf
  R[:, -1, -1] = R[:, -2, -2]
  S[:, : , -1] = -np.inf
  S[:, -1, :] = -np.inf
  S[:, -1, -1] = S[:, -2, -2]
  for k in prange(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):

        if np.isinf(R[k, i, j]):
            R[k, i, j] = -np.inf

        # Check the pruning condition
        if 0 < bandwidth < np.abs(i - j):
            continue

        a0 = (2 * S[k, i + 1, j] - R[k, i + 1, j] - R[k, i, j]) / gamma
        b0 = (2 * S[k, i, j + 1] - R[k, i, j + 1] - R[k, i, j]) / gamma
        c0 = (2 * S[k, i + 1, j + 1] - R[k, i + 1, j + 1] - R[k, i, j]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return (E * np.exp((D - R) / gamma))[:, 1:N + 1, 1:M + 1]

class _SoftFrechet(Function):
  @staticmethod
  def forward(ctx, D, gamma, bandwidth=np.inf):
    dev = D.device
    dtype = D.dtype
    gamma = torch.tensor([gamma]).to(dev).type(dtype) # dtype fixed
    bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    b_ = bandwidth.item()
    R, S = compute_softfrechet(D_, g_, b_)
    R = torch.tensor(R).to(dev).type(dtype)
    S = torch.tensor(S).to(dev).type(dtype)
    ctx.save_for_backward(D, R, S, gamma, bandwidth)
    return R[:, -2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, S, gamma, bandwidth = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    S_ = S.detach().cpu().numpy()
    g_ = gamma.item()
    b_ = bandwidth.item()
    F = torch.tensor(compute_softfrechet_backward(D_, R_, S_, g_, b_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(F) * F, None, None
