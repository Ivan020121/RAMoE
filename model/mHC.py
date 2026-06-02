import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x**2).mean(-1, keepdim=True)
        out_mean = x / torch.sqrt(mean + self.eps) # root mean square
        out = self.gamma * out_mean 
        return out
    
class ManifoldHyperConnectionFuse(nn.Module):
    """
    h: hyper hidden matrix (BxLxNxD)
        B: batch_size
        L: Seq_len
        N: expansion rate
        D: feature dim
    """
    def __init__(self, dim, rate, layer_id, max_sk_it):
        super(ManifoldHyperConnectionFuse, self).__init__()

        self.n = rate
        self.dim = dim

        self.nc = self.n * self.dim
        self.n2 = self.n * self.n

        # norm flatten
        """
        Observing that RMSNorm in \mhcshort{} imposes significant latency when operating on 
        the high-dimensional hidden state $\vec{\mathbf{x}}_l \in \mathbb{R}^{1\times nC}$, 
        we reorder the dividing-by-norm operation to follow the matrix multiplication. 
        This optimization maintains mathematical equivalence while improving efficiency.
        """
        self.norm = RMSNorm(dim*rate)

        # parameters
        self.w = nn.Parameter(torch.zeros(self.nc, self.n2 + 2*self.n))
        self.alpha = nn.Parameter(torch.ones(3) * 0.01)
        self.beta = nn.Parameter(torch.zeros(self.n2 + 2*self.n) * 0.01)

        # max sinkhorn knopp iterations
        self.max_sk_it = max_sk_it

    def mapping(self, h, res_norm):
        B, L, N, D = h.shape

        # 1.vectorize
        h_vec_flat = h.reshape(B, L, N*D)
        
        # RMSNorm Fused Trick: gamma-scaling
        h_vec = self.norm.gamma * h_vec_flat

        # 2.projection
        H = h_vec @ self.w

        # RMSNorm Fused: compute r
        r = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        r_ = 1.0 / r
        
        # 4. mapping
        n = N
        H_pre = r_ * H[:,:, :n] * self.alpha[0] + self.beta[:n]
        H_post = r_ * H[:,:, n:2*n] * self.alpha[1] + self.beta[n:2*n]
        H_res = r_ * H[:,:, 2*n:] * self.alpha[2] + self.beta[2*n:]

        # 5. final constrained mapping 
        H_pre = F.sigmoid(H_pre)
        H_post = 2 * F.sigmoid(H_post)

        # 6. sinkhorn_knopp iteration
        H_res = H_res.reshape(B, L, N, N)
        H_res_exp = H_res.exp()
        with torch.no_grad():
            _, U, V = res_norm(H_res_exp.reshape(B*L, N, N), self.max_sk_it)
        # recover
        P = torch.bmm(torch.bmm(U.detach(), H_res_exp.reshape(B*L, N, N)), V.detach())
        H_res = P.reshape(B, L, N, N)

        return H_pre, H_post, H_res

    def process(self, h, H_pre, H_res):
        h_pre = H_pre.unsqueeze(dim=2) @ h
        h_res = H_res @ h
        return h_pre, h_res

    def depth_connection(self, h_res, h_out, beta):
        post_mapping = beta.unsqueeze(dim=-1) @ h_out
        out = post_mapping + h_res
        return out
