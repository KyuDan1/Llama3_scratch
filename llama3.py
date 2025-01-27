import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(torch.BFloat16Tensor)

DIM = 4096
FFN_DIM = 14336
N_LAYERS = 32
N_HEADS = 32 # query를 32개로 나눔.
N_KV_HEADS = 8 # grouped query attention with 8 key-value heads (speed 강점)
VOCAB_SIZE = 128256
NORM_EPS = 1e-5
ROPE_THETA = 500000.0 
MAX_BATCH_SIZE = 4
MAX_SEQ_LEN = 128
N_KV_HEAD_REP = N_HEADS // N_KV_HEADS # GQA에서 32개 query, 8개 head니까 4개 query씩 묶임.
HEAD_DIM = DIM // N_HEADS

# RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight # (2, 8, DIM) Values stays the same. We make the tensor grad_fn.
    

##############   RoPE   ##############
def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

#########    RoPE end    ##########

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, FFN_DIM, bias=False)
        self.w3 = nn.Linear(DIM, FFN_DIM, bias=False)

        self.w2 = nn.Linear(FFN_DIM, DIM, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) # *은 element multiplication (swish gelu)

"""dummy_inp = torch.randn(2,8, DIM)

feed_forward = FeedForward()

output = feed_forward(dummy_inp)
del feed_forward
print(output.shape)"""

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias = False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

        # Create empty caches for keys and values. 
        self.cache_k = torch.zeros(
            (
                MAX_BATCH_SIZE,
                MAX_SEQ_LEN,
                N_KV_HEADS,
                HEAD_DIM,
            )
        ).to(device)
        self.cache_v = torch.zeros(
            (
                MAX_BATCH_SIZE,
                MAX_SEQ_LEN,
                N_KV_HEADS,
                HEAD_DIM,
            )
        ).to(device)
    
    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape # (bsz, seqlen, DIM)
        q, k, v = self.wq(x), self.wk(x), self.wv(x) # q: (bsz,seqlen,N_HEADS*HEAD_DIM)| k,v: (bsz, seqlen, N_KV_HEADS*HEAD_DIM)
        
        q = q.view(bsz, seqlen, N_HEADS, HEAD_DIM)
        k = k.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)
        v = v.view(bsz, seqlen, N_KV_HEADS, HEAD_DIM)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(q.device)
        self.cache_v = self.cache_v.to(q.device)

        self.cache_k[ :bsz, start_pos : start_pos+seqlen] = k
        self.cache_v[ :bsz, start_pos : start_pos+seqlen] = v

        k = self.cache_k[:bsz, : start_pos+seqlen]
        v = self.cache_v[:bsz, : start_pos+seqlen]

        k = torch.repeat_interleave(
            k, dim=2, repeats=N_KV_HEAD_REP #N_HEADS // N_KV_HEADS # GQA에서 32개 query, 8개 head니까 4개 query씩 묶임.
        ) # (bsz, seqlen, N_KV_HEADS (8), HEAD_DIM)  ->  (bsz, seqlen, N_HEADS (32), HEAD_DIM)

        v = torch.repeat_interleave(
            v, dim=2, repeats=N_KV_HEAD_REP #N_HEADS // N_KV_HEADS # GQA에서 32개 query, 8개 head니까 4개 query씩 묶임.
        ) # (bsz, seqlen, N_KV_HEADS (8), HEAD_DIM)  ->  (bsz, seqlen, N_HEADS (32), HEAD_DIM)

        q = q.transpose(1,2)  # (bsz, seqlen, N_HEADS, HEAD_DIM) -> (bsz, N_HEADS, seqlen, HEAD_DIM)
        k = k.transpose(1,2)  # (bsz, seqlen, N_HEADS, HEAD_DIM) -> (bsz, N_HEADS, seqlen, HEAD_DIM)
        v = v.transpose(1,2)  # (bsz, seqlen, N_HEADS, HEAD_DIM) -> (bsz, N_HEADS, seqlen, HEAD_DIM)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # (bsz, N_HEADS, seqlen, HEAD_DIM)

        out = out.transpose(1,2).contiguous().view(bsz, seqlen, -1)
        # -1 은 나머지 데이터를 하나의 차원에 넣으라는 뜻.
        return self.wo(out) #(bsz, seqlen, DIM)



class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM, NORM_EPS)
        self.ffn_norm = RMSNorm(DIM, NORM_EPS)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x+ self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out #(bsz, seqlen, DIM)
    

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            VOCAB_SIZE, DIM
        )

        self.layers = torch.nn.ModuleList()
        for _ in range(N_LAYERS):
            self.layers.append(TransformerBlock())
        
        self.norm = RMSNorm(DIM, NORM_EPS)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False,)

        self.freqs_cis = precompute_freqs_cis(
            HEAD_DIM,
            end = MAX_SEQ_LEN,
            theta = ROPE_THETA,
        )
    @torch.inference_mode()
    def forward(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h=self.tok_embeddings(tokens) #(bsz, seqlen, DIM)
        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        mask = None
        if seqlen >1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).to(tokens.device) # upper triangle exclude
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h) # (bsz, seqlen, DIM)
        out = self.output(h).float()
        return out
    
"""dummy_tokens = torch.rand(2,8).long()
dummy_start_pos = 0
transformer = Transformer()
output = transformer(dummy_tokens, dummy_start_pos)
print(output.shape)
del transformer"""

