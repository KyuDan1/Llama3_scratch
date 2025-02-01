import torch
from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show
from torch import nn
from transformers import AutoConfig
from math import sqrt
import torch.nn.functional as F
from bertviz import head_view
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"
#show(model, "bert", tokenizer, text, display_mode = "light", layer=0, head=8)

inputs = tokenizer(text, return_tensors = "pt", add_special_tokens=False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)


config.intermediate_size = 768*2
config.num_attention_heads = 2
config.num_hidden_layers = 1

inputs_embeds = token_emb(inputs.input_ids)

query = key = value = inputs_embeds
dim_k = sqrt(key.size(-1))
scores = torch.bmm(query, key.transpose(1, 2)) / dim_k
weights = F.softmax(scores, dim=-1)
attn_outputs = torch.bmm(weights, value)

#-----------------------------------------------------------------------------
def rotate_half(x):
    
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    
    #2*2 2*1 행렬곱 연산인데, 2*1 뒤에 있는 행렬이 쫙 있고, rotate_half가 그 역할임.
    return x * cos + rotate_half(x) * sin

def get_rotary_embedding(seq_len, head_dim, base=10000, device=None):
    
    if device is None:
        device = torch.device("cpu")
    # head_dim을 2로 나눈 차원에 대해 주파수를 계산 (세타 i)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float)
    # positions와 inv_freq의 외적을 계산하여 각 위치별 각 주파수의 각도를 구합니다.
    sinusoid_inp = torch.einsum("i, j -> ij", positions, inv_freq)  # shape: [seq_len, head_dim/2]
    sin = torch.sin(sinusoid_inp)  # shape: [seq_len, head_dim/2]
    cos = torch.cos(sinusoid_inp)  # shape: [seq_len, head_dim/2]


    # 각 주파수를 head_dim의 두 개의 요소에 대응시키기 위해 반복합니다.
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)  # shape: [seq_len, head_dim]
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)  # shape: [seq_len, head_dim]
    # 차원을 맞추기 위해 unsqueeze
    sin = sin.unsqueeze(0)  # [1, seq_len, head_dim]
    cos = cos.unsqueeze(0)  # [1, seq_len, head_dim]
    return cos, sin

def scaled_dot_product_attention(query, key, value):
  dim_k = query.size(-1)
  scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
  weights = F.softmax(scores, dim=-1)
  return torch.bmm(weights, value)

class AttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)
    self.head_dim = head_dim
  def forward(self, hidden_state):
    seq_len = hidden_state.size(-2)
    #print(seq_len)
    q, k, v = self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
    #print(f"이전 q size: {q.size()}, k size: {k.size()}")
    cos, sin = get_rotary_embedding(seq_len, self.head_dim, device=hidden_state.device)
    # cos: [1, seq_len, head_dim]
    # cos: [1, seq_len, head_dim]
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    #print(f"q size: {q.size()}, k size: {k.size()}")

    attn_outputs = scaled_dot_product_attention(q, k, v)
    return attn_outputs

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
        [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, hidden_states):
    x = torch.cat([h(hidden_states) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x
  
#---------------------test------------------------------
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)
attn_output.size()

model = AutoModel.from_pretrained(model_ckpt, output_attentions=True)
#============================================================

sentence_a = "time flies like an arrow"
sentence_b = "fruit flies like a banana"
viz_inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')


attention = model(**viz_inputs).attentions
sentence_b_start = (viz_inputs.token_type_ids ==0).sum(dim=1)
tokens = tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])
head_view(attention, tokens, sentence_b_start, heads=[8])

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.gelu(x)
    x = self.linear_2(x)
    x = self.dropout(x)
    return x

feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
ff_outputs.size()

class TransformerEncoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.attention = MultiHeadAttention(config)
    self.feed_forward = FeedForward(config)

  def forward(self, x):
    hidden_state = self.layer_norm_1(x)
    x = x + self.attention(hidden_state)
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x
  

encoder_layer = TransformerEncoderLayer(config)
#encoder_layer = encoder_layer.to(device)
#inputs_embeds = inputs_embeds.to(device)

x = encoder_layer(inputs_embeds)
x.size()


class Embeddings(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
    self.dropout = nn.Dropout()

  def forward(self, input_ids):
    seq_length = input_ids.size(1)
    #position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    token_embeddings = self.token_embddings(input_ids)
    #position_embeddings = self.position_embeddings(position_ids)
    #embeddings = token_embeddings + position_embeddings
    embeddings = token_embeddings
    embeddings = self.layer_norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings
  

embedding_layer = Embeddings(config)
embedding_layer(inputs.input_ids).size()

class TransformerEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = Embeddings(config)
    self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, x):
    x = self.embeddings(x)
    for layer in self.layers:
      x = layer(x)
    return x

encoder = TransformerEncoder(config)
encoder(inputs.input_ids).size()

seq_len = inputs.input_ids.size(1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

scores.masked_fill(mask == 0, -float("inf"))

def masked_scaled_dot_product_attention(query, key, value):
  dim_k = query.size(-1)
  scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
  mask = torch.tril(torch.ones(scores.size(-1), scores.size(-1), device=scores.device)).unsqueeze(0)
  scores = scores.masked_fill(mask == 0, -float("inf"))
  weights = F.softmax(scores, dim=-1)
  return torch.bmm(weights, value)

class MaskedAttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)
    self.head_dim = head_dim
  def forward(self, hidden_state):
    seq_len = hidden_state.size(-2)
    q, k, v = self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
    cos, sin = get_rotary_embedding(seq_len, self.head_dim, device=hidden_state.device)
    # cos: [1, seq_len, head_dim]
    # cos: [1, seq_len, head_dim]
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    
    
    attn_outputs = masked_scaled_dot_product_attention(q, k, v)
    return attn_outputs

class MaskedMultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
        [MaskedAttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, hidden_states):
    x = torch.cat([h(hidden_states) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x
  
class EncDecAttentionHead(nn.Module):
  def __init__(self, embed_dim, head_dim):
    super().__init__()
    self.q = nn.Linear(embed_dim, head_dim)
    self.k = nn.Linear(embed_dim, head_dim)
    self.v = nn.Linear(embed_dim, head_dim)
    self.head_dim = head_dim
  def forward(self, enc_hidden_state, dec_hidden_state):
    dec_seq_len = dec_hidden_state.size(-2)
    enc_seq_len = enc_hidden_state.size(-2)
    q, k, v = self.q(dec_hidden_state), self.k(enc_hidden_state), self.v(enc_hidden_state)

    dec_cos, dec_sin = get_rotary_embedding(dec_seq_len, self.head_dim, device=dec_hidden_state.device)
    enc_cos, enc_sin = get_rotary_embedding(enc_seq_len, self.head_dim, device=enc_hidden_state.device)
    
    q = apply_rotary_pos_emb(q, dec_cos, dec_sin)
    k = apply_rotary_pos_emb(k, enc_cos, enc_sin)

    attn_outputs = scaled_dot_product_attention(q, k, v)
    return attn_outputs

class EncDecMultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList(
        [EncDecAttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
    )
    self.output_linear = nn.Linear(embed_dim, embed_dim)

  def forward(self, enc_hidden_state, dec_hidden_state):
    x = torch.cat([h(enc_hidden_state, dec_hidden_state) for h in self.heads], dim=-1)
    x = self.output_linear(x)
    return x
  
class TransformerDecoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
    self.layer_norm_3 = nn.LayerNorm(config.hidden_size)
    self.masked_attention = MaskedMultiHeadAttention(config)
    self.encdecattention = EncDecMultiHeadAttention(config)
    self.feed_forward = FeedForward(config)

  def forward(self, x, enc_hidden_state):
    hidden_state = self.layer_norm_1(x)
    x = x + self.masked_attention(hidden_state)
    x = self.layer_norm_2(x)
    x = x + self.encdecattention(enc_hidden_state, x)
    x = self.feed_forward(self.layer_norm_3(x))
    return x

class TransformerDecoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = Embeddings(config)
    self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, x, enc_hidden_state):
    x = self.embeddings(x)
    for layer in self.layers:
      x = layer(x, enc_hidden_state)
    return x
  
class Transformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoder = TransformerEncoder(config)
    self.decoder = TransformerDecoder(config)
    self.output = nn.Linear(config.hidden_size, config.vocab_size)
    #self.softmax = nn.Softmax(dim=-1) softmax는 transformer 안에 있으면 안된다고 함.

    
    # 가중치 초기화
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)



  def forward(self, enc_input, dec_input):
    enc_hidden_state = self.encoder(enc_input)
    dec_hidden_state = self.decoder(dec_input, enc_hidden_state)
    output = self.output(dec_hidden_state)
    #output = self.softmax(output)
    return output

transformer = Transformer(config)
transformer(inputs.input_ids, inputs.input_ids).size()
print("transformer 완료.")

text = "time flies like an arrow"
text2 = "시간이 참 빨리 간다"
inputs2 = tokenizer(text2, return_tensors = "pt", add_special_tokens=False)
transformer(inputs.input_ids, inputs2.input_ids).size()