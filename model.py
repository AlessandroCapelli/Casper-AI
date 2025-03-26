import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMAConfig:
	"""Configuration for the LLaMA model."""
	def __init__(self, 
				 vocab_size: int,
				 hidden_size: int = 4096, 
				 num_layers: int = 32, 
				 num_heads: int = 32,
				 intermediate_size: int = None, 
				 ffn_multiplier: int = 4, 
				 multiple_of: int = 256,
				 rms_norm_eps: float = 1e-6, 
				 max_seq_len: int = 2048,
				 num_key_value_heads: int = None,
				 tie_word_embeddings: bool = False):
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.num_key_value_heads = num_key_value_heads or num_heads  # GQA support (LLaMA-65B uses fewer num_kv_heads)
		self.rms_norm_eps = rms_norm_eps
		self.max_seq_len = max_seq_len
		self.tie_word_embeddings = tie_word_embeddings
		# Calculate intermediate feed-forward (FFN) size if not specified, using SwiGLU factor
		if intermediate_size is None:
			# Apply LLaMA scheme: hidden_dim = multiple_of * ceil((2/3 * hidden_size * ffn_multiplier) / multiple_of)
			hidden_dim = (2 * hidden_size) // 3
			hidden_dim = hidden_dim * ffn_multiplier
			# Round hidden_dim to nearest multiple of `multiple_of` (excess)
			intermediate_size = multiple_of * math.ceil(hidden_dim / multiple_of)
		self.intermediate_size = intermediate_size
		# Consistency checks
		assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
		assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"

class RMSNorm(nn.Module):
	"""Root Mean Square Layer Normalization (RMSNorm) without bias."""
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		# `weight` is a trainable scaling coefficient (bias not used in LLaMA RMSNorm)
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Calculate RMS Norm: normalize x by dividing by square root of mean of squares (variance)
		# Then multiply by trainable scaling coefficient.
		# Operate in float32 precision for stability, then convert back to original dtype.
		orig_dtype = x.dtype
		x = x.to(torch.float32)
		# mean of squares along last dimension
		variance = x.pow(2).mean(dim=-1, keepdim=True)
		x = x * torch.rsqrt(variance + self.eps)
		x = x.to(orig_dtype)
		return self.weight * x

class LLaMAAttention(nn.Module):
	"""Multi-head causal attention with rotary embeddings (RoPE)."""
	def __init__(self, config: LLaMAConfig):
		super().__init__()
		self.num_heads = config.num_heads
		self.num_kv_heads = config.num_key_value_heads
		self.head_dim = config.hidden_size // config.num_heads
		self.scale = self.head_dim ** -0.5  # scaling factor for attention scores (1/sqrt(d))
		# Q, K, V and output (O) projections. Bias disabled as per LLaMA.
		self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
		self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
		self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
		self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
		# Pre-compute cosine and sine tensors for RoPE up to max length
		# Note: base theta=10000 as per standard RoPE implementation.
		base = 10000.0
		inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
		# inv_freq has shape [head_dim/2]
		t = torch.arange(config.max_seq_len, dtype=torch.float32)  # positions [0, max_seq_len-1]
		freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
		emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim], duplicate to apply to pairs (cos, sin)
		# Buffer cosine and sine (persistent=False to not include in state_dict)
		self.register_buffer("cosine", emb.cos().unsqueeze(1), persistent=False)  # shape [max_seq_len, 1, head_dim]
		self.register_buffer("sine", emb.sin().unsqueeze(1), persistent=False)    # shape [max_seq_len, 1, head_dim]

	def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, start_pos: int = 0):
		"""
		Apply rotary embeddings to q and k tensors.
		`start_pos` is the position offset (0 for sequence start, >0 for incremental generation).
		"""
		# Extract cosine and sine for requested positions: initial shape [seq_len, 1, head_dim]
		cos = self.cosine[start_pos: start_pos + seq_len]  # [seq_len, 1, head_dim]
		sin = self.sine[start_pos: start_pos + seq_len]      # [seq_len, 1, head_dim]
		
		# Permute and expand to get shape [1, 1, seq_len, head_dim]
		cos = cos.permute(1, 0, 2).unsqueeze(0)  # from [seq_len, 1, head_dim] to [1, 1, seq_len, head_dim]
		sin = sin.permute(1, 0, 2).unsqueeze(0)  # same shape
		
		def rotate_half(x: torch.Tensor):
			# Split tensor in half along last dimension
			x1 = x[..., :x.shape[-1]//2]
			x2 = x[..., x.shape[-1]//2:]
			return torch.cat([-x2, x1], dim=-1)
		
		# Now q and k have shape [B, num_heads, seq_len, head_dim] and cos, sin are [1, 1, seq_len, head_dim]
		# Broadcast operations will work correctly
		q_rot = (q * cos) + (rotate_half(q) * sin)
		k_rot = (k * cos) + (rotate_half(k) * sin)
		return q_rot, k_rot

	def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor = None, past_kv=None, past_start_pos: int = 0):
		# hidden_states: [batch_size, seq_len, hidden_size]
		bsz, seq_len, _ = hidden_states.size()
		# Project to Q, K, V
		q = self.q_proj(hidden_states)  # [B, seq_len, num_heads*head_dim]
		k = self.k_proj(hidden_states)  # [B, seq_len, num_kv_heads*head_dim]
		v = self.v_proj(hidden_states)  # [B, seq_len, num_kv_heads*head_dim]
		# Reshape to [B, num_heads, seq_len, head_dim] (or num_kv_heads for K,V) and transpose
		q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, seq_len, head_dim]
		k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, seq_len, head_dim]
		v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, seq_len, head_dim]
		# Apply Rotary Positional Embedding to q and k
		q, k = self._apply_rotary_pos_emb(q, k, seq_len, start_pos=past_start_pos)
		# If using Grouped Query Attention (n_kv_heads < n_heads), duplicate K,V to cover all query head groups
		if self.num_kv_heads < self.num_heads:
			repeat = self.num_heads // self.num_kv_heads  # how many query head groups per kv head
			k = k.repeat_interleave(repeat, dim=1)  # [B, n_heads, seq_len, head_dim]
			v = v.repeat_interleave(repeat, dim=1)  # [B, n_heads, seq_len, head_dim]
		# If we have K,V cache (for incremental generation), concatenate them
		if past_kv is not None:
			# past_kv: tuple(prev_key, prev_value) each [B, n_heads, prev_len, head_dim]
			prev_k, prev_v = past_kv
			# Append new keys/values
			k = torch.cat([prev_k, k], dim=2)  # update seq length K
			v = torch.cat([prev_v, v], dim=2)
		# Calculate attention scores (Q @ K^T) and apply causal mask
		# q: [B, n_heads, seq_len, head_dim], k: [B, n_heads, total_len, head_dim]
		attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, n_heads, seq_len, total_len]
		# Create causal mask: prevent seeing future tokens (dimensions seq_len x total_len, considering past context included)
		# If past_kv is present, total_len = past_len + seq_len; need to mask only futures relative to last generated token.
		total_len = attn_scores.size(-1)
		# Triangular mask: True at points to mask (above diagonal)
		# The "0" diagonal of this current block starts at column = past_len
		if attn_mask is None:
			# Create a lower triangular mask for *this* seq_len window (assuming contiguity with any past context)
			mask = torch.triu(torch.ones((seq_len, total_len), device=attn_scores.device, dtype=torch.bool), diagonal=1 + past_start_pos)
			# Set -inf where mask is True (forbidden future positions)
			attn_scores.masked_fill_(mask, float('-inf'))
		else:
			# If a mask is provided (e.g. for padding), apply it directly.
			attn_scores += attn_mask  # assumes attn_mask contains -inf for masked positions, 0 elsewhere
		# Calculate attention probabilities
		attn_weights = F.softmax(attn_scores, dim=-1)
		# Apply attention to values
		attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, seq_len, head_dim]
		# Recombine heads and project with W_o
		attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [B, seq_len, n_heads*head_dim]
		output = self.o_proj(attn_output)  # [B, seq_len, hidden_size]
		return output, (k, v)  # also return k,v for caching (optional)

class LLaMAMLP(nn.Module):
	"""Feed-forward MLP with SwiGLU activation (2 linear projections + SiLU)."""
	def __init__(self, config: LLaMAConfig):
		super().__init__()
		hs = config.hidden_size
		ims = config.intermediate_size
		# Two input projections (gate and up) and one output (down). Bias = False in LLaMA.
		self.gate_proj = nn.Linear(hs, ims, bias=False)   # equivalent to W1
		self.up_proj   = nn.Linear(hs, ims, bias=False)   # equivalent to W3
		self.down_proj = nn.Linear(ims, hs, bias=False)   # equivalent to W2

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Apply SiLU activation to gate projection, multiply by up projection (GLU gating)
		return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LLaMABlock(nn.Module):
	"""LLaMA Transformer block composed of attention + MLP, with RMSNorm pre-sublayer and residual connections."""
	def __init__(self, config: LLaMAConfig):
		super().__init__()
		self.attn = LLaMAAttention(config)
		self.mlp = LLaMAMLP(config)
		# Pre-attention and pre-FFN norm (pre-normalization).
		self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.ffn_norm  = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, past_kv=None, past_start_pos: int = 0):
		# Norm input, apply attention (causal)
		h = self.attn_norm(x)
		attn_out, kv_cache = self.attn(h, attn_mask=attn_mask, past_kv=past_kv, past_start_pos=past_start_pos)
		# Residual add for attention
		x = x + attn_out
		# Norm result and apply MLP (SwiGLU)
		h2 = self.ffn_norm(x)
		ffn_out = self.mlp(h2)
		# Residual add for MLP
		x = x + ffn_out
		return x, kv_cache  # return updated state and (k,v) cache if present

class LLaMAModel(nn.Module):
	"""Complete LLaMA model with token embedding, transformer blocks and output language model head."""
	def __init__(self, config: LLaMAConfig):
		super().__init__()
		self.config = config
		# Token embedding
		self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
		# List of blocks
		self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.num_layers)])
		# Final norm (RMSNorm) before computing logits.
		self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
		# Head for language model logits (untied weights from embedding if configured)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		if config.tie_word_embeddings:
			# Optionally, tie output weights to embedding weights (not used by default in LLaMA)
			self.lm_head.weight = self.embed_tokens.weight

	def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, past_key_values: list = None):
		"""
		Execute autoregressive forward pass.
		`input_ids`: tensor [B, T] with token indices.
		`attention_mask`: mask (optional) [B, 1, 1, T_total] with 0 on available positions and -inf on masked ones.
		`past_key_values`: list (per layer) of tuples (k_prev, v_prev) for caching during generation.
		"""
		B, T = input_ids.shape
		# Get initial embeddings
		x = self.embed_tokens(input_ids)  # [B, T, hidden_size]
		new_kv_cache = []  # collect new K,V if in incremental generation
		# Apply all Transformer blocks
		for layer_idx, layer in enumerate(self.layers):
			past_kv = None
			past_len = 0
			if past_key_values is not None:
				past_kv = past_key_values[layer_idx]  # tuple of (k_prev, v_prev) for this layer
				past_len = past_kv[0].size(2)  # past context length
			# Pass through block (attention + MLP). 
			# Use start_pos = past_len to correctly calculate rotary positions for new tokens.
			x, kv_out = layer(x, attn_mask=attention_mask, past_kv=past_kv, past_start_pos=past_len)
			new_kv_cache.append(kv_out)
		# Apply final RMSNorm
		x = self.output_norm(x)
		# Compute logits via lm_head
		logits = self.lm_head(x)  # [B, T, vocab_size]
		return logits, new_kv_cache

	def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, do_sample: bool = False, 
				 temperature: float = 1.0, top_k: int = 50):
		"""
		Autoregressive generation starting from a prompt (input_ids).
		Returns the complete token sequence (prompt + generated).
		"""
		self.eval()
		# Initialize generated token list with initial prompt
		generated = input_ids.tolist()[0]  # assume batch size = 1 for simplicity
		# Cache key/values to avoid re-computing entire sequence each time
		past_key_values = None
		for _ in range(max_new_tokens):
			# Take last tokens as current input
			inp = torch.tensor([generated[-1:]], dtype=torch.long, device=next(self.parameters()).device) if past_key_values is not None else torch.tensor([generated], dtype=torch.long, device=next(self.parameters()).device)
			# Execute forward only on last token (using cache for previous context)
			with torch.no_grad():
				logits, past_key_values = self.forward(inp, past_key_values=past_key_values)
			next_token_logits = logits[0, -1, :]  # logits of last token
			if do_sample:
				# Stochastic sampling: apply temperature and top-k filtering
				probs = F.softmax(next_token_logits / temperature, dim=-1)
				if top_k is not None:
					# set to zero all probabilities except top_k highest
					topk_vals, _ = torch.topk(probs, top_k)
					min_prob = topk_vals.min()
					probs = torch.where(probs < min_prob, torch.zeros_like(probs), probs)
					probs /= probs.sum()  # renormalize
				next_token_id = torch.multinomial(probs, num_samples=1).item()
			else:
				# Greedy: choose token with maximum probability (argmax)
				next_token_id = int(torch.argmax(next_token_logits))
			generated.append(next_token_id)
			# Break if end-of-sequence token is generated (if defined in vocabulary)
			# Example: if next_token_id == tokenizer.eos_token_id: break
		return generated

	def load_pretrained(self, state_dict: dict):
		"""Load pretrained weights in state_dict format (matching keys)."""
		self.load_state_dict(state_dict)

	def generate_text(self, prompt: str) -> str:
		"""
		Simplified method to generate text from an input string.
		NOTE: This is a minimal example of tokenization and decoding. In a real application
		one should use an appropriate tokenizer (e.g. SentencePiece or HuggingFace Tokenizers)
		with an inverse dictionary to convert IDs to readable tokens.
		"""
		# Simple tokenization: split by spaces
		tokens = prompt.split()
		# Assign a fictitious token ID to each word (e.g. hash modulo vocabulary size)
		token_ids = [abs(hash(token)) % self.config.vocab_size for token in tokens]
		# Create tensor from token IDs (batch size = 1)
		input_tensor = torch.tensor([token_ids], dtype=torch.long, device=next(self.parameters()).device)
		# Generate new tokens (max_new_tokens can be adjusted)
		generated_ids = self.generate(input_tensor, max_new_tokens=20, do_sample=False)
		# Minimal decoding: convert each ID to string (in practice, you would get numbers)
		# In a real application, you need an id->token mapping
		generated_text = " ".join([str(tok) for tok in generated_ids])
		return generated_text