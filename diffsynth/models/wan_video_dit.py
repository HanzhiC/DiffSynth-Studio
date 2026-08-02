import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward
from .wantodance import WanToDanceRotaryEmbedding, WanToDanceMusicEncoderLayer

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class ConditionVideoAdapter(nn.Module):
    """Pools a variable-length condition video's VAE-latent patches into a fixed-per-frame number
    of tokens (`tokens_per_frame`, PER latent frame -- NOT collapsed across frames), plus a learned
    absolute positional embedding per latent-frame index, so the output stays a
    *temporally-structured* sequence (`F_latent * tokens_per_frame` tokens) rather than one global
    pooled bag. This is a deliberate revision of an earlier design that pooled ALL patches (across
    every frame) into one small, globally-fixed token count -- that design decoupled the token count
    from the condition video's length perfectly, but also destroyed every patch's frame identity
    before `ConditionVideoCrossAttention` (below) ever saw it, so the DiT's per-target-frame query
    (which already has its own temporal identity via RoPE) had no way to learn "attend to the
    condition-video position matching my own progress" -- there was no frame-tagged signal left on
    the key/value side to correlate against, no matter how the DiT side was trained. Verified
    empirically too: attention visualizations from the old design only ever showed elevated weight
    at the condition video's start/end (an artifact of always-included episode-boundary frames
    being visually distinctive, not real progress-tracking).

    This revision follows the mechanism a related work uses for a similar problem -- Behavior
    Prompting Policy (arXiv:2606.30457, real-stanford/behavior_prompting), which conditions a
    manipulation policy on a full human/robot demonstration and reports attention that tracks task
    progress. Their `TransformerDecoderWithAttn`/`ICRTPromptObsEncoder` (see
    `transformer_decoder_with_attention.py`/`prompt_obs_encoder.py` in that repo) adds a plain
    **learned absolute positional embedding, one vector per index along the demo sequence**,
    directly to the demo/prompt tokens before cross-attention (`prompt_tokens +
    self.prompt_pos_emb[:, :prompt_tokens.shape[1], :]`) -- no explicit progress estimator, no
    attention masking trick; progress-aware attention emerges from training once that positional
    signal is present and the demo tokens keep their per-position identity into the cross-attention.
    `frame_pos_emb` below is the direct analog for our condition video, added per latent-frame index
    instead of per demo-timestep.

    Still lets a condition video of any length feed the DiT without a frame-aligned channel-concat
    (unlike this same project's own `WanVideoUnit_FunControl`/`control_video`) -- see class
    docstring history/git blame for the fixed-token-count precursor and
    arXiv:2512.02015 (Edit-by-Track)'s "condition via attention, not channel-concat" principle this
    still follows, just with per-frame instead of fully-global pooling.

    Not part of any pretrained Wan checkpoint's state dict -- attached post-hoc to an
    already-loaded `WanModel` via `WanModel.add_condition_video_adapter`, not resolved through
    `configs/model_configs.py`'s state-dict-hash registry, since there is no pretrained checkpoint
    whose weights this module could be loaded from.
    """

    def __init__(self, in_channels, dim, tokens_per_frame=8, num_heads=8, patch_size=(1, 2, 2), max_latent_frames=128):
        super().__init__()
        self.patch_embedding = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.tokens_per_frame = tokens_per_frame
        self.max_latent_frames = max_latent_frames
        # Shared across every latent frame (not one set of weights per frame) -- keeps param count
        # independent of max_latent_frames, matching the per-frame pooling being the SAME learned
        # operation applied at each frame, only distinguished afterward by frame_pos_emb below.
        self.query_tokens = nn.Parameter(torch.randn(1, tokens_per_frame, dim) * 0.02)
        # The actual per-frame-position signal (see class docstring) -- sliced to the condition
        # video's real F_latent at forward time, same pattern as BPP's `prompt_pos_emb` slicing.
        self.frame_pos_emb = nn.Parameter(torch.randn(1, max_latent_frames, 1, dim) * 0.02)
        self.norm_tokens = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

        # Attention-capture instrumentation (off by default, zero cost/behavior change when unset --
        # see WanModel.enable_condition_attention_capture). This module runs once per generation
        # (before the denoising loop, not per-step), so a single snapshot is enough -- no history
        # list needed, unlike ConditionVideoCrossAttention below.
        self.capture_attention = False
        self.last_pooling_attn = None  # [B*F, tokens_per_frame, H*W] once captured -- purely
        # spatial (within-frame) attention now, since pooling never crosses frame boundaries.
        self.last_patch_grid = None  # (F, H, W) patch-grid shape, for mapping patches back to frames

    def forward(self, latents):
        """latents: [B, C, F, H, W] (variable F/H/W) -> [B, F*tokens_per_frame, dim]"""
        x = self.patch_embedding(latents)  # [B, dim, F, H, W]
        b, _, f, h, w = x.shape
        if f > self.max_latent_frames:
            raise ValueError(
                f"condition video has {f} VAE-latent frames, exceeding max_latent_frames="
                f"{self.max_latent_frames} this adapter was constructed with -- pass a larger "
                f"max_latent_frames to add_condition_video_adapter, or shorten condition_num_frames."
            )
        # Fold the frame axis into the batch axis: one batched nn.MultiheadAttention call pools
        # every frame independently (frames never attend to each other's patches here) instead of
        # a Python loop over frames.
        x = rearrange(x, "b d f h w -> (b f) (h w) d")
        context = self.norm_context(x)
        queries = self.norm_tokens(self.query_tokens.expand(context.shape[0], -1, -1))
        if self.capture_attention:
            out, attn_weights = self.attn(
                query=queries, key=context, value=context, need_weights=True, average_attn_weights=True
            )
            self.last_pooling_attn = attn_weights.detach()
            self.last_patch_grid = (f, h, w)
        else:
            out, _ = self.attn(query=queries, key=context, value=context, need_weights=False)
        out = rearrange(out, "(b f) k d -> b f k d", b=b, f=f)
        out = out + self.frame_pos_emb[:, :f, :, :]
        out = rearrange(out, "b f k d -> b (f k) d")
        return out


class EpisodeControlVideoResampler(nn.Module):
    """Resamples a variable-length condition (episode demo) video's `F_cond` VAE-latent frames down
    to exactly `F_target` frames (the target clip's own latent frame count), so the result can
    channel-concat through `WanVideoUnit_FunControl`'s pretrained `control_video` pathway even when
    the raw episode video and the target clip don't share a frame count -- an alternative to forcing
    `DATA.condition_num_frames == DATA.clip_length` at the dataset level (see ego-moma's
    `VideoGenModel.control_video_from_episode` docstring: when the two already match, this module
    is skipped entirely and the existing direct-encode path is used unchanged).

    Mechanism: each of the `F_target` output frames is produced by a cross-attention-weighted blend
    of the condition video's OWN `F_cond` latent frames (full spatial content preserved -- the
    attention weights are computed from small pooled per-frame descriptors, but then applied to the
    full-resolution latents themselves, not to the descriptors), where the query for output frame
    `i` is `target_pos_emb[i] + proj(reference_descriptor)` -- i.e. BOTH a learned per-target-slot
    position and the episode's own reference (current/start) frame jointly decide which parts of
    the demo are relevant for that output slot. `reference_latent` is `WanVideoUnit_FunReference`'s
    own output (reused directly, no extra VAE encode needed for it).

    Gated via a plain per-target-frame-slot scalar (`self.gate`, shape `[max_target_frames]`, NOT a
    dynamic per-token function this time): `output = reference_latent + gate * (blended -
    reference_latent)`, a convex combination when `gate` in `[0, 1]` (not hard-constrained to that
    range, but that is the intended operating regime). `gate_init` sets its initial value directly
    (no sigmoid reparameterization -- a plain, directly-readable-and-settable parameter, matching
    ConditionVideoCrossAttention's own history: a single global dynamic-per-token gate turned out to
    be hard to monitor meaningfully on its own; this module's gate has only `max_target_frames`
    (~10) values total, small enough to track directly without needing a realized-forward-value
    trick). At `gate=0` (any target frame slot), that slot's output is EXACTLY `reference_latent` --
    i.e. at `gate_init=0` construction reduces to "control_video = reference frame repeated",
    itself a real, in-distribution, non-degenerate starting point (not noise) for the pretrained,
    ungated `patch_embedding` this feeds into -- deliberately safer than feeding that pathway a
    freshly-initialized module's raw output the way `ConditionVideoCrossAttention` now does for its
    OWN (residual-additive, not channel-concat) branch.

    Not part of any pretrained Wan checkpoint's state dict -- attached post-hoc via
    `WanModel.add_episode_control_video_resampler`, same convention as `add_condition_video_adapter`.

    **Kept in float32, deliberately NOT cast to the backbone's bf16** (see
    `add_episode_control_video_resampler`): this whole model trains natively in bf16 with no FP32
    master-weight copy, and AdamW's per-step update magnitude is roughly `lr` (~1e-4) regardless of
    a parameter's own gradient scale (that's the point of Adam's normalization) -- for a parameter
    sitting at a value where bf16's absolute precision (~value/128) exceeds that update size, the
    update silently rounds to exactly zero every step, forever, no matter how large or consistent
    the true gradient is. Confirmed empirically: on a real training run, `self.gate` (value ~0.05,
    bf16 ULP there ~0.0004) had the LARGEST gradient of every parameter in this module
    (grad_abs_mean ~2.5e-4, ~100x every other parameter's) yet never moved a single bit across 3000
    steps, while `k.bias` (value near 0, where bf16 has fine absolute precision) moved substantially
    despite a ~1000x SMALLER gradient -- ruling out "vanishing gradient" and pointing squarely at
    this update-rounds-to-zero mechanism (also visibly affected `norm_query`/`norm_key.weight`,
    both initialized to 1.0). `forward` upcasts its bf16 inputs to float32 internally and downcasts
    the output back at the very end, so this module's own optimizer step always happens in full
    precision regardless of what dtype the rest of the pipeline runs in.
    """

    def __init__(self, in_channels, attn_dim=256, num_heads=8, patch_size=(1, 2, 2),
                 max_target_frames=128, max_condition_frames=256, gate_init=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.max_target_frames = max_target_frames
        self.max_condition_frames = max_condition_frames
        # Per-frame descriptor extractor -- coarse, spatially-pooled (mean over H,W after the patch
        # conv), used ONLY to decide attention weights, not to reconstruct output content. Shared
        # between condition frames and the reference frame (same conv, same channel count).
        self.frame_descriptor = nn.Conv3d(in_channels, attn_dim, kernel_size=patch_size, stride=patch_size)
        self.target_pos_emb = nn.Parameter(torch.randn(1, max_target_frames, attn_dim) * 0.02)
        self.reference_proj = nn.Linear(attn_dim, attn_dim)
        self.norm_query = nn.LayerNorm(attn_dim)
        self.norm_key = nn.LayerNorm(attn_dim)
        self.q = nn.Linear(attn_dim, attn_dim)
        self.k = nn.Linear(attn_dim, attn_dim)
        # Plain per-target-frame-slot scalar gate -- see class docstring. NOT sigmoid-reparameterized
        # (unlike ConditionVideoCrossAttention's earlier per-token design): directly settable/
        # readable, small enough (max_target_frames values) to track meaningfully as-is.
        self.gate = nn.Parameter(torch.full((max_target_frames,), float(gate_init)))

    def _frame_descriptors(self, latents):
        """latents: [B, C, F, H, W] -> [B, F, attn_dim] (mean-pooled over the patch grid)."""
        x = self.frame_descriptor(latents)  # [B, attn_dim, F, H', W']
        return x.mean(dim=(3, 4)).transpose(1, 2)  # [B, F, attn_dim]

    def forward(self, condition_latents, reference_latent, num_target_frames):
        """`condition_latents`: [B, C, F_cond, H, W]. `reference_latent`: [B, C, 1, H, W] (
        `WanVideoUnit_FunReference`'s own output). `num_target_frames`: the target clip's own
        latent frame count (`F_target`) -- this module has no other way to know it, since
        `condition_latents`/`reference_latent` alone don't carry that information. Returns
        `[B, C, num_target_frames, H, W]`, cast back to the INPUT dtype (see class docstring for
        why this module's own parameters/computation stay in float32 regardless)."""
        output_dtype = condition_latents.dtype
        condition_latents = condition_latents.float()
        reference_latent = reference_latent.float()
        b, c, f_cond, h, w = condition_latents.shape
        if f_cond > self.max_condition_frames:
            raise ValueError(
                f"condition video has {f_cond} VAE-latent frames, exceeding max_condition_frames="
                f"{self.max_condition_frames} this resampler was constructed with."
            )
        if num_target_frames > self.max_target_frames:
            raise ValueError(
                f"target clip has {num_target_frames} VAE-latent frames, exceeding "
                f"max_target_frames={self.max_target_frames} this resampler was constructed with."
            )
        condition_desc = self._frame_descriptors(condition_latents)  # [B, F_cond, attn_dim]
        reference_desc = self._frame_descriptors(reference_latent).squeeze(1)  # [B, attn_dim]

        queries = self.target_pos_emb[:, :num_target_frames, :] + self.reference_proj(reference_desc).unsqueeze(1)
        queries = self.norm_query(queries)  # [B, F_target, attn_dim]
        keys = self.norm_key(condition_desc)  # [B, F_cond, attn_dim]

        head_dim = self.attn_dim // self.num_heads
        q = rearrange(self.q(queries), "b t (n d) -> b n t d", n=self.num_heads)
        k = rearrange(self.k(keys), "b s (n d) -> b n s d", n=self.num_heads)
        attn_weights = torch.softmax(q @ k.transpose(-1, -2) / (head_dim ** 0.5), dim=-1)
        attn_weights = attn_weights.mean(dim=1)  # [B, F_target, F_cond] -- already float32 throughout

        blended = torch.einsum("bts,bcshw->bcthw", attn_weights, condition_latents)  # [B, C, F_target, H, W]
        reference_expanded = reference_latent.expand(b, c, num_target_frames, h, w)
        gate = self.gate[:num_target_frames].view(1, 1, num_target_frames, 1, 1)
        output = reference_expanded + gate * (blended - reference_expanded)
        return output.to(output_dtype)


class ConditionVideoCrossAttention(nn.Module):
    """A new, separate cross-attention branch for injecting `ConditionVideoAdapter`'s pooled
    tokens into a `DiTBlock` -- deliberately NOT the same code path as `CrossAttention`'s existing
    `has_image_input` CLIP-feature fusion above, which shares Wan's own pretrained output
    projection `self.o` and was trained end-to-end by the original authors, so it isn't zero-init
    and isn't a template reusable for a module we're adding fresh post-hoc.

    NO gate at all -- deliberately removed (this class went through two gated designs first: a
    single global scalar per block, then a dynamic per-token `sigmoid(gate_proj(x))`; see git
    history). Both were empirically observed, on real training runs of this exact module, to
    suppress the branch rather than learn to use it: the scalar gate never moved off its zero-init
    after 17k steps (pure noise, no directional signal); the dynamic per-token gate DID receive a
    real, directional gradient (gate_proj.weight's norm grew steadily) but used it to learn to
    shrink the realized gate toward 0 for real inputs (from ~0.027 at init down to ~0.00014 by step
    1100, monotonically, no sign of self-correcting) -- i.e. the model was actively learning to
    turn this branch OFF, not gradually learn to use it. Rather than keep chasing a gating scheme
    that the optimizer keeps using against us, this now matches arXiv:2512.02015 (Edit-by-Track)'s
    own injection for its (different, track-based) conditioning signal: a plain, ungated
    element-wise addition (`τ` added directly to the video tokens, no learned gate) -- forcing the
    branch to matter from step 1 rather than giving the optimizer an easy escape hatch to ignore
    it. This is a real, deliberate tradeoff, not a free improvement: `q`/`k`/`v`/`o` are still
    randomly initialized (nothing here is part of any pretrained checkpoint), so this branch now
    injects a genuinely random signal at FULL strength into every block's residual stream from
    step 1 -- unlike the gated designs, there is no "close to unperturbed at step 0" guarantee
    anymore, and the pretrained backbone's own quality at the very start of training may visibly
    degrade until this branch's weights converge to something useful. `last_branch_norm`/
    `last_input_norm` (populated every forward call, see below) are this class's replacement for
    the old gate-value monitoring (`ego-moma`'s `video_gen_algo.py::_log_condition_gate_stats`) --
    with no explicit gate to read, "how much is this branch actually contributing" is measured
    directly as this branch's output norm relative to the residual stream's own norm instead."""

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.attn = AttentionModule(num_heads)

        # Attention-capture instrumentation (off by default -- see
        # WanModel.enable_condition_attention_capture). Unlike ConditionVideoAdapter above, this
        # module runs once per DiT block per denoising *step*, so captured weights are appended to
        # a history list rather than overwritten -- reset_capture() clears it before a fresh
        # generation call.
        self.capture_attention = False
        self.captured_attn_history = []

    def reset_capture(self):
        self.captured_attn_history = []

    def _attend_with_capture(self, q, k, v):
        # AttentionModule.forward (flash_attention) never materializes attention weights (that's
        # the point of a fused kernel) -- fall back to a plain, unfused softmax(qk^T/sqrt(d))v so we
        # can record the weight matrix. Only taken when capture_attention is set (evaluation/
        # visualization runs), never during normal train/inference.
        b, sq, _ = q.shape
        sk = k.shape[1]
        head_dim = self.dim // self.num_heads
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads).float()
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads).float()
        v_ = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads).float()
        attn_weights = torch.softmax(q @ k.transpose(-1, -2) / (head_dim ** 0.5), dim=-1)  # [b, n, sq, sk]
        out = attn_weights @ v_  # [b, n, sq, d]
        out = rearrange(out, "b n s d -> b s (n d)").to(v.dtype)
        # Mean over heads and query positions -> [b, sk] (sk == num_tokens), one snapshot per
        # forward call (i.e. per DiT block per denoising step).
        self.captured_attn_history.append(attn_weights.mean(dim=(1, 2)).detach())
        return out

    def forward(self, x, condition_tokens):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(condition_tokens))
        v = self.v(condition_tokens)
        if self.capture_attention:
            out = self._attend_with_capture(q, k, v)
        else:
            out = self.attn(q, k, v)
        branch_out = self.o(out)
        # Cheap running stats (mean per-token L2 norm this forward call) for training-time
        # monitoring (see ego-moma's video_gen_algo.py::_log_condition_gate_stats) -- with no
        # explicit gate anymore, this is the closest available substitute for "how much is this
        # branch actually contributing right now": branch_out's own norm relative to x's (the
        # residual stream it's being added into). Needs no unfused-attention fallback/
        # capture_attention flag, unlike captured_attn_history -- always populated during ordinary
        # training.
        self.last_branch_norm = branch_out.detach().float().norm(dim=-1).mean()
        self.last_input_norm = x.detach().float().norm(dim=-1).mean()
        return branch_out


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        # Set by WanModel.add_condition_video_adapter (post-hoc, not a constructor arg) -- see
        # ConditionVideoCrossAttention's docstring.
        self.cond_cross_attn = None

    def forward(self, x, context, t_mod, freqs, condition_video_tokens=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        if condition_video_tokens is not None and self.cond_cross_attn is not None:
            x = x + self.cond_cross_attn(self.norm3(x), condition_video_tokens)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


def wantodance_torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = wantodance_torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


class WanToDanceInjector(nn.Module):
    def __init__(self, all_modules, all_modules_names, dim=2048, num_heads=32, inject_layer=[0, 27]):
        super().__init__()
        self.injected_block_id = {}
        injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'root.transformer_blocks.{inject_id}' == mod_name:
                        self.injected_block_id[inject_id] = injector_id
                        injector_id += 1

        self.injector = nn.ModuleList(
            [
                CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [
                nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6,)
                for _ in range(injector_id)
            ]
        )


class WanModel(torch.nn.Module):

    _repeated_blocks = ["DiTBlock"]

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        in_dim_ref_conv: int = 16,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads

        if wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            end = int(22350 / 8 + 0.5) # 149f * 30fps * 5s = 22350
            self.freqs = precompute_freqs_cis_3d(head_dim, end=end)
        else:
            self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

        # Set by add_condition_video_adapter (post-hoc opt-in, not a constructor arg -- see
        # ConditionVideoAdapter's docstring for why this isn't wired through model_configs.py).
        self.has_condition_video_input = False
        self.condition_video_adapter = None
        # Set by add_episode_control_video_resampler (post-hoc opt-in -- see
        # EpisodeControlVideoResampler's docstring).
        self.episode_control_video_resampler = None

        self.prepare_wantodance(in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
                                wantodance_enable_music_inject, wantodance_music_inject_layers, wantodance_enable_refimage, wantodance_enable_refface,
                                wantodance_enable_global, wantodance_enable_dynamicfps, wantodance_enable_unimodel)

    def prepare_wantodance(
        self,
        in_dim, dim, num_heads, has_image_pos_emb, out_dim, patch_size, eps,
        wantodance_enable_music_inject: bool = False,
        wantodance_music_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27],
        wantodance_enable_refimage: bool = False,
        wantodance_enable_refface: bool = False,
        wantodance_enable_global: bool = False,
        wantodance_enable_dynamicfps: bool = False,
        wantodance_enable_unimodel: bool = False,
    ):
        if wantodance_enable_music_inject:
            all_modules, all_modules_names = wantodance_torch_dfs(self.blocks, parent_name="root.transformer_blocks")
            self.music_injector = WanToDanceInjector(all_modules, all_modules_names, dim=dim, num_heads=num_heads, inject_layer=wantodance_music_inject_layers)
        if wantodance_enable_refimage:
            self.img_emb_refimage = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_refface:
            self.img_emb_refface = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if wantodance_enable_global or wantodance_enable_dynamicfps or wantodance_enable_unimodel:
            music_feature_dim = 35
            ff_size = 1024
            dropout = 0.1
            latent_dim = 256
            nhead = 4
            activation = F.gelu
            rotary = WanToDanceRotaryEmbedding(dim=latent_dim)
            self.music_projection = nn.Linear(music_feature_dim, latent_dim)
            self.music_encoder = nn.Sequential()
            for _ in range(2):
                self.music_encoder.append(
                    WanToDanceMusicEncoderLayer(
                        d_model=latent_dim,
                        nhead=nhead,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=True,
                        rotary=rotary,
                        device='cuda',
                    )
                )
        if wantodance_enable_unimodel:
            self.patch_embedding_global = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if wantodance_enable_unimodel:
            self.head_global = Head(dim, out_dim, patch_size, eps)
        self.wantodance_enable_music_inject = wantodance_enable_music_inject
        self.wantodance_enable_refimage = wantodance_enable_refimage
        self.wantodance_enable_refface = wantodance_enable_refface
        self.wantodance_enable_global = wantodance_enable_global
        self.wantodance_enable_dynamicfps = wantodance_enable_dynamicfps
        self.wantodance_enable_unimodel = wantodance_enable_unimodel

    def wantodance_after_transformer_block(self, block_idx, hidden_states):
        if self.wantodance_enable_music_inject:
            if block_idx in self.music_injector.injected_block_id.keys():
                audio_attn_id = self.music_injector.injected_block_id[block_idx]
                audio_emb = self.merged_audio_emb  # b f n c
                num_frames = audio_emb.shape[1]
                input_hidden_states = hidden_states.clone()  # b (f h w) c
                input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)
                attn_hidden_states = self.music_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
                audio_emb = rearrange(audio_emb, "b t c -> (b t) 1 c", t=num_frames)
                attn_audio_emb = audio_emb
                residual_out = self.music_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
                residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
                hidden_states = hidden_states + residual_out
        return hidden_states

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None, enable_wantodance_global=False):
        if enable_wantodance_global:
            x = self.patch_embedding_global(x)
        else:
            x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def add_condition_video_adapter(self, in_channels=48, tokens_per_frame=8, num_heads=8, max_latent_frames=128):
        """Post-hoc opt-in (see ConditionVideoAdapter/ConditionVideoCrossAttention's docstrings for
        the full rationale): attaches a fresh ConditionVideoAdapter pooler to this model plus a new,
        UNGATED cross-attention branch to every DiTBlock (see ConditionVideoCrossAttention's own
        docstring for why the gate was removed). Called by the training wrapper (see ego-moma's
        src/models/vla/video_gen.py) after loading pretrained weights via from_pretrained -- not
        part of any pretrained checkpoint's state dict, so not resolved through
        configs/model_configs.py's hash-based registry.

        `in_channels=48` matches the Wan2.2 VAE's latent channel count (WanVideoVAE38) used by both
        wan2.2_ti2v_5b and wan2.2_fun_5b_control -- the only backbones this has been used with so
        far; override if pairing with a different VAE.

        `max_latent_frames=128` sizes ConditionVideoAdapter's learned per-latent-frame positional
        embedding table (see its docstring) -- must be >= the condition video's actual VAE-latent
        frame count at call time (`1 + (condition_num_frames - 1) // 4`), with headroom for any
        config that raises `condition_num_frames`; 128 latent frames covers up to a 509-raw-frame
        condition video, comfortably above the 49/61 used by this repo's existing configs.
        """
        # Match the already-loaded model's device/dtype (typically bfloat16, set at from_pretrained
        # load time, not by a later blanket `.to()` -- a freshly constructed nn.Module defaults to
        # float32/CPU regardless of what device/dtype the rest of the model was cast to, so a caller
        # that adds this adapter after from_pretrained but before its own `.to(device)` would
        # otherwise hit a dtype mismatch the first time this module actually runs).
        ref_param = self.patch_embedding.weight
        self.condition_video_adapter = ConditionVideoAdapter(
            in_channels=in_channels, dim=self.dim, tokens_per_frame=tokens_per_frame, num_heads=num_heads,
            max_latent_frames=max_latent_frames,
        ).to(device=ref_param.device, dtype=ref_param.dtype)
        for block in self.blocks:
            block.cond_cross_attn = ConditionVideoCrossAttention(self.dim, num_heads=num_heads).to(
                device=ref_param.device, dtype=ref_param.dtype
            )
        self.has_condition_video_input = True

    def add_episode_control_video_resampler(self, in_channels=48, attn_dim=256, num_heads=8,
                                              max_target_frames=128, max_condition_frames=256, gate_init=0.0):
        """Post-hoc opt-in (see EpisodeControlVideoResampler's own docstring for the full
        rationale): attaches a resampler that lets `WanVideoUnit_FunControl`'s `control_video`
        channel-concat pathway accept a condition (episode) video whose frame count DIFFERS from
        the target clip's own -- resampled down to match via a reference-frame-informed
        cross-attention blend, rather than requiring `DATA.condition_num_frames ==
        DATA.clip_length` (ego-moma's `control_video_from_episode` docstring covers the
        equal-length case, which skips this resampler entirely).

        `in_channels=48` matches the Wan2.2 VAE's latent channel count, same as
        `add_condition_video_adapter`. `attn_dim` is this module's OWN internal attention
        dimension for computing blend weights (deliberately much smaller than `self.dim`, since it
        only needs to represent "which condition frame is relevant," not reconstruct content --
        the actual blended output stays in the original `in_channels`-channel VAE-latent space).
        `max_target_frames`/`max_condition_frames` size, respectively, the learned per-target-slot
        embedding table and a soft cap on how long a condition video this resampler tolerates
        (raises if exceeded at call time) -- pass values with headroom over any config's actual
        `clip_length`/`condition_num_frames`, same convention as `add_condition_video_adapter`'s
        `max_latent_frames`.

        Deliberately NOT cast to `ref_param.dtype` (bf16) -- see EpisodeControlVideoResampler's own
        docstring: this module's parameters stay float32 so its optimizer updates don't silently
        round to zero at bf16 precision; only the device is matched here.
        """
        ref_param = self.patch_embedding.weight
        self.episode_control_video_resampler = EpisodeControlVideoResampler(
            in_channels=in_channels, attn_dim=attn_dim, num_heads=num_heads,
            max_target_frames=max_target_frames, max_condition_frames=max_condition_frames,
            gate_init=gate_init,
        ).to(device=ref_param.device)

    def enable_condition_attention_capture(self):
        """Turn on attention-weight recording for both the condition-video pooling step
        (`ConditionVideoAdapter`) and every DiT block's injection cross-attention
        (`ConditionVideoCrossAttention`) -- see `get_condition_attention_capture` for retrieval.
        Only meaningful after `add_condition_video_adapter`. Adds real overhead (an unfused,
        weight-materializing attention computation per block per denoising step instead of a fused
        kernel) -- intended for evaluation/visualization runs, not training."""
        if self.condition_video_adapter is None:
            raise RuntimeError("enable_condition_attention_capture called before add_condition_video_adapter.")
        self.condition_video_adapter.capture_attention = True
        for block in self.blocks:
            block.cond_cross_attn.capture_attention = True

    def disable_condition_attention_capture(self):
        if self.condition_video_adapter is not None:
            self.condition_video_adapter.capture_attention = False
        for block in self.blocks:
            if block.cond_cross_attn is not None:
                block.cond_cross_attn.capture_attention = False

    def reset_condition_attention_capture(self):
        """Clear any weights captured by a previous generation call -- call before a fresh one, so
        `get_condition_attention_capture` doesn't mix history across separate `pipe(...)` calls."""
        if self.condition_video_adapter is not None:
            self.condition_video_adapter.last_pooling_attn = None
            self.condition_video_adapter.last_patch_grid = None
        for block in self.blocks:
            if block.cond_cross_attn is not None:
                block.cond_cross_attn.reset_capture()

    def get_condition_attention_capture(self):
        """Returns `{"pooling_attn": [B, num_tokens, num_patches] or None, "patch_grid": (F, H, W)
        or None, "block_attn_history": [block_0_history, ...]}`, where each block's history is a
        list of `[B, num_tokens]` tensors, one per denoising step captured since the last
        `reset_condition_attention_capture()` call. Composing these into a per-condition-video-frame
        attention distribution is downstream-repo-specific (needs to know how raw condition-video
        frames map to VAE-latent frames) -- see ego-moma's
        `src/utils/condition_video_attention.py`."""
        if self.condition_video_adapter is None:
            return {"pooling_attn": None, "patch_grid": None, "block_attn_history": []}
        return {
            "pooling_attn": self.condition_video_adapter.last_pooling_attn,
            "patch_grid": self.condition_video_adapter.last_patch_grid,
            "block_attn_history": [block.cond_cross_attn.captured_attn_history for block in self.blocks],
        }

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                condition_video_tokens: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs, condition_video_tokens,
                )
            else:
                x = block(x, context, t_mod, freqs, condition_video_tokens)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
