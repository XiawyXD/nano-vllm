"""Device-resident standard speculative rejection correction, no engine policy.

Algorithm 1: https://arxiv.org/html/2211.17192v2 . Candidates MUST have been
sampled autoregressively from the supplied q. p[i] and q[i] must condition on
the same committed history plus candidates[:i]; p[K] is the bonus distribution.
No function can infer or verify that history/provenance from probabilities alone.

Only temperature>1e-10, no top-k/top-p/penalty. p/q are FP32 or FP64
probability tensors. Logits-to-probabilities preserves the ordinary FP32 Torch
temperature/softmax weights, then explicitly normalizes them in FP64 and retains
FP64 distributions. Correction arithmetic and categorical CDFs also use FP64. This is
finite-precision distribution correction, not seed/text equivalence with Torch
multinomial. This module makes no GPU speed or CUDA Graph support claim.
Optional leading dimensions exist for independent CPU distribution trials.
All tensor validity checks stay on-device. Invalid data returns error metadata;
the caller MUST use CorrectionResult.to_host() before publishing any tokens.
"""
from dataclasses import dataclass
import math
import secrets

import torch


INVALID_PROBABILITIES = 1
INVALID_CANDIDATE = 2
INVALID_UNIFORM = 4
ZERO_REJECTION_RESIDUAL = 8


@dataclass
class SamplingGenerators:
    """Three persistent RNG domains; never reseed these per speculative round."""
    draft: torch.Generator
    acceptance: torch.Generator
    target: torch.Generator

    def validate(self, device):
        streams = (self.draft, self.acceptance, self.target)
        if any(not isinstance(g, torch.Generator) for g in streams):
            raise TypeError('explicit torch.Generator objects are required')
        if len({id(g) for g in streams}) != 3 or len({g.initial_seed() for g in streams}) != 3:
            raise ValueError('draft, acceptance and target require distinct RNG domains/seeds')
        if any(torch.device(g.device) != torch.device(device) for g in streams):
            raise ValueError('all generators must be on the probability tensor device')


def make_generators(seed=None, *, device='cpu'):
    """Create per-request streams once. None uses host entropy, not global Torch RNG."""
    if seed is None:
        seed = secrets.randbits(64)
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError('seed must fit the supported 64-bit Torch seed range')
    # SplitMix64 is a permutation; distinct counters yield distinct stream seeds.
    def mix(value):
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)
    return SamplingGenerators(*(torch.Generator(device=device).manual_seed(mix(seed + i)) for i in range(3)))


def _shape(probabilities):
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 1 or probabilities.shape[-1] < 1:
        raise ValueError('probabilities need a nonempty vocabulary axis')
    if probabilities.dtype not in (torch.float32, torch.float64):
        raise TypeError('probabilities must be FP32 or FP64, not rounded BF16/FP16')


def _normalize(probabilities):
    _shape(probabilities)
    weights = probabilities.to(torch.float64)
    total = weights.sum(-1)
    # Fixed representation tolerances, not a quality/distribution acceptance gate.
    tolerance = 1e-6 if probabilities.dtype == torch.float32 else 1e-12
    valid = (torch.isfinite(weights).all(-1) & (weights >= 0).all(-1)
             & torch.isfinite(total) & ((total - 1).abs() <= tolerance))
    clean = torch.where(torch.isfinite(weights) & (weights >= 0), weights, 0.)
    maximum = clean.amax(-1, keepdim=True)
    scaled = clean / torch.where(maximum > 0, maximum, 1.)
    mass = scaled.sum(-1, keepdim=True)
    return scaled / torch.where(mass > 0, mass, 1.), valid


def _uniform(shape, device, generator, supplied):
    if supplied is None:
        return torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if (not isinstance(supplied, torch.Tensor) or tuple(supplied.shape) != tuple(shape)
            or supplied.device != device or supplied.dtype not in (torch.float32, torch.float64)):
        raise ValueError('diagnostic uniforms must match shape/device and be FP32/FP64')
    return supplied.to(torch.float64)


def _categorical(weights, uniforms):
    """Sample nonnegative weights without multinomial's possible host validation.

    Right-sided search skips zero-mass intervals even when u==0. Clamp to the
    preceding representable value of total so rounding u*total cannot select a
    zero trailing entry. FP64 still has finite CDF/RNG resolution; no real-number
    exactness claim is made. No epsilon floor or p fallback is used.
    """
    cdf = weights.cumsum(-1)
    total = cdf[..., -1]
    threshold = torch.minimum(uniforms * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cdf.contiguous(), threshold.unsqueeze(-1).contiguous(), right=True).squeeze(-1)


def validate_sampling_options(temperature, *, top_k=0, top_p=1., repetition_penalty=1.):
    """Match the existing ordinary Sampler's <=1e-10 greedy boundary."""
    if (type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 1e-10
            or top_k != 0 or top_p != 1. or repetition_penalty != 1.):
        raise ValueError('requires finite temperature>1e-10, top_k=0, top_p=1, repetition_penalty=1')


@torch.inference_mode()
def temperature_probabilities(logits, temperature, *, top_k=0, top_p=1., repetition_penalty=1.):
    validate_sampling_options(temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    if not isinstance(logits, torch.Tensor) or logits.ndim < 1 or logits.shape[-1] < 1 or not logits.is_floating_point():
        raise ValueError('logits require a nonempty floating-point vocabulary axis')
    values = logits.float() / temperature
    valid = (~torch.isnan(values).any(-1) & ~torch.isposinf(values).any(-1) & torch.isfinite(values).any(-1))
    # Keep Sampler._reference's FP32 temperature/softmax weights. A large-vocab
    # FP32 softmax can have total mass outside the strict 1e-6 distribution
    # tolerance even when every weight is finite and nonnegative. Normalize at
    # the producer in FP64, retain FP64, and keep _normalize's stricter 1e-12
    # validation unchanged. No epsilon, probability floor, or p fallback.
    weights = torch.softmax(values, dim=-1).to(torch.float64)
    mass = weights.sum(-1, keepdim=True)
    valid = (valid & torch.isfinite(weights).all(-1) & (weights >= 0).all(-1)
             & torch.isfinite(mass.squeeze(-1)) & (mass.squeeze(-1) > 0))
    probs = weights / mass
    # Invalid logits/weights stay explicit invalid rows, including zero or
    # negative mass; normalization must never turn invalid weights into data.
    return torch.where(valid.unsqueeze(-1), probs, torch.full_like(probs, float('nan')))


@torch.inference_mode()
def sample_from_probs(probabilities, *, generator, uniforms=None):
    """Samples [...,V] -> [...] token IDs; -1 denotes invalid probabilities/RNG.

    q retained by the caller must be these exact input probability rows. Sampling
    and correction apply the identical FP64 normalization, never a later q recompute.
    ``uniforms`` is deterministic diagnostic injection only, never serving policy.
    """
    norm, valid = _normalize(probabilities)
    if not isinstance(generator, torch.Generator) or torch.device(generator.device) != probabilities.device:
        raise ValueError('explicit generator on the probability device is required')
    uniform = _uniform(probabilities.shape[:-1], probabilities.device, generator, uniforms)
    valid &= torch.isfinite(uniform) & (uniform >= 0) & (uniform < 1)
    token = _categorical(norm, torch.where(valid, uniform, 0.))
    return torch.where(valid, token, -1)


@dataclass
class DraftSample:
    ids: torch.Tensor
    probabilities: torch.Tensor


@torch.inference_mode()
def sample_draft(logits, temperature, *, generators, top_k=0, top_p=1., repetition_penalty=1.):
    generators.validate(logits.device)
    q = temperature_probabilities(logits, temperature, top_k=top_k, top_p=top_p,
                                  repetition_penalty=repetition_penalty)
    return DraftSample(sample_from_probs(q, generator=generators.draft), q)


@dataclass
class CorrectionResult:
    """packet[...,K+3] = [error bitmask, accepted count, K+1 padded token slots].

    Success publishes accepted+1 slots. Remaining slots are -1. Failure sets
    accepted=-1 and ALL token slots=-1, including otherwise accepted prefixes.
    EOS/stop/max_tokens truncation is exclusively the caller's commit policy.
    """
    packet: torch.Tensor

    def to_host(self):
        """Single-request only: exactly one small D2H; returns (tokens, accepted)."""
        if self.packet.ndim != 1:
            raise ValueError('to_host only supports one request')
        values = self.packet.cpu().tolist()
        error, accepted = values[:2]
        if error:
            raise ValueError(f'speculative correction rejected invalid data; error_bits={error}')
        if not 0 <= accepted < len(values) - 2:
            raise ValueError('invalid accepted count in correction packet')
        return values[2:3 + accepted], accepted


@torch.inference_mode()
def speculative_correction(candidates, q, p, *, generators,
                           acceptance_uniforms=None, sample_uniforms=None):
    """Standard correction for candidates[...,K], q[...,K,V], p[...,K+1,V].

    Return only a fixed small integer packet; p/q never leave their device.
    No .item(), nonzero(), tensor truth tests or dynamic device->host shapes.
    Every candidate must have positive q support, even beyond first rejection.
    K=0 samples directly from p[0]. Independent streams consume K acceptance
    uniforms and one target uniform per round, including all-accept rounds.
    """
    _shape(q)
    _shape(p)
    if q.ndim < 2 or p.ndim != q.ndim or p.shape != (*q.shape[:-2], q.shape[-2] + 1, q.shape[-1]):
        raise ValueError('expected q[...,K,V] and p[...,K+1,V]')
    if (not isinstance(candidates, torch.Tensor) or candidates.dtype != torch.int64
            or candidates.shape != q.shape[:-1] or candidates.device != q.device or p.device != q.device):
        raise ValueError('candidates must be int64 [...,K] on the p/q device')
    generators.validate(q.device)
    k, vocab = q.shape[-2:]
    pn, pvalid = _normalize(p)
    qn, qvalid = _normalize(q)
    prefix_shape = q.shape[:-2]
    u = _uniform((*prefix_shape, k), q.device, generators.acceptance, acceptance_uniforms)
    v = _uniform(prefix_shape, q.device, generators.target, sample_uniforms)
    uniform_valid = ((torch.isfinite(u) & (u >= 0) & (u < 1)).all(-1)
                     & torch.isfinite(v) & (v >= 0) & (v < 1))
    in_bounds = (candidates >= 0) & (candidates < vocab)
    indices = candidates.clamp(0, vocab - 1).unsqueeze(-1)
    qx = qn.gather(-1, indices).squeeze(-1)
    px = pn[..., :k, :].gather(-1, indices).squeeze(-1)
    ratio = (px / torch.where(qx > 0, qx, 1.)).clamp(max=1.)
    # Strict '<' ensures p(x)=0 cannot accept even the representable draw u=0.
    accepted_mask = (u < ratio) & in_bounds & (qx > 0)
    accepted = accepted_mask.to(torch.int64).cumprod(-1).sum(-1)
    select = accepted.unsqueeze(-1).unsqueeze(-1).expand(*prefix_shape, 1, vocab)
    selected_p = pn.gather(-2, select).squeeze(-2)
    # The bonus row's q is zero, so full acceptance selects p[K] unchanged.
    padded_q = torch.cat((qn, torch.zeros_like(pn[..., :1, :])), dim=-2)
    selected_q = padded_q.gather(-2, select).squeeze(-2)
    residual = (selected_p - selected_q).clamp_min(0.)
    maximum = residual.amax(-1, keepdim=True)
    # Scaling before summation preserves tiny positive residuals, including
    # subnormal mass. Zero mass is an error, never replaced by p or epsilon.
    scaled = residual / torch.where(maximum > 0, maximum, 1.)
    residual_valid = (maximum.squeeze(-1) > 0) & torch.isfinite(maximum.squeeze(-1))
    error = ((~(pvalid.all(-1) & qvalid.all(-1))).to(torch.int64) * INVALID_PROBABILITIES
             | (~(in_bounds & (qx > 0)).all(-1)).to(torch.int64) * INVALID_CANDIDATE
             | (~uniform_valid).to(torch.int64) * INVALID_UNIFORM
             | ((accepted < k) & ~residual_valid).to(torch.int64) * ZERO_REJECTION_RESIDUAL)
    sampled = _categorical(scaled, torch.where(uniform_valid, v, 0.))
    slots = torch.arange(k + 1, device=q.device)
    padded = torch.cat((candidates, torch.full_like(accepted.unsqueeze(-1), -1)), -1)
    tokens = torch.where(slots < accepted.unsqueeze(-1), padded, -1)
    tokens.scatter_(-1, accepted.unsqueeze(-1), sampled.unsqueeze(-1))
    tokens = torch.where((error == 0).unsqueeze(-1), tokens, -1)
    count = torch.where(error == 0, accepted, -1)
    return CorrectionResult(torch.cat((error.unsqueeze(-1), count.unsqueeze(-1), tokens), -1))
