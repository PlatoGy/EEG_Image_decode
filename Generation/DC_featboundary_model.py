import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusion_prior import DiffusionPriorUNet
from DC_train_atms import ATMS


FEATBOUNDARY_CHUNKS = ((0, 63), (63, 125), (125, 188), (188, 250))
FEATBOUNDARY_BOUNDARIES = tuple(chunk[1] for chunk in FEATBOUNDARY_CHUNKS[:-1])


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def load_frozen_atms(checkpoint_path, num_subjects, device):
    model = ATMS(63, 250, num_subjects=num_subjects)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    return freeze_module(model)


def moving_average_1d(scores, kernel_size=5):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return scores
    if kernel_size % 2 == 0:
        raise ValueError("smoothing_kernel must be odd, for example 3 or 5.")
    pad = kernel_size // 2
    x = F.pad(scores[:, None, :], (pad, pad), mode="replicate")
    weight = torch.ones(1, 1, kernel_size, device=scores.device, dtype=scores.dtype) / kernel_size
    return F.conv1d(x, weight).squeeze(1)


def temporal_change_score(temporal_feature, smoothing_kernel=5):
    if temporal_feature.ndim != 3:
        raise ValueError(f"Expected temporal feature [B,T,D], got {tuple(temporal_feature.shape)}")
    prev_feature = temporal_feature[:, :-1, :]
    next_feature = temporal_feature[:, 1:, :]
    cosine = F.cosine_similarity(next_feature, prev_feature, dim=-1)
    scores = 0.5 * (1.0 - cosine)
    return moving_average_1d(scores, smoothing_kernel)


def interpolate_scores_to_raw_time(scores, sequence_length=250):
    if scores.ndim != 2:
        raise ValueError(f"Expected change scores [B,T-1], got {tuple(scores.shape)}")
    return F.interpolate(
        scores[:, None, :],
        size=int(sequence_length),
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def constrained_top3_boundaries(raw_scores, min_len=20, sequence_length=250):
    if raw_scores.ndim != 2:
        raise ValueError(f"Expected raw scores [B,250], got {tuple(raw_scores.shape)}")
    sequence_length = int(sequence_length)
    min_len = int(min_len)
    if raw_scores.shape[1] != sequence_length:
        raise ValueError(f"Expected raw score length {sequence_length}, got {raw_scores.shape[1]}")
    if min_len * 4 >= sequence_length:
        raise ValueError(f"min_len is too large: {min_len} * 4 >= {sequence_length}")

    batch_size = raw_scores.shape[0]
    device = raw_scores.device
    dtype = raw_scores.dtype
    positions = torch.arange(sequence_length, device=device)
    neg = torch.finfo(dtype).min

    first_valid = (positions >= min_len) & (positions <= sequence_length - 3 * min_len)
    second_valid = (positions >= 2 * min_len) & (positions <= sequence_length - 2 * min_len)
    third_valid = (positions >= 3 * min_len) & (positions <= sequence_length - min_len)

    candidate_scores = raw_scores.clone()
    candidate_scores[:, 0] = neg
    candidate_scores[:, -1] = neg

    dp1 = candidate_scores.masked_fill(~first_valid.view(1, -1), neg)
    best1_values, best1_indices = torch.cummax(dp1, dim=1)

    prev_pos = (positions - min_len).clamp(0, sequence_length - 1)
    dp2_prev_values = best1_values[:, prev_pos]
    dp2 = (candidate_scores + dp2_prev_values).masked_fill(~second_valid.view(1, -1), neg)
    best2_values, best2_indices = torch.cummax(dp2, dim=1)

    dp3_prev_values = best2_values[:, prev_pos]
    dp3 = (candidate_scores + dp3_prev_values).masked_fill(~third_valid.view(1, -1), neg)
    _, b3 = torch.max(dp3, dim=1)

    b3_lookup = (b3 - min_len).clamp(0, sequence_length - 1)
    b2 = best2_indices.gather(1, b3_lookup.view(batch_size, 1)).squeeze(1)

    b2_lookup = (b2 - min_len).clamp(0, sequence_length - 1)
    b1 = best1_indices.gather(1, b2_lookup.view(batch_size, 1)).squeeze(1)

    boundaries = torch.stack([b1, b2, b3], dim=1).long()
    lengths = torch.stack(
        [
            boundaries[:, 0],
            boundaries[:, 1] - boundaries[:, 0],
            boundaries[:, 2] - boundaries[:, 1],
            torch.full_like(boundaries[:, 2], sequence_length) - boundaries[:, 2],
        ],
        dim=1,
    )
    selected_scores = raw_scores.gather(1, boundaries)
    return boundaries, lengths, selected_scores


class HardBoundaryMasker(nn.Module):
    def __init__(self, sequence_length=250):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.register_buffer("positions", torch.arange(self.sequence_length))

    def forward(self, eeg, boundaries):
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG [B,63,250], got {tuple(eeg.shape)}")
        boundaries = boundaries.to(eeg.device).long()
        positions = self.positions.to(eeg.device).view(1, self.sequence_length)
        b1 = boundaries[:, 0].view(-1, 1)
        b2 = boundaries[:, 1].view(-1, 1)
        b3 = boundaries[:, 2].view(-1, 1)

        masks = torch.stack(
            [
                positions < b1,
                (positions >= b1) & (positions < b2),
                (positions >= b2) & (positions < b3),
                positions >= b3,
            ],
            dim=1,
        ).to(eeg.dtype)
        chunked = eeg[:, None, :, :] * masks[:, :, None, :]
        return chunked, masks


class FeatureBoundaryChunkATMS(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        num_subjects=2,
        device="cuda",
        min_chunk_length=20,
        smoothing_kernel=5,
    ):
        super().__init__()
        self.global_atms = load_frozen_atms(checkpoint_path, num_subjects, device)
        self.chunk_atms = load_frozen_atms(checkpoint_path, num_subjects, device)
        self.masker = HardBoundaryMasker().to(device)
        self.min_chunk_length = int(min_chunk_length)
        self.smoothing_kernel = int(smoothing_kernel)
        self.device = torch.device(device)

    def extract_global_and_temporal_feature(self, eeg, subject_ids):
        encoded = self.global_atms.encoder(eeg, None, subject_ids)
        temporal_feature = self.global_atms.enc_eeg[0](encoded)
        eeg_embedding = self.global_atms.enc_eeg[1](temporal_feature)
        z_global = self.global_atms.proj_eeg(eeg_embedding)
        return z_global, temporal_feature

    @torch.no_grad()
    def forward(self, eeg, subject_ids):
        eeg = eeg.to(self.device)
        subject_ids = subject_ids.to(self.device)

        z_global, temporal_feature = self.extract_global_and_temporal_feature(eeg, subject_ids)
        feature_scores = temporal_change_score(temporal_feature, self.smoothing_kernel)
        raw_scores = interpolate_scores_to_raw_time(feature_scores, sequence_length=eeg.shape[-1])
        boundaries, lengths, selected_scores = constrained_top3_boundaries(
            raw_scores,
            min_len=self.min_chunk_length,
            sequence_length=eeg.shape[-1],
        )

        chunked, masks = self.masker(eeg, boundaries)
        batch_size, num_chunks, channels, time = chunked.shape
        flat_chunked = chunked.reshape(batch_size * num_chunks, channels, time)
        flat_subject_ids = subject_ids.repeat_interleave(num_chunks)
        z_chunks = self.chunk_atms(flat_chunked, flat_subject_ids)
        z_chunks = z_chunks.reshape(batch_size, num_chunks, -1)

        return {
            "z_global": z_global.float(),
            "z_chunks": z_chunks.float(),
            "boundaries": boundaries.float(),
            "lengths": lengths.float(),
            "score_peaks": selected_scores.float(),
            "feature_change_score": feature_scores.float(),
            "raw_change_score": raw_scores.float(),
            "masks": masks.float(),
        }


class DiffusionTimeRouter(nn.Module):
    def __init__(self, time_embed_dim=512, hidden_dim=128, num_chunks=4):
        super().__init__()
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(time_embed_dim, time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_chunks),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, timesteps):
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        time_features = self.time_proj(timesteps)
        time_features = self.time_embedding(time_features)
        logits = self.mlp(time_features)
        return torch.softmax(logits, dim=-1)


class FeatureBoundaryRouterCondition(nn.Module):
    def __init__(self, init_gamma=0.1, num_chunks=4):
        super().__init__()
        self.router = DiffusionTimeRouter(num_chunks=num_chunks)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, timesteps, z_global, z_chunks):
        batch_size = z_global.shape[0]
        if timesteps.ndim == 0 or timesteps.numel() == 1:
            timesteps = torch.full((batch_size,), float(timesteps.reshape(-1)[0].item()), device=z_global.device)
        timesteps = timesteps.to(z_global.device)
        weights = self.router(timesteps).to(z_chunks.dtype)
        if weights.shape[1] != z_chunks.shape[1]:
            raise ValueError(f"Router/chunk mismatch: weights {tuple(weights.shape)}, z_chunks {tuple(z_chunks.shape)}")
        z_chunk = torch.sum(z_chunks * weights[:, :, None], dim=1)
        c_t = z_global + self.gamma.to(z_global.dtype) * z_chunk
        return c_t, weights


class FeatureBoundaryRouterPipe:
    def __init__(self, diffusion_prior=None, router_condition=None, scheduler=None, device="cuda"):
        self.diffusion_prior = diffusion_prior.to(device)
        self.router_condition = router_condition.to(device)
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler()
        else:
            self.scheduler = scheduler
        self.device = torch.device(device)

    def parameters(self):
        yield from self.diffusion_prior.parameters()
        yield from self.router_condition.parameters()

    def condition_at_t(self, condition_cache, timesteps):
        return self.router_condition(
            timesteps,
            condition_cache["z_global"].to(self.device),
            condition_cache["z_chunks"].to(self.device),
        )

    def train_epoch(self, dataloader, conditioner, optimizer, lr_scheduler, router_entropy_reg_weight=0.0, freeze_router=False):
        self.diffusion_prior.train()
        self.router_condition.train()
        conditioner.eval()
        for param in self.router_condition.router.parameters():
            param.requires_grad_(not freeze_router)
        criterion = nn.MSELoss(reduction="none")
        num_train_timesteps = self.scheduler.config.num_train_timesteps
        max_entropy = torch.log(torch.tensor(4.0, device=self.device))

        loss_sum = 0.0
        diffusion_loss_sum = 0.0
        router_entropy_penalty_sum = 0.0
        router_entropy_penalty_count = 0
        boundary_values = []
        length_values = []
        score_peak_values = []
        router_weight_values = []

        for batch in dataloader:
            eeg = batch["eeg"].to(self.device)
            subject_ids = batch["subject_id"].to(self.device)
            h_embeds = batch["h_embedding"].to(self.device)
            batch_size = h_embeds.shape[0]

            condition_cache = conditioner(eeg, subject_ids)
            use_condition = (torch.rand(1, device=self.device) >= 0.1).item()

            noise = torch.randn_like(h_embeds)
            timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=self.device)
            perturbed_h_embeds = self.scheduler.add_noise(h_embeds, noise, timesteps)

            if use_condition:
                c_t, weights = self.condition_at_t(condition_cache, timesteps)
                noise_pred = self.diffusion_prior(perturbed_h_embeds, timesteps, c_t)
                router_weight_values.append(weights.detach().float().cpu())
            else:
                noise_pred = self.diffusion_prior(perturbed_h_embeds, timesteps, None)

            diffusion_loss = criterion(noise_pred, noise).mean()
            loss = diffusion_loss
            if use_condition:
                entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
                router_entropy_penalty = 1.0 - entropy / max_entropy
                loss = loss + float(router_entropy_reg_weight) * router_entropy_penalty
                router_entropy_penalty_sum += router_entropy_penalty.detach().float().item()
                router_entropy_penalty_count += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.parameters()), 1.0)
            lr_scheduler.step()
            optimizer.step()

            loss_sum += loss.item()
            diffusion_loss_sum += diffusion_loss.detach().float().item()
            boundary_values.append(condition_cache["boundaries"].detach().float().cpu())
            length_values.append(condition_cache["lengths"].detach().float().cpu())
            score_peak_values.append(condition_cache["score_peaks"].detach().float().cpu())

        boundaries = torch.cat(boundary_values, dim=0)
        lengths = torch.cat(length_values, dim=0)
        score_peaks = torch.cat(score_peak_values, dim=0)
        if router_weight_values:
            router_weights = torch.cat(router_weight_values, dim=0)
            router_weight_mean = router_weights.mean(dim=0)
        else:
            router_weight_mean = torch.full((4,), float("nan"))

        return {
            "loss": loss_sum / len(dataloader),
            "diffusion_loss": diffusion_loss_sum / len(dataloader),
            "router_entropy_penalty": router_entropy_penalty_sum / max(router_entropy_penalty_count, 1),
            "boundary_mean": boundaries.mean(dim=0),
            "boundary_std": boundaries.std(dim=0, unbiased=False),
            "length_mean": lengths.mean(dim=0),
            "length_std": lengths.std(dim=0, unbiased=False),
            "score_peak_mean": score_peaks.mean(dim=0),
            "router_weight_mean": router_weight_mean,
            "gamma": self.router_condition.gamma.detach().float().cpu(),
        }

    @torch.no_grad()
    def generate(self, condition_cache, num_inference_steps=50, timesteps=None, guidance_scale=5.0, generator=None):
        self.diffusion_prior.eval()
        self.router_condition.eval()
        batch_size = condition_cache["z_global"].shape[0]

        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        h_t = torch.randn(batch_size, self.diffusion_prior.embed_dim, generator=generator, device=self.device)
        for _, timestep in tqdm(enumerate(timesteps), total=len(timesteps)):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * timestep
            if guidance_scale == 0:
                noise_pred = self.diffusion_prior(h_t, t, None)
            else:
                c_t, _ = self.condition_at_t(condition_cache, t)
                noise_pred_cond = self.diffusion_prior(h_t, t, c_t)
                noise_pred_uncond = self.diffusion_prior(h_t, t, None)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            h_t = self.scheduler.step(noise_pred, int(timestep.item()), h_t, generator=generator).prev_sample
        return h_t


def make_featboundary_router_modules(device, cond_dim=1024, dropout=0.1, init_gamma=0.1):
    diffusion_prior = DiffusionPriorUNet(cond_dim=cond_dim, dropout=dropout)
    router_condition = FeatureBoundaryRouterCondition(init_gamma=init_gamma, num_chunks=4)
    return FeatureBoundaryRouterPipe(diffusion_prior, router_condition, device=device)


def make_optimizer_and_scheduler(pipe, dataloader, epochs, lr):
    optimizer = optim.Adam(list(pipe.parameters()), lr=lr)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * epochs),
    )
    return optimizer, lr_scheduler
