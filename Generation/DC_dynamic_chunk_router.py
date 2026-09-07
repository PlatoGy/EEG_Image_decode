import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusion_prior import DiffusionPriorUNet
from DC_train_atms import ATMS


DYNAMIC_INIT_CHUNKS = ((0, 63), (63, 125), (125, 188), (188, 250))


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


class DynamicBoundaryPredictor(nn.Module):
    def __init__(
        self,
        sequence_length=250,
        num_channels=63,
        hidden_dim=128,
        min_chunk_length=20.0,
        init_chunks=DYNAMIC_INIT_CHUNKS,
        delta_scale=0.1,
    ):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_chunks = len(init_chunks)
        self.min_chunk_length = float(min_chunk_length)
        self.delta_scale = float(delta_scale)

        min_total = self.num_chunks * self.min_chunk_length
        if min_total >= self.sequence_length:
            raise ValueError(
                f"min_chunk_length is too large: {self.min_chunk_length} * {self.num_chunks} >= {self.sequence_length}"
            )

        init_lengths = torch.tensor([end - start for start, end in init_chunks], dtype=torch.float32)
        if int(init_lengths.sum().item()) != self.sequence_length:
            raise ValueError(f"init chunks must cover {self.sequence_length} samples, got {init_lengths.sum().item()}")
        if torch.any(init_lengths <= self.min_chunk_length):
            raise ValueError("Each init chunk must be longer than min_chunk_length.")

        remaining = self.sequence_length - min_total
        init_probs = (init_lengths - self.min_chunk_length) / remaining
        self.register_buffer("base_logits", torch.log(init_probs))

        self.net = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=5, stride=5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_chunks),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, eeg):
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG [B,63,250], got {tuple(eeg.shape)}")
        if eeg.shape[-1] != self.sequence_length:
            raise ValueError(f"Expected EEG time length {self.sequence_length}, got {eeg.shape[-1]}")

        delta_logits = self.net(eeg) * self.delta_scale
        logits = self.base_logits.to(eeg.device, eeg.dtype).unsqueeze(0) + delta_logits
        remaining = self.sequence_length - self.num_chunks * self.min_chunk_length
        lengths = self.min_chunk_length + remaining * torch.softmax(logits, dim=-1)
        boundaries = torch.cumsum(lengths, dim=-1)[:, :-1]
        return boundaries, lengths


class DynamicSoftMasker(nn.Module):
    def __init__(self, sequence_length=250, temperature=2.0):
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.temperature = float(temperature)
        positions = torch.arange(self.sequence_length, dtype=torch.float32)
        self.register_buffer("positions", positions)

    def forward(self, eeg, boundaries):
        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG [B,63,250], got {tuple(eeg.shape)}")
        batch_size = eeg.shape[0]
        positions = self.positions.to(eeg.device, eeg.dtype).view(1, 1, self.sequence_length)
        boundaries = boundaries.to(eeg.device, eeg.dtype)
        tau = max(self.temperature, 1e-6)

        b1 = boundaries[:, 0].view(batch_size, 1, 1)
        b2 = boundaries[:, 1].view(batch_size, 1, 1)
        b3 = boundaries[:, 2].view(batch_size, 1, 1)

        m1 = torch.sigmoid((b1 - positions) / tau)
        m2 = torch.sigmoid((positions - b1) / tau) * torch.sigmoid((b2 - positions) / tau)
        m3 = torch.sigmoid((positions - b2) / tau) * torch.sigmoid((b3 - positions) / tau)
        m4 = torch.sigmoid((positions - b3) / tau)
        masks = torch.cat([m1, m2, m3, m4], dim=1)
        masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        chunked = eeg[:, None, :, :] * masks[:, :, None, :]
        return chunked, masks


class DynamicChunkATMS(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        num_subjects=2,
        device="cuda",
        min_chunk_length=20.0,
        mask_temperature=2.0,
        boundary_hidden_dim=128,
    ):
        super().__init__()
        self.global_atms = load_frozen_atms(checkpoint_path, num_subjects, device)
        self.chunk_atms = load_frozen_atms(checkpoint_path, num_subjects, device)
        self.boundary_predictor = DynamicBoundaryPredictor(
            min_chunk_length=min_chunk_length,
            hidden_dim=boundary_hidden_dim,
        ).to(device)
        self.masker = DynamicSoftMasker(temperature=mask_temperature).to(device)
        self.device = torch.device(device)

    def forward(self, eeg, subject_ids):
        eeg = eeg.to(self.device)
        subject_ids = subject_ids.to(self.device)
        z_global = self.global_atms(eeg, subject_ids)

        boundaries, lengths = self.boundary_predictor(eeg)
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

    def forward(self, timesteps):
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        time_features = self.time_proj(timesteps)
        time_features = self.time_embedding(time_features)
        logits = self.mlp(time_features)
        return torch.softmax(logits, dim=-1)


class DynamicChunkRouterCondition(nn.Module):
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


class DynamicChunkRouterPipe:
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

    def train_epoch(self, dataloader, conditioner, optimizer, lr_scheduler):
        self.diffusion_prior.train()
        self.router_condition.train()
        conditioner.train()
        conditioner.global_atms.eval()
        conditioner.chunk_atms.eval()
        criterion = nn.MSELoss(reduction="none")
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        loss_sum = 0.0
        boundary_values = []
        length_values = []
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

            loss = criterion(noise_pred, noise).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.parameters()) + list(conditioner.boundary_predictor.parameters()), 1.0)
            lr_scheduler.step()
            optimizer.step()

            loss_sum += loss.item()
            boundary_values.append(condition_cache["boundaries"].detach().float().cpu())
            length_values.append(condition_cache["lengths"].detach().float().cpu())

        boundaries = torch.cat(boundary_values, dim=0)
        lengths = torch.cat(length_values, dim=0)
        if router_weight_values:
            router_weights = torch.cat(router_weight_values, dim=0)
            router_weight_mean = router_weights.mean(dim=0)
        else:
            router_weight_mean = torch.full((4,), float("nan"))

        return {
            "loss": loss_sum / len(dataloader),
            "boundary_mean": boundaries.mean(dim=0),
            "boundary_std": boundaries.std(dim=0, unbiased=False),
            "length_mean": lengths.mean(dim=0),
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


def make_dynamic_chunk_router_modules(device, cond_dim=1024, dropout=0.1, init_gamma=0.1):
    diffusion_prior = DiffusionPriorUNet(cond_dim=cond_dim, dropout=dropout)
    router_condition = DynamicChunkRouterCondition(init_gamma=init_gamma, num_chunks=4)
    return DynamicChunkRouterPipe(diffusion_prior, router_condition, device=device)


def make_dynamic_optimizer_and_scheduler(pipe, conditioner, dataloader, epochs, lr):
    params = list(pipe.parameters()) + list(conditioner.boundary_predictor.parameters())
    optimizer = optim.Adam(params, lr=lr)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * epochs),
    )
    return optimizer, lr_scheduler
