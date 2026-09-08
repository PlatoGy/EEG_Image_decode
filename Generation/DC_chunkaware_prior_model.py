import torch
from torch import nn

from DC_featboundary_model import (
    FEATBOUNDARY_BOUNDARIES,
    FEATBOUNDARY_CHUNKS,
    FeatureBoundaryRouterPipe,
    FeatureBoundaryRouterCondition,
    HardBoundaryMasker,
    constrained_top3_boundaries,
    interpolate_scores_to_raw_time,
    make_featboundary_router_modules,
    make_optimizer_and_scheduler,
    temporal_change_score,
)
from DC_train_atms import ATMS


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def load_atms_checkpoint(checkpoint_path, num_subjects, device):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model = ATMS(63, 250, num_subjects=num_subjects)
    model.load_state_dict(state)
    return model.to(device)


class ChunkAwareFeatureBoundaryATMS(nn.Module):
    def __init__(
        self,
        global_atms_ckpt,
        chunk_atms_ckpt,
        num_subjects=2,
        device="cuda",
        min_chunk_length=20,
        smoothing_kernel=5,
    ):
        super().__init__()
        self.global_atms = freeze_module(load_atms_checkpoint(global_atms_ckpt, num_subjects, device))
        self.chunk_atms = freeze_module(load_atms_checkpoint(chunk_atms_ckpt, num_subjects, device))
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
