import torch
from typing import Dict, Tuple, Set, Optional, Any

StepLayerBranchKey = Tuple[int, int, int]


class ActivationRecorder:
    def __init__(self):
        self.cache: Dict[StepLayerBranchKey, torch.Tensor] = {}
        self.current_step: int = -1
        self.enabled: bool = False

    def clear(self):
        self.cache = {}

    def set_step(self, step_idx: int):
        self.current_step = step_idx

    def record(self, layer_idx: int, x: torch.Tensor):
        """
        x: [B, T, C]
        这里把 batch 里的每一路都分别记录下来
        key = (step, layer, branch)
        """
        if not self.enabled:
            return

        if x.dim() != 3:
            return

        num_branches = x.size(0)
        for branch_idx in range(num_branches):
            self.cache[(self.current_step, layer_idx, branch_idx)] = (
                x[branch_idx:branch_idx + 1].detach().float().cpu()
            )


class ActivationSteerer:
    def __init__(self):
        self.current_step: int = -1
        self.enabled: bool = False

        # 默认主分支强一点，另外两路弱一点/不改
        self.alpha_branch: Dict[int, float] = {
            0: 0.4,   # cond_txt_spk
            1: 0.0,   # cond_txt
            2: 0.0,   # uncond
        }

        self.steering_vectors: Dict[StepLayerBranchKey, torch.Tensor] = {}
        self.selected_points: Set[StepLayerBranchKey] = set()

    def set_step(self, step_idx: int):
        self.current_step = step_idx

    def load_profile(
        self,
        profile: Dict[str, Any],
        alpha_branch: Optional[Dict[int, float]] = None,
    ):
        self.steering_vectors = profile["steering_vectors"]
        self.selected_points = set(profile["selected_points"])
        if alpha_branch is not None:
            self.alpha_branch.update(alpha_branch)

    def get_alpha(self, branch_idx: int) -> float:
        return self.alpha_branch.get(branch_idx, 0.0)

    def apply(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        对每一路分别判断是否要做 steering
        x: [B, T, C]
        """
        if not self.enabled:
            return x

        if x.dim() != 3:
            return x

        outputs = []
        num_branches = x.size(0)

        for branch_idx in range(num_branches):
            xb = x[branch_idx:branch_idx + 1]
            key = (self.current_step, layer_idx, branch_idx)

            if key not in self.selected_points or key not in self.steering_vectors:
                outputs.append(xb)
                continue

            alpha = self.get_alpha(branch_idx)
            if alpha == 0.0:
                outputs.append(xb)
                continue

            s = self.steering_vectors[key].to(device=xb.device, dtype=xb.dtype)  # [C]
            s = s / (torch.norm(s, p=2) + 1e-8)

            # xb: [1, T, C]
            proj = (xb * s.view(1, 1, -1)).sum(dim=-1, keepdim=True)  # [1, T, 1]
            xb = xb - alpha * proj * s.view(1, 1, -1)
            outputs.append(xb)

        return torch.cat(outputs, dim=0)


class TruSManager:
    def __init__(self):
        self.recorder = ActivationRecorder()
        self.steerer = ActivationSteerer()

    def clear(self):
        self.recorder.clear()

    def set_step(self, step_idx: int):
        self.recorder.set_step(step_idx)
        self.steerer.set_step(step_idx)


def make_trus_hook(trus_manager: TruSManager, layer_idx: int):
    def hook(module, inputs, output):
        trus_manager.recorder.record(layer_idx, output)
        output = trus_manager.steerer.apply(layer_idx, output)
        return output
    return hook