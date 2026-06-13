from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import io

# --- Physical constants ---
MAX_ACCEL       = 50.0   # m/s² forward
MAX_BRAKE       = 80.0   # m/s² braking
MAX_STEER_RATE  = 0.8    # rad/s per (m/s) of speed
MAX_SPEED       = 90.0   # m/s (~324 km/h, F1 top speed at Monza)
TRACK_WIDTH     = 15.0   # m (Monza is ~15m wide)
DT              = 0.05   # s (20 Hz)

class TrackEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        track_data,
        max_steps,
        render_mode = None,
    ) -> None:
        super().__init__()

        self.centerline      = track_data['centerline']       # (N, 2) meters
        self.tangents        = track_data['tangents']         # (N, 2) unit vectors
        self.circuit_length  = track_data['circuit_length_m'] # meters
        self.n_points        = len(self.centerline)
        self.max_steps       = max_steps

        # Observation space: [s, d, v_long, v_lat]
        low  = np.array([0.0, -TRACK_WIDTH/2, -MAX_SPEED, -MAX_SPEED], dtype=np.float32)
        high = np.array([1.0,  TRACK_WIDTH/2,  MAX_SPEED,  MAX_SPEED], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: [accel, steer]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # ...

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode={render_mode!r}. "
                f"Supported: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        self._agent_pos = None
        self._heading = None
        self._speed = None
        self._closest_idx = None
        self._step_count = None


    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _get_obs(self):

        s = self._closest_idx / self.n_points

        # 1. compute norm tangent as nm
        # 2. compute dot prod betw agent_pos and nm to get left/right side of track as sd
        d = 0

        return np.array([s, d, v_long, v_lat], dtype=np.float32)

    def _get_info(self):
        return {
            "pos": self._agent_pos,
            "heading": self._heading,
            "speed": self._speed,
            "closest_idx": self._closest_idx,
            "step_count": self._step_count,
        }

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self._agent_pos = self.centerline[0]
        self._heading = self.tangents[0]
        self._speed = 0
        self._closest_idx = 0
        self._step_count = 0
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}.")

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    def render(self):
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#12122a")
        ax.set_facecolor("#12122a")
        ax.set_aspect("equal")
        ax.axis("off")

        cl = self.centerline
        normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)

        left  = cl + normals * (TRACK_WIDTH / 2)
        right = cl - normals * (TRACK_WIDTH / 2)

        left_c  = np.vstack([left,  left[0]])
        right_c = np.vstack([right, right[0]])
        cl_c    = np.vstack([cl,    cl[0]])

        # --- Track: white fill between boundaries, thin grey edges ---
        track_x = np.concatenate([left_c[:, 0], right_c[::-1, 0]])
        track_y = np.concatenate([left_c[:, 1], right_c[::-1, 1]])
        ax.fill(track_x, track_y, color="white", zorder=1)

        # Boundary edges (thin dark lines so the border is crisp)
        ax.plot(left_c[:, 0],  left_c[:, 1],  color="#555", linewidth=0.8, zorder=2)
        ax.plot(right_c[:, 0], right_c[:, 1], color="#555", linewidth=0.8, zorder=2)

        # --- Start/finish line: bold red line extending beyond track borders ---
        sf_mid = (left[0] + right[0]) / 2
        perp   = left[0] - right[0]   # vector across full track width
        unit_perp = perp / np.linalg.norm(perp)
        overshoot = TRACK_WIDTH * 0.8   # how far beyond each edge

        start = right[0] - unit_perp * overshoot
        end   = left[0]  + unit_perp * overshoot

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#e8002d", linewidth=3.0, zorder=4, solid_capstyle="butt"
        )

        # --- Car arrow: sized relative to track width ---
        if self._agent_pos is not None:
            car_x, car_y = self._agent_pos[:2]

            diffs = self.centerline - np.array([car_x, car_y])
            idx   = int(np.argmin((diffs ** 2).sum(axis=1)))
            heading = self.tangents[idx]

            arrow_len = TRACK_WIDTH * 0.5   # half track width — fits inside the track
            dx = heading[0] * arrow_len
            dy = heading[1] * arrow_len

            ax.annotate(
                "",
                xy    =(car_x + dx,        car_y + dy),
                xytext=(car_x - dx * 0.5,  car_y - dy * 0.5),
                arrowprops=dict(
                    arrowstyle="simple,head_width=0.8,head_length=0.6",
                    color="#e8002d",
                    lw=0,
                ),
                zorder=5,
            )

        ax.set_title(f"Step {getattr(self, '_step', 0)}",
                    color="white", fontsize=11, pad=8)

        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        import PIL.Image
        img = np.array(PIL.Image.open(buf).convert("RGB"))
        return img

    def close(self):
        pass