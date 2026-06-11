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

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _get_obs(self):
        return None

    def _get_info(self):
        return None

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
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0..3.")

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.axis("off")

        # --- Draw track boundaries ---
        cl = self.centerline  # (N, 2)
        normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)  # (N, 2) left normals

        left  = cl + normals * (TRACK_WIDTH / 2)
        right = cl - normals * (TRACK_WIDTH / 2)

        # Close the loop for drawing
        left_closed  = np.vstack([left,  left[0]])
        right_closed = np.vstack([right, right[0]])
        cl_closed    = np.vstack([cl,    cl[0]])

        # Tarmac fill between boundaries
        track_x = np.concatenate([left_closed[:, 0], right_closed[::-1, 0]])
        track_y = np.concatenate([left_closed[:, 1], right_closed[::-1, 1]])
        ax.fill(track_x, track_y, color="#2d2d2d", zorder=1)

        # White dashed centerline
        ax.plot(cl_closed[:, 0], cl_closed[:, 1],
                color="white", linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)

        # Track edges (kerb-style red/white — just solid lines here)
        ax.plot(left_closed[:, 0],  left_closed[:, 1],  color="#e63946", linewidth=1.5, zorder=3)
        ax.plot(right_closed[:, 0], right_closed[:, 1], color="#e63946", linewidth=1.5, zorder=3)

        # --- Start/finish line ---
        start_l = left[0]
        start_r = right[0]
        ax.plot([start_l[0], start_r[0]], [start_l[1], start_r[1]],
                color="white", linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.text((start_l[0] + start_r[0]) / 2,
                (start_l[1] + start_r[1]) / 2,
                "S/F", color="white", fontsize=7, ha="center", va="center",
                fontweight="bold", zorder=5)

        # --- Draw the car (if episode has started) ---
        if self._agent_pos is not None:
            car_x, car_y = self._agent_pos[:2]

            # Find nearest centerline index to get heading
            diffs = self.centerline - np.array([car_x, car_y])
            idx = int(np.argmin((diffs ** 2).sum(axis=1)))
            heading = self.tangents[idx]  # unit vector

            arrow_len = TRACK_WIDTH * 0.6
            ax.annotate(
                "",
                xy=(car_x + heading[0] * arrow_len, car_y + heading[1] * arrow_len),
                xytext=(car_x, car_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#f4d35e",
                    lw=2.0,
                    mutation_scale=14,
                ),
                zorder=6,
            )
            ax.plot(car_x, car_y, "o", color="#f4d35e",
                    markersize=7, markeredgecolor="white", markeredgewidth=0.8, zorder=7)

        # --- Title / step counter ---
        step_str = f"Step {getattr(self, '_step', 0)}"
        ax.set_title(step_str, color="white", fontsize=10, pad=6)

        # --- Render to numpy array ---
        fig.tight_layout(pad=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)

        import PIL.Image
        img = np.array(PIL.Image.open(buf).convert("RGB"))
        return img

    def close(self):
        pass