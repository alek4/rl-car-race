import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import fastf1


def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)


def load_track(circuit_name: str, year: int = 2023, spacing_m: float = 5.0) -> dict:
    """
    Load and process a circuit centerline from F1 telemetry data.

    Args:
        circuit_name: Name of the circuit (e.g. 'Monza', 'Monaco', 'Spa')
        year:         F1 season year to fetch data from
        spacing_m:    Distance between resampled centerline points in meters

    Returns:
        A dict with:
            centerline      (N, 2) array of (x, y) in meters, uniformly spaced
            circuit_length_m  total circuit length in meters
            scale           scale factor from native units to meters
            tangents        (N, 2) array of unit tangent vectors at each point
            tck             scipy spline object (for queries at arbitrary arc length)
    """

    # --- Load session ---
    session = fastf1.get_session(year, circuit_name, 'Q')
    session.load()

    circuit_info = session.get_circuit_info()
    lap = session.laps.pick_fastest()

    # --- Raw track points ---
    pos = lap.get_pos_data()
    track = pos.loc[:, ('X', 'Y')].to_numpy()

    # Rotate to canonical orientation
    track_angle = circuit_info.rotation / 180 * np.pi
    rotated = rotate(track, angle=track_angle)
    rotated = np.vstack([rotated, rotated[0]])  # close the loop

    # --- Fit periodic spline ---
    tck, _ = splprep([rotated[:, 0], rotated[:, 1]], s=0, per=True)

    # Compute total length in native units
    u_fine = np.linspace(0, 1, 10000)
    x_fine, y_fine = splev(u_fine, tck)
    pts_fine = np.column_stack([x_fine, y_fine])
    total_length_native = np.sum(np.linalg.norm(np.diff(pts_fine, axis=0), axis=1))

    # Scale factor: native units -> meters
    circuit_length_m = lap.get_telemetry().add_distance()['Distance'].max()
    scale = circuit_length_m / total_length_native

    # --- Resample uniformly every spacing_m meters ---
    n_points = int(circuit_length_m / spacing_m)
    u_uniform = np.linspace(0, 1, n_points)
    x_res, y_res = splev(u_uniform, tck)
    centerline = np.column_stack([x_res, y_res]) * scale

    # --- Compute unit tangent vectors at each centerline point ---
    dx, dy = splev(u_uniform, tck, der=1)
    tangents_raw = np.column_stack([dx, dy]) * scale
    norms = np.linalg.norm(tangents_raw, axis=1, keepdims=True)
    tangents = tangents_raw / norms

    return {
        'centerline': centerline,
        'circuit_length_m': circuit_length_m,
        'scale': scale,
        'tangents': tangents,
        'tck': tck,
    }


def plot_track(track_data: dict, title: str = ''):
    centerline = track_data['centerline']
    plt.figure()
    plt.plot(centerline[:, 0], centerline[:, 1])
    plt.title(title or 'Circuit')
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    circuit = sys.argv[1] if len(sys.argv) > 1 else 'Monza'
    track_data = load_track(circuit)
    plot_track(track_data, title=circuit)