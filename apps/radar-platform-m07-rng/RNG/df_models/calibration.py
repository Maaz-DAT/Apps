"""
A-03 — Calibration table builder for CIDF.

REQ-A03-1: Table shape = frequencies × azimuths (2° grid) × N complex values,
           normalised to the reference element.
REQ-A03-2: Built from the A-01 array model; a loader accepts a measured table later.
REQ-A03-3: Interpolation between frequency points (linear in unwrapped phase).
REQ-A03-4: Self-check autocorrelation vs azimuth is computed and can be plotted.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .arrays import steering_vector_uca
from .conventions import wrap_phase


class CalibrationTable:
    """
    Frequency × azimuth × complex-response table for correlative interferometer.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        azimuths_rad: np.ndarray,
        responses: np.ndarray,          # shape (n_freq, n_az, n_elements)
        n_elements: int,
        radius: float,
        array_id: str = "uca_default",
    ):
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
        self.azimuths_rad = np.asarray(azimuths_rad, dtype=np.float64)
        self.responses = np.asarray(responses, dtype=np.complex64)
        self.n_elements = n_elements
        self.radius = radius
        self.array_id = array_id

        assert self.responses.shape == (
            len(self.frequencies_hz),
            len(self.azimuths_rad),
            n_elements,
        )

    # ------------------------------------------------------------------
    # Builder from geometry (A-01 model)
    # ------------------------------------------------------------------

    @classmethod
    def from_geometry(
        cls,
        n_elements: int = 5,
        radius: float = 0.5,
        freq_start_hz: float = 1e9,
        freq_stop_hz: float = 2e9,
        freq_step_hz: float = 50e6,
        az_step_deg: float = 2.0,
        array_id: str = "uca_default",
    ) -> "CalibrationTable":
        """
        Build the table from the ideal UCA steering-vector model.
        Azimuth grid defaults to 2° as in the DAT CIDF system.
        """
        frequencies = np.arange(freq_start_hz, freq_stop_hz + freq_step_hz / 2, freq_step_hz)
        azimuths_deg = np.arange(0.0, 360.0, az_step_deg)
        azimuths_rad = np.deg2rad(azimuths_deg)

        n_freq = len(frequencies)
        n_az = len(azimuths_rad)
        responses = np.zeros((n_freq, n_az, n_elements), dtype=np.complex64)

        for fi, f in enumerate(frequencies):
            for ai, az in enumerate(azimuths_rad):
                sv = steering_vector_uca(n_elements, radius, f, az)
                # Normalise to reference element (element 0)
                ref = sv[0]
                if abs(ref) > 1e-12:
                    sv = sv / ref
                responses[fi, ai] = sv

        return cls(
            frequencies_hz=frequencies,
            azimuths_rad=azimuths_rad,
            responses=responses,
            n_elements=n_elements,
            radius=radius,
            array_id=array_id,
        )

    # ------------------------------------------------------------------
    # Lookup / interpolation (REQ-A03-3)
    # ------------------------------------------------------------------

    def get_response(self, frequency_hz: float, azimuth_rad: float) -> np.ndarray:
        """
        Return the complex response vector at the requested frequency and azimuth.
        Frequency is interpolated (linear in unwrapped phase).
        Azimuth is taken from the nearest table entry (table stays at 2°).
        """
        # Nearest azimuth
        az_idx = int(np.argmin(np.abs(self.azimuths_rad - azimuth_rad)))
        # Frequency interpolation
        if frequency_hz <= self.frequencies_hz[0]:
            return self.responses[0, az_idx].copy()
        if frequency_hz >= self.frequencies_hz[-1]:
            return self.responses[-1, az_idx].copy()

        fi = np.searchsorted(self.frequencies_hz, frequency_hz) - 1
        f0, f1 = self.frequencies_hz[fi], self.frequencies_hz[fi + 1]
        alpha = (frequency_hz - f0) / (f1 - f0)

        r0 = self.responses[fi, az_idx]
        r1 = self.responses[fi + 1, az_idx]

        # Linear interpolation of unwrapped phase + magnitude
        mag0, mag1 = np.abs(r0), np.abs(r1)
        ph0 = np.unwrap(np.angle(r0))
        ph1 = np.unwrap(np.angle(r1))
        # Make phases continuous between the two frequency points
        dph = wrap_phase(ph1 - ph0)
        ph = ph0 + alpha * dph
        mag = (1 - alpha) * mag0 + alpha * mag1
        return (mag * np.exp(1j * ph)).astype(np.complex64)

    def correlation_surface(self, measurement: np.ndarray, frequency_hz: float) -> np.ndarray:
        """
        Compute correlation coefficient vs every table azimuth at the given frequency.
        measurement : complex vector (n_elements,), already normalised to ref element.
        Returns array of shape (n_azimuths,).
        """
        # Interpolate whole azimuth cut at this frequency
        n_az = len(self.azimuths_rad)
        table_cut = np.zeros((n_az, self.n_elements), dtype=np.complex64)
        for ai in range(n_az):
            table_cut[ai] = self.get_response(frequency_hz, self.azimuths_rad[ai])

        # Normalised correlation |aᴴ x| / (‖a‖ ‖x‖)
        meas_norm = measurement / (np.linalg.norm(measurement) + 1e-30)
        corr = np.abs(table_cut.conj() @ meas_norm)
        table_norms = np.linalg.norm(table_cut, axis=1) + 1e-30
        corr = corr / table_norms
        return corr.astype(np.float64)

    # ------------------------------------------------------------------
    # Self-check: autocorrelation vs azimuth (REQ-A03-4)
    # ------------------------------------------------------------------

    def autocorrelation(self, frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """
        For a source at 0°, compute correlation vs every table azimuth.
        Returns (azimuths_deg, correlation).
        """
        # Ideal measurement at az=0
        meas = self.get_response(frequency_hz, 0.0)
        corr = self.correlation_surface(meas, frequency_hz)
        return np.rad2deg(self.azimuths_rad), corr

    # ------------------------------------------------------------------
    # Persistence (so a measured table can replace the geometric one later)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            frequencies_hz=self.frequencies_hz,
            azimuths_rad=self.azimuths_rad,
            responses=self.responses,
            n_elements=self.n_elements,
            radius=self.radius,
        )
        # small JSON sidecar for human-readable metadata
        meta = {
            "array_id": self.array_id,
            "n_elements": self.n_elements,
            "radius_m": self.radius,
            "n_frequencies": len(self.frequencies_hz),
            "n_azimuths": len(self.azimuths_rad),
            "az_step_deg": float(np.rad2deg(self.azimuths_rad[1] - self.azimuths_rad[0]))
            if len(self.azimuths_rad) > 1
            else 0.0,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationTable":
        path = Path(path)
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        array_id = "loaded"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            array_id = meta.get("array_id", "loaded")
        return cls(
            frequencies_hz=data["frequencies_hz"],
            azimuths_rad=data["azimuths_rad"],
            responses=data["responses"],
            n_elements=int(data["n_elements"]),
            radius=float(data["radius"]),
            array_id=array_id,
        )
