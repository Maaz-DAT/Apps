"""
DF Foundations – Phase 1 (A-01, A-02, A-03)

Owned by Afaq.
"""

from .conventions import C, wrap_phase, wavelength, deg2rad, rad2deg
from .arrays import (
    steering_vector_uca,
    steering_vector_soda,
    spinning_pattern,
    validate_soda_geometry,
    uca_element_positions,
    soda_element_positions,
)
from .impairments import apply_impairments, apply_position_error
from .signals import generate_baseband
from .codecs import IQBlock, PDW, IQDW, Bearing, encode_pdw, decode_pdw, write_pdw_file, read_pdw_file
from .pdw_extractor import extract_pdws
from .calibration import CalibrationTable

__all__ = [
    "C",
    "wrap_phase",
    "wavelength",
    "steering_vector_uca",
    "steering_vector_soda",
    "spinning_pattern",
    "validate_soda_geometry",
    "generate_baseband",
    "apply_impairments",
    "IQBlock",
    "PDW",
    "IQDW",
    "Bearing",
    "extract_pdws",
    "CalibrationTable",
]
