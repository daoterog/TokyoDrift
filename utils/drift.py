"""Public imports for coordinate and invariant-descriptor drift."""

from drifting import DirectCoordinateDrift, median_bandwidth

from .descriptors import DescriptorDrift, descriptor_bandwidth, particle_descriptors

# Preserve the original public class name while routing it to coordinate drift.
UnnormalizedDrift = DirectCoordinateDrift

__all__ = [
    "DescriptorDrift",
    "DirectCoordinateDrift",
    "UnnormalizedDrift",
    "descriptor_bandwidth",
    "median_bandwidth",
    "particle_descriptors",
]
