"""Compatibility imports for the direct-coordinate drift implementation."""

from .unnormalized_drifting import DirectCoordinateDrift, median_bandwidth

# Preserve the original public class name while routing it to coordinate drift.
UnnormalizedDrift = DirectCoordinateDrift

__all__ = ["DirectCoordinateDrift", "UnnormalizedDrift", "median_bandwidth"]
