"""
Core processing modules - pure Python, zero Qt imports.
"""
from .input_source import InputSource, SourceType, read_image_file
from .pipeline import downscale_frame, FPSCounter

__all__ = [
    'InputSource',
    'SourceType',
    'read_image_file',
    'downscale_frame',
    'FPSCounter',
]
