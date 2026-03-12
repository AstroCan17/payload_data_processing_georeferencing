"""
Pipeline package for image processing operations.
"""
from .base import Pipeline, PipelineStep
from .steps.image_processing import (BandSegmentationStep,
                                     InterFrameRegistrationStep,
                                     MosaicGenerationStep)

__all__ = [
    'Pipeline',
    'PipelineStep',
    'BandSegmentationStep',
    'InterFrameRegistrationStep',
    'MosaicGenerationStep'
]
