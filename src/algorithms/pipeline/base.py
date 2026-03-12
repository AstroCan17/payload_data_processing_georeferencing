"""
Base pipeline class for image processing operations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process the input data.

        Args:
            data: Input data to process
            **kwargs: Additional keyword arguments

        Returns:
            Processed data
        """
        pass


class Pipeline:
    """Main pipeline class for managing processing steps."""

    def __init__(self, name: str):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.config: Dict[str, Any] = {}

    def add_step(self, step: PipelineStep) -> None:
        """Add a processing step to the pipeline.

        Args:
            step: PipelineStep instance to add
        """
        self.steps.append(step)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set pipeline configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def run(self, data: Any) -> Any:
        """Run the complete pipeline.

        Args:
            data: Input data to process

        Returns:
            Processed data after all pipeline steps
        """
        current_data = data
        for step in self.steps:
            current_data = step.process(current_data, **self.config)
        return current_data
