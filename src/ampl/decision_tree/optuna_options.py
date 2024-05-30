from dataclasses import dataclass, field

from typing import TypeVar, ClassVar

T = TypeVar('T')


@dataclass
class Range:
    min: T
    max: T

    def __post_init__(self):
        assert self.max > self.min


@dataclass
class OptunaOptions:
    n_estimators: Range = field(default_factory=lambda: Range(100, 1000))
    max_depth: Range = field(default_factory=lambda: Range(4, 10))
    learning_rate: Range = field(default_factory=lambda: Range(0.005, 0.05))
    col_sample_by_tree: Range = field(default_factory=lambda: Range(0.2, 0.6))
    subsample: Range = field(default_factory=lambda: Range(0.4, 0.8))
    alpha: Range = field(default_factory=lambda: Range(0.01, 10.0))
    lambda_: Range = field(default_factory=lambda: Range(1e-8, 10.0))
    gamma: Range = field(default_factory=lambda: Range(1e-8, 10.0))
    min_child_weight: Range = field(default_factory=lambda: Range(10, 1000))
