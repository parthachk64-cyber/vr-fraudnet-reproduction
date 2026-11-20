"""Exception types. Every one names the missing input or violated rule, so a run fails early and usefully."""
from __future__ import annotations


class VRFraudNetError(Exception):
    """Base class."""


class MissingInputError(VRFraudNetError):
    """A dataset, model, adapter or artifact required for the requested step is absent."""

    def __init__(self, what: str, *, needed_for: str, how_to_fix: str):
        self.what, self.needed_for, self.how_to_fix = what, needed_for, how_to_fix
        super().__init__(f"MISSING {what} (needed for {needed_for}). {how_to_fix}")


class FixtureInResearchModeError(VRFraudNetError):
    """A fixture/synthetic component was supplied where benchmark (research) mode requires a real one."""


class LeakageError(VRFraudNetError):
    """A partition, retrieval or graph rule that guards against leakage was violated."""


class SchemaError(VRFraudNetError):
    """A rationale object does not satisfy the versioned schema."""


class OptionalDependencyError(VRFraudNetError):
    """An optional extra is needed. The message names the extra to install."""

    def __init__(self, package: str, extra: str, feature: str):
        super().__init__(f"{feature} needs '{package}'. Install it with: pip install 'vr-fraudnet-reproduction[{extra}]'")


class ProtocolError(VRFraudNetError):
    """The requested evaluation cannot be computed as specified (for example a no-positive window)."""
