"""Exceções fail-fast do protocolo científico."""


class ScientificProtocolError(ValueError):
    """Erro bloqueante que invalida uma execução científica."""


class OntologyLeakageError(ScientificProtocolError):
    pass


class OntologyQualityError(ScientificProtocolError):
    pass


class OntologyConsistencyError(OntologyQualityError):
    pass


class FeatureSpaceMismatchError(ScientificProtocolError):
    pass


class ClassOrderMismatchError(ScientificProtocolError):
    pass


class DataContractError(ScientificProtocolError):
    pass


class PreprocessingError(ScientificProtocolError):
    pass


class TrainingError(ScientificProtocolError):
    pass


class NonConvergenceError(TrainingError):
    pass


class InsufficientSignalError(TrainingError):
    pass


class ArtifactCompatibilityError(ScientificProtocolError):
    pass


__all__ = [
    "ScientificProtocolError", "OntologyLeakageError", "OntologyQualityError",
    "OntologyConsistencyError", "FeatureSpaceMismatchError",
    "ClassOrderMismatchError", "DataContractError", "PreprocessingError",
    "TrainingError", "NonConvergenceError", "InsufficientSignalError",
    "ArtifactCompatibilityError",
]
