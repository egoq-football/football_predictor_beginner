import numpy as np

from football_predictor.model import ConsensusBlender, VectorScaler, expected_calibration_error


def test_vector_scaler_returns_valid_probabilities():
    probs = np.array([
        [0.20, 0.30, 0.50],
        [0.45, 0.35, 0.20],
        [0.25, 0.50, 0.25],
        [0.10, 0.20, 0.70],
        [0.60, 0.25, 0.15],
        [0.30, 0.40, 0.30],
    ])
    y = np.array([2, 0, 1, 2, 0, 1])
    calibrated = VectorScaler().fit(probs, y).transform(probs)
    assert calibrated.shape == probs.shape
    assert np.all(calibrated > 0)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_consensus_blender_learns_bounded_weight():
    meta = np.array([[0.2, 0.2, 0.6], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
    consensus = np.array([[0.3, 0.2, 0.5], [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    y = np.array([2, 0, 1])
    blender = ConsensusBlender().fit(meta, consensus, y)
    result = blender.transform(meta, consensus)
    assert 0.0 <= blender.meta_weight_ <= 1.0
    assert np.allclose(result.sum(axis=1), 1.0)


def test_ece_is_zero_for_perfect_binary_like_multiclass_forecasts():
    y = np.array([0, 1, 2])
    probs = np.eye(3)
    assert expected_calibration_error(y, probs) == 0.0
