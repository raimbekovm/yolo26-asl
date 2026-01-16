"""Tests for constants module."""

from src.utils.constants import (
    ASL_CLASSES,
    ASL_GESTURES,
    ASL_LETTERS,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    NUM_KEYPOINTS,
)


class TestASLConstants:
    """Test ASL class constants."""

    def test_asl_letters_count(self):
        """Test that we have 26 ASL letters."""
        assert len(ASL_LETTERS) == 26

    def test_asl_letters_are_uppercase(self):
        """Test that all letters are uppercase A-Z."""
        assert list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == ASL_LETTERS

    def test_asl_gestures_count(self):
        """Test that we have 5 gestures."""
        assert len(ASL_GESTURES) == 5

    def test_total_classes(self):
        """Test total number of classes."""
        assert NUM_CLASSES == 31
        assert len(ASL_CLASSES) == NUM_CLASSES

    def test_class_to_idx_mapping(self):
        """Test class to index mapping."""
        assert CLASS_TO_IDX["A"] == 0
        assert CLASS_TO_IDX["Z"] == 25
        assert CLASS_TO_IDX["Hello"] == 26

    def test_idx_to_class_mapping(self):
        """Test index to class mapping."""
        assert IDX_TO_CLASS[0] == "A"
        assert IDX_TO_CLASS[25] == "Z"
        assert IDX_TO_CLASS[26] == "Hello"

    def test_bidirectional_mapping(self):
        """Test that mappings are consistent."""
        for cls, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == cls


class TestKeypointConstants:
    """Test keypoint constants."""

    def test_num_keypoints(self):
        """Test number of hand keypoints."""
        assert NUM_KEYPOINTS == 21
