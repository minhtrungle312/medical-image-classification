"""
Data quality tests.

Validates dataset integrity, format correctness, and
ensures the data pipeline handles edge cases properly.
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import patch

from src.data_pipeline import (
    ISICSkinDataset,
    get_transforms,
    compute_class_weights,
    get_weighted_sampler,
    validate_data_quality,
    CLASS_NAMES,
    NUM_CLASSES,
    IMG_SIZE,
)


class TestTransforms:
    def test_train_transform_output_shape(self):
        transform = get_transforms("train")
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_transform_output_shape(self):
        transform = get_transforms("val")
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_transforms_normalize(self):
        transform = get_transforms("val")
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = transform(img)
        # After ImageNet normalization, values should not be in [0,255]
        assert tensor.max() < 10
        assert tensor.min() > -10

    def test_different_input_sizes(self):
        transform = get_transforms("val")
        for size in [(100, 100), (500, 500), (224, 224), (640, 480)]:
            img = Image.new("RGB", size)
            tensor = transform(img)
            assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)


class TestDataset:
    def test_dataset_length(self, tmp_path):
        # Create temporary images
        paths = []
        labels = []
        for i in range(5):
            img_path = tmp_path / f"img_{i}.jpg"
            Image.new("RGB", (224, 224)).save(img_path)
            paths.append(str(img_path))
            labels.append(i % NUM_CLASSES)

        dataset = ISICSkinDataset(paths, labels, get_transforms("val"))
        assert len(dataset) == 5

    def test_dataset_getitem(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

        dataset = ISICSkinDataset([str(img_path)], [0], get_transforms("val"))
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, IMG_SIZE, IMG_SIZE)
        assert label == 0

    def test_dataset_label_range(self, tmp_path):
        paths = []
        labels = list(range(NUM_CLASSES))
        for i in range(NUM_CLASSES):
            img_path = tmp_path / f"img_{i}.jpg"
            Image.new("RGB", (224, 224)).save(img_path)
            paths.append(str(img_path))

        dataset = ISICSkinDataset(paths, labels, get_transforms("val"))
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert 0 <= label < NUM_CLASSES


class TestClassWeights:
    def test_weights_shape(self):
        labels = [0, 0, 0, 1, 1, 2, 3, 4, 5, 6]
        weights = compute_class_weights(labels)
        assert weights.shape == (NUM_CLASSES,)

    def test_weights_inverse_frequency(self):
        # Majority class should have lower weight
        labels = [0] * 100 + [1] * 10
        weights = compute_class_weights(labels)
        assert weights[0] < weights[1]

    def test_weights_all_positive(self):
        labels = [0, 1, 2, 3, 4, 5, 6]
        weights = compute_class_weights(labels)
        assert (weights > 0).all()


class TestWeightedSampler:
    def test_sampler_creation(self):
        labels = [0] * 100 + [1] * 10 + [2] * 5
        sampler = get_weighted_sampler(labels)
        assert len(sampler) == len(labels)

    def test_sampler_oversamples_minority(self):
        np.random.seed(42)
        labels = [0] * 100 + [1] * 10
        sampler = get_weighted_sampler(labels)
        # Sample and check distribution is more balanced
        indices = list(iter(sampler))
        sampled_labels = [labels[i] for i in indices]
        class_1_ratio = sum(1 for l in sampled_labels if l == 1) / len(sampled_labels)
        # With weighted sampling, class 1 should be more than 10/110
        assert class_1_ratio > 0.1


class TestClassConstants:
    def test_num_classes(self):
        assert NUM_CLASSES == 9

    def test_class_names_count(self):
        assert len(CLASS_NAMES) == NUM_CLASSES

    def test_class_names_content(self):
        assert "melanoma" in CLASS_NAMES
        assert "nevus" in CLASS_NAMES
        assert "basal cell carcinoma" in CLASS_NAMES
