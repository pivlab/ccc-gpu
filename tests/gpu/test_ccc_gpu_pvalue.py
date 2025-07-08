import time
import pytest
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
import pandas as pd
from sklearn.preprocessing import minmax_scale
from ccc.coef.impl_gpu import ccc as ccc_gpu
from ccc.coef.impl import ccc
from utils import clean_gpu_memory, generate_categorical_data


# Original CCC test from tests/test_coef_pval.py
def test_cm_basic_pvalue_n_permutations_not_given():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    cm_value = ccc_gpu(feature0, feature1, pvalue_n_perms=None)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.01, abs=0.01)


# Original CCC test from tests/test_coef_pval.py
def test_cm_basic_pvalue_n_permutations_is_zero():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    cm_value = ccc_gpu(feature0, feature1, pvalue_n_perms=0)

    # Validate
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.01, abs=0.01)


def test_cm_basic_pvalue_n_permutations_is_1():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=1)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.01, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert 0.0 < pvalue <= 1.0
    assert pvalue in (0.5, 1.0)


def test_cm_basic_pvalue_n_permutations_is_10():
    # Prepare
    rs = np.random.RandomState(123)

    # two features on 100 objects (random data)
    feature0 = rs.rand(100)
    feature1 = rs.rand(100)

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=10)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.01, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert 0.0 < pvalue <= 1.0


def test_cm_linear_pvalue_n_permutations_100():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(100)
    feature1 = feature0 * 5.0

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == 1.0

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert pvalue == (0 + 1) / (100 + 1)


def test_cm_quadratic_pvalue():
    # Prepare
    rs = np.random.RandomState(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(rs.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0)

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.49, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert pvalue == (0 + 1) / (100 + 1)


def test_cm_quadratic_noisy_pvalue_with_random_state():
    # Prepare
    rs = np.random.RandomState(1)

    # two features on 100 objects with a quadratic relationship
    feature0 = minmax_scale(rs.rand(100), (-1.0, 1.0))
    feature1 = np.power(feature0, 2.0) + (2.0 * rs.rand(feature0.shape[0]))

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert cm_value == pytest.approx(0.05, abs=0.01)

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert pvalue < 0.15


def test_cm_one_feature_with_all_same_values_pvalue():
    # if there is no variation in at least one of the two variables to be
    #  compared, ccc returns nan

    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects; all values in feature1 are the same
    feature0 = rs.rand(100)
    feature1 = np.array([5] * feature0.shape[0])

    # Run
    res = ccc_gpu(feature0, feature1, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert isinstance(cm_value, np.float32)
    assert np.isnan(cm_value), cm_value

    assert pvalue is not None
    assert isinstance(pvalue, np.float32)
    assert np.isnan(pvalue), pvalue


def test_cm_single_argument_is_matrix():
    # Prepare
    rs = np.random.RandomState(0)

    # two features on 100 objects with a linear relationship
    feature0 = rs.rand(100)
    feature1 = feature0 * 5.0
    feature2 = rs.rand(feature0.shape[0])

    input_data = np.array([feature0, feature1, feature2])

    # Run
    res = ccc_gpu(input_data, pvalue_n_perms=100)

    # Validate
    assert len(res) == 2
    cm_value, pvalue = res
    assert cm_value is not None
    assert hasattr(cm_value, "shape")
    assert cm_value.shape == (3,)
    assert cm_value[0] == 1.0
    assert cm_value[1] < 0.03
    assert cm_value[2] < 0.03

    assert pvalue is not None
    assert hasattr(pvalue, "shape")
    assert pvalue.shape == (3,)
    assert pvalue[0] == (0 + 1) / (100 + 1)
    assert pvalue[1] > 0.10
    assert pvalue[2] > 0.10
