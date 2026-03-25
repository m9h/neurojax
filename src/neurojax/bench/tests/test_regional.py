"""Tests for RegionalParameterSpace — per-region heterogeneous parameter support.

Covers:
- n_params reports correct total dimensionality
- to_flat_array / from_flat_array roundtrip preserves values
- Bounds are correct for each parameter (global and regional)
- Can create parameter dict for vbjax adapter from flat array
- Works with 4-node and 80-node networks
- Edge cases: all global, all regional, mixed
"""

from __future__ import annotations

import numpy as np
import pytest

from neurojax.bench.adapters.regional import RegionalParameterSpace


class TestRegionalParameterSpaceBasic:
    """Core invariants for RegionalParameterSpace."""

    def test_n_params_mixed(self):
        """n_params = n_global + n_regional * n_regions."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=4,
        )
        # 1 global + 2 regional * 4 regions = 9
        assert space.n_params == 9

    def test_n_params_all_global(self):
        """All parameters global: n_params = n_global."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03), "A": (2.0, 5.0)},
            regional_params={},
            n_regions=4,
        )
        assert space.n_params == 2

    def test_n_params_all_regional(self):
        """All parameters regional: n_params = n_regional * n_regions."""
        space = RegionalParameterSpace(
            global_params={},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=4,
        )
        assert space.n_params == 8

    def test_n_params_80_regions(self):
        """Scales correctly with 80-region network."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=80,
        )
        # 1 global + 1 regional * 80 regions = 81
        assert space.n_params == 81


class TestFlatArrayRoundtrip:
    """to_flat_array and from_flat_array are exact inverses."""

    def test_roundtrip_mixed_4node(self):
        """Mixed global/regional params roundtrip on 4-node network."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=4,
        )
        params = {
            "K_gl": np.float64(0.01),
            "A": np.array([3.0, 3.5, 4.0, 2.5]),
            "B": np.array([20.0, 25.0, 30.0, 15.0]),
        }
        flat = space.to_flat_array(params)
        assert flat.shape == (9,)

        recovered = space.from_flat_array(flat)
        assert np.isclose(recovered["K_gl"], 0.01)
        np.testing.assert_array_almost_equal(recovered["A"], [3.0, 3.5, 4.0, 2.5])
        np.testing.assert_array_almost_equal(recovered["B"], [20.0, 25.0, 30.0, 15.0])

    def test_roundtrip_all_global(self):
        """All-global roundtrip: values are scalars."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03), "A": (2.0, 5.0)},
            regional_params={},
            n_regions=4,
        )
        params = {"K_gl": np.float64(0.015), "A": np.float64(3.5)}
        flat = space.to_flat_array(params)
        assert flat.shape == (2,)

        recovered = space.from_flat_array(flat)
        assert np.isclose(recovered["K_gl"], 0.015)
        assert np.isclose(recovered["A"], 3.5)

    def test_roundtrip_all_regional(self):
        """All-regional roundtrip: all values are arrays."""
        space = RegionalParameterSpace(
            global_params={},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        params = {"A": np.array([2.1, 3.2, 4.3, 4.9])}
        flat = space.to_flat_array(params)
        assert flat.shape == (4,)

        recovered = space.from_flat_array(flat)
        np.testing.assert_array_almost_equal(recovered["A"], [2.1, 3.2, 4.3, 4.9])

    def test_roundtrip_80_regions(self):
        """Roundtrip on 80-region network."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=80,
        )
        rng = np.random.default_rng(42)
        params = {
            "K_gl": np.float64(0.02),
            "A": rng.uniform(2.0, 5.0, size=80),
        }
        flat = space.to_flat_array(params)
        assert flat.shape == (81,)

        recovered = space.from_flat_array(flat)
        assert np.isclose(recovered["K_gl"], 0.02)
        np.testing.assert_array_almost_equal(recovered["A"], params["A"])

    def test_roundtrip_preserves_exact_values(self):
        """No floating-point drift from pack/unpack."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        original_flat = np.array([0.015, 2.5, 3.0, 3.5, 4.0])
        recovered = space.from_flat_array(original_flat)
        re_flat = space.to_flat_array(recovered)
        np.testing.assert_array_equal(original_flat, re_flat)


class TestBounds:
    """Bounds generation for optimizer interface."""

    def test_bounds_shape_mixed(self):
        """Bounds have shape (n_params, 2)."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        lower, upper = space.bounds
        assert lower.shape == (5,)
        assert upper.shape == (5,)

    def test_bounds_values_global(self):
        """Global param bounds appear once."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={},
            n_regions=4,
        )
        lower, upper = space.bounds
        assert lower[0] == 0.0
        assert upper[0] == 0.03

    def test_bounds_values_regional_replicated(self):
        """Regional param bounds are replicated n_regions times."""
        space = RegionalParameterSpace(
            global_params={},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        lower, upper = space.bounds
        np.testing.assert_array_equal(lower, [2.0, 2.0, 2.0, 2.0])
        np.testing.assert_array_equal(upper, [5.0, 5.0, 5.0, 5.0])

    def test_bounds_ordering_matches_flat(self):
        """Bounds index matches to_flat_array index."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=3,
        )
        lower, upper = space.bounds
        # Layout: K_gl, A_0, A_1, A_2, B_0, B_1, B_2
        assert len(lower) == 7
        assert lower[0] == 0.0   # K_gl
        assert upper[0] == 0.03  # K_gl
        for i in range(1, 4):
            assert lower[i] == 2.0   # A
            assert upper[i] == 5.0
        for i in range(4, 7):
            assert lower[i] == 10.0  # B
            assert upper[i] == 40.0

    def test_bounds_lower_less_than_upper(self):
        """All lower bounds strictly less than upper bounds."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=80,
        )
        lower, upper = space.bounds
        assert np.all(lower < upper)


class TestParameterNames:
    """Flat parameter names for logging/debugging."""

    def test_param_names_mixed(self):
        """Parameter names list matches flat array ordering."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=3,
        )
        names = space.param_names
        assert names == ["K_gl", "A_0", "A_1", "A_2"]

    def test_param_names_length(self):
        """Number of names equals n_params."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=4,
        )
        assert len(space.param_names) == space.n_params


class TestAdapterCompatibility:
    """Parameters can be converted to a dict usable by VbjaxFitnessAdapter."""

    def test_to_adapter_params_dict(self):
        """from_flat_array produces dict compatible with adapter.evaluate()."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0), "B": (10.0, 40.0)},
            n_regions=4,
        )
        flat = np.array([
            0.01,                       # K_gl
            3.0, 3.5, 4.0, 2.5,        # A per region
            20.0, 25.0, 30.0, 15.0,    # B per region
        ])
        params = space.from_flat_array(flat)

        # K_gl should be a scalar
        assert np.ndim(params["K_gl"]) == 0
        # A and B should be arrays of length n_regions
        assert params["A"].shape == (4,)
        assert params["B"].shape == (4,)

    def test_default_array_creation(self):
        """Can create a default flat array from midpoints of bounds."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        default = space.default_flat_array()
        assert default.shape == (5,)
        assert np.isclose(default[0], 0.015)  # midpoint of K_gl
        for i in range(1, 5):
            assert np.isclose(default[i], 3.5)  # midpoint of A


class TestEdgeCases:
    """Edge cases and validation."""

    def test_wrong_flat_array_length_raises(self):
        """from_flat_array rejects arrays of wrong length."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        with pytest.raises(ValueError, match="Expected.*5"):
            space.from_flat_array(np.zeros(3))

    def test_wrong_regional_array_length_raises(self):
        """to_flat_array rejects regional arrays of wrong length."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=4,
        )
        with pytest.raises(ValueError, match="length 4"):
            space.to_flat_array({
                "K_gl": np.float64(0.01),
                "A": np.array([3.0, 3.5]),  # wrong length
            })

    def test_single_region(self):
        """Works with n_regions=1 (degenerate but valid)."""
        space = RegionalParameterSpace(
            global_params={"K_gl": (0.0, 0.03)},
            regional_params={"A": (2.0, 5.0)},
            n_regions=1,
        )
        assert space.n_params == 2
        params = {"K_gl": np.float64(0.01), "A": np.array([3.0])}
        flat = space.to_flat_array(params)
        recovered = space.from_flat_array(flat)
        assert np.isclose(recovered["K_gl"], 0.01)
        np.testing.assert_array_almost_equal(recovered["A"], [3.0])

    def test_no_params_raises(self):
        """Must have at least one parameter."""
        with pytest.raises(ValueError, match="at least one"):
            RegionalParameterSpace(
                global_params={},
                regional_params={},
                n_regions=4,
            )

    def test_n_regions_positive(self):
        """n_regions must be >= 1."""
        with pytest.raises(ValueError, match="n_regions"):
            RegionalParameterSpace(
                global_params={"K_gl": (0.0, 0.03)},
                regional_params={},
                n_regions=0,
            )
