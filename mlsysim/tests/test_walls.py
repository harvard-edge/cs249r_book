"""
Unit tests for mlsysim.core.walls — the 22 ML Systems Wall taxonomy.

Validates registry completeness, lookup helpers, and data integrity.
"""

import pytest

from mlsysim.core.walls import (
    ALL_WALLS,
    COMPUTE,
    Domain,
    Wall,
    taxonomy,
    wall,
    walls_for_resolver,
    walls_in_domain,
)


class TestWallRegistry:
    """Verify the wall registry is complete and consistent."""

    def test_exactly_22_walls(self):
        assert len(ALL_WALLS) == 22

    def test_no_duplicate_numbers(self):
        numbers = [w.number for w in ALL_WALLS]
        assert len(numbers) == len(set(numbers))

    def test_all_numbers_1_through_22(self):
        numbers = {w.number for w in ALL_WALLS}
        assert numbers == set(range(1, 23))

    def test_every_wall_has_nonempty_equation(self):
        for w in ALL_WALLS:
            assert w.equation, f"Wall {w.number} ({w.name}) has empty equation"

    def test_every_wall_has_nonempty_constraint(self):
        for w in ALL_WALLS:
            assert w.constraint, f"Wall {w.number} ({w.name}) has empty constraint"


class TestWallLookup:
    """Verify lookup helpers return correct results."""

    def test_wall_1_is_compute(self):
        assert wall(1) is COMPUTE

    def test_wall_invalid_number_raises(self):
        with pytest.raises(KeyError):
            wall(0)

    def test_wall_23_raises(self):
        with pytest.raises(KeyError):
            wall(23)


class TestWallsForResolver:
    """Verify resolver-to-wall mapping."""

    def test_single_node_model_resolves_walls_1_and_2(self):
        walls = walls_for_resolver("SingleNodeModel")
        numbers = {w.number for w in walls}
        assert numbers == {1, 2}

    def test_unknown_resolver_returns_empty(self):
        assert walls_for_resolver("NonExistentSolver") == []


class TestWallsInDomain:
    """Verify domain grouping."""

    def test_node_domain_has_7_walls(self):
        node_walls = walls_in_domain(Domain.NODE)
        assert len(node_walls) == 7

    def test_domain_walls_are_sorted_by_number(self):
        for domain in Domain:
            domain_walls = walls_in_domain(domain)
            numbers = [w.number for w in domain_walls]
            assert numbers == sorted(numbers)

    def test_all_domains_cover_all_walls(self):
        total = sum(len(walls_in_domain(d)) for d in Domain)
        assert total == 22


class TestTaxonomy:
    """Verify the taxonomy display string."""

    def test_returns_nonempty_string(self):
        result = taxonomy()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_all_domain_labels(self):
        result = taxonomy()
        assert "Domain 1" in result
        assert "Domain 6" in result

    def test_contains_wall_names(self):
        result = taxonomy()
        assert "Compute" in result
        assert "Synthesis" in result
