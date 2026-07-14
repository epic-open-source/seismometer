import numpy as np
import pandas as pd
import pytest

from seismometer.data.filter import FilterRule, filter_rule_from_cohort_dictionary


@pytest.fixture
def test_dataframe():
    return pd.DataFrame(
        {
            "Val": np.random.normal(20, 10, 50),
            "Cat": np.random.choice(["A", "B", "C", "D"], size=50, replace=True, p=[2 / 5, 1 / 2, 3 / 40, 1 / 40]),
            "T/F": np.random.choice([0, 1, np.nan], size=50, replace=True, p=[4 / 7, 2 / 7, 1 / 7]),
            "Other": np.random.normal(0, 5, 50),
            "Med": np.random.choice([-1, 0, 1, 2, 3], size=50, replace=True, p=[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]),
            "Filtered": [0] * 45 + [1] * 5,
        }
    )


@pytest.fixture(autouse=True)
def reset_FilterRule_min_rows():
    original = FilterRule.MIN_ROWS
    yield
    FilterRule.MIN_ROWS = original


class TestFilterRulesFiltering:
    def test_filter_rule_can_hide_small_counts(self, test_dataframe):
        assert len(FilterRule("Filtered", "==", 1).filter(test_dataframe)) == 0
        assert len(FilterRule("Filtered", "==", 0).filter(test_dataframe)) == 45

    def test_filter_universal_rule_equals(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        pd.testing.assert_frame_equal(FilterRule.all().filter(test_dataframe), test_dataframe)
        assert len(FilterRule.none().filter(test_dataframe)) == 0
        assert all(FilterRule.none().filter(test_dataframe).columns == test_dataframe.columns)

    @pytest.mark.parametrize(
        "column,operator,value,expected_mask_expr,expected_filter_expr",
        [
            # Comparison operators
            ("T/F", "==", 0, lambda df: df["T/F"] == 0, lambda df: df[df["T/F"] == 0]),
            ("T/F", "!=", 0, lambda df: df["T/F"] != 0, lambda df: df[df["T/F"] != 0]),
            ("Val", "<", 20, lambda df: df["Val"] < 20, lambda df: df[df["Val"] < 20]),
            ("Val", ">", 20, lambda df: df["Val"] > 20, lambda df: df[df["Val"] > 20]),
            ("Val", "<=", 20, lambda df: df["Val"] <= 20, lambda df: df[df["Val"] <= 20]),
            ("Val", ">=", 20, lambda df: df["Val"] >= 20, lambda df: df[df["Val"] >= 20]),
            # Set operators
            (
                "Cat",
                "isin",
                ["A", "B"],
                lambda df: df["Cat"].isin(["A", "B"]),
                lambda df: df[df["Cat"].isin(["A", "B"])],
            ),
            (
                "Cat",
                "notin",
                ["A", "B"],
                lambda df: ~df["Cat"].isin(["A", "B"]),
                lambda df: df[~df["Cat"].isin(["A", "B"])],
            ),
            # Null operators
            ("T/F", "isna", None, lambda df: df["T/F"].isna(), lambda df: df[df["T/F"].isna()]),
            ("T/F", "notna", None, lambda df: ~df["T/F"].isna(), lambda df: df[~df["T/F"].isna()]),
        ],
        ids=[
            "equals",
            "not_equals",
            "less_than",
            "greater_than",
            "less_than_or_eq",
            "greater_than_or_eq",
            "isin",
            "notin",
            "isna",
            "notna",
        ],
    )
    def test_filter_base_rule_operators(
        self, test_dataframe, column, operator, value, expected_mask_expr, expected_filter_expr
    ):
        """Test FilterRule operators with parametrization to reduce code duplication."""
        FilterRule.MIN_ROWS = None

        # Create the rule (handle operators that don't need a value)
        if operator in ["isna", "notna"]:
            rule = FilterRule(column, operator)
        else:
            rule = FilterRule(column, operator, value)

        # Test mask
        assert rule.mask(test_dataframe).equals(expected_mask_expr(test_dataframe))

        # Test filter
        assert rule.filter(test_dataframe).equals(expected_filter_expr(test_dataframe))

    @pytest.mark.parametrize(
        "k, expected_values",
        [
            (1, {"A"}),  # "A" appears 3 times
            (2, {"A", "B"}),  # "B" appears 2 times
        ],
    )
    def test_filter_base_rule_topk(self, k, expected_values):
        df = pd.DataFrame({"Cat": ["A", "A", "B", "C", "A", "B", "D"]})
        FilterRule.MIN_ROWS = None
        rule = FilterRule("Cat", "topk", k)
        result = rule.filter(df)
        assert set(result["Cat"].unique()) == expected_values

    @pytest.mark.parametrize(
        "k, excluded_values",
        [
            (1, {"A"}),
            (2, {"A", "B"}),
        ],
    )
    def test_filter_base_rule_nottopk(self, k, excluded_values):
        df = pd.DataFrame({"Cat": ["A", "A", "B", "C", "A", "B", "D"]})
        FilterRule.MIN_ROWS = None
        rule = FilterRule("Cat", "nottopk", k)
        result = rule.filter(df)
        assert not any(val in result["Cat"].unique() for val in excluded_values)

    @pytest.mark.parametrize(
        "data, k, expected_topk, expected_nottopk",
        [
            (["A", "A", "B", "B", "C", "D"], 1, {"A"}, {"B", "C", "D"}),
            (["A", "A", "B", "B", "C", "D"], 2, {"A", "B"}, {"C", "D"}),
            (["B", "B", "A", "A", "C", "D"], 2, {"A", "B"}, {"C", "D"}),  # ties, test alphabetical tie-breaking
            (["C", "B", "A", "A", "B", "C"], 2, {"A", "B"}, {"C"}),  # same freq, resolve by label
            (["X"] * 3 + ["Y"] * 3 + ["Z"] * 3 + ["W"], 2, {"X", "Y"}, {"Z", "W"}),
            (["X"] * 3 + ["Y"] * 3 + ["Z"] * 3 + ["W"], 3, {"X", "Y", "Z"}, {"W"}),
        ],
    )
    def test_filter_topk_and_nottopk_tie_handling(self, data, k, expected_topk, expected_nottopk):
        df = pd.DataFrame({"cat": data})
        FilterRule.MIN_ROWS = None

        topk_result = FilterRule("cat", "topk", k).filter(df)
        nottopk_result = FilterRule("cat", "nottopk", k).filter(df)

        assert set(topk_result["cat"].unique()) == expected_topk
        assert set(nottopk_result["cat"].unique()) == expected_nottopk


class TestFilterRuleConstructors:
    @pytest.mark.parametrize(
        "rule,exception",
        [
            (("Some", "quasi-equals", "Relation"), ValueError),
            ((FilterRule("Good", "==", "Rule"), "or", "NotARule"), TypeError),
            (("NotARule", "or", FilterRule("Good", "==", "Rule")), TypeError),
            (("NeedsAList", "notin", "a_string"), TypeError),
            (("NeedsAList", "isin", "a_string"), TypeError),
            (("NeedsNone", "notna", "a_string"), TypeError),
            (("NeedsNone", "isna", "a_string"), TypeError),
            ((None, "none", "a_string"), TypeError),
            (("ShouldBeNone", "all", None), TypeError),
            (("Col", "topk", "not_an_int"), TypeError),
            (("Col", "topk", 0), ValueError),
            (("Col", "topk", -1), ValueError),
            ((123, "topk", 3), TypeError),
            (("Col", "nottopk", "bad"), TypeError),
            (("Col", "nottopk", 0), ValueError),
            (("Col", "nottopk", -5), ValueError),
            ((123, "nottopk", 3), TypeError),
        ],
    )
    def test_filter_base_bad_relation(self, rule, exception):
        with pytest.raises(exception):
            FilterRule(*rule)

    @pytest.mark.parametrize(
        "left,right",
        [
            (FilterRule.isin("Col", ["list"]), FilterRule("Col", "isin", ["list"])),
            (FilterRule.notin("Col", ["list"]), FilterRule("Col", "notin", ["list"])),
            (FilterRule.isna("Col"), FilterRule("Col", "isna")),
            (FilterRule.notna("Col"), FilterRule("Col", "notna")),
            (FilterRule.lt("Col", 1), FilterRule("Col", "<", 1)),
            (FilterRule.gt("Col", 1), FilterRule("Col", ">", 1)),
            (FilterRule.leq("Col", 1), FilterRule("Col", "<=", 1)),
            (FilterRule.geq("Col", 1), FilterRule("Col", ">=", 1)),
            (FilterRule.eq("Col", 1), FilterRule("Col", "==", 1)),
            (FilterRule.neq("Col", 1), FilterRule("Col", "!=", 1)),
            (FilterRule.between("Col", lower=0), FilterRule("Col", ">=", 0)),
            (FilterRule.between("Col", upper=1), FilterRule("Col", "<", 1)),
            (FilterRule.between("Col", lower=0, upper=1), FilterRule("Col", ">=", 0) & FilterRule("Col", "<", 1)),
            (FilterRule.between("Col", 0, 1), FilterRule("Col", ">=", 0) & FilterRule("Col", "<", 1)),
            (FilterRule.between("Col", 0), FilterRule("Col", ">=", 0)),
        ],
    )
    def test_base_convience_methods(self, left, right):
        assert left == right

    def test_between_requires_bounds(self):
        with pytest.raises(ValueError):
            FilterRule.between("Col")

    @pytest.mark.parametrize(
        "count, class_default, expected_right, expected_cats",
        [
            (3, None, 3, ["A", "B", "C"]),  # Explicit count
            (3, 5, 3, ["A", "B", "C"]),  # Explicit count with class default not None
            (None, 2, 2, ["A", "B"]),  # No count → fallback to class default
        ],
    )
    def test_from_filter_config_topk_behavior(self, monkeypatch, count, class_default, expected_right, expected_cats):
        from seismometer.configuration.model import FilterConfig

        df = pd.DataFrame({"Cat": ["A", "A", "B", "B", "C", "D", "E", "C", "D", "E"]})
        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        monkeypatch.setattr(FilterRule, "MAXIMUM_NUM_COHORTS", class_default)
        config = FilterConfig(source="Cat", action="keep_top", count=count)
        rule = FilterRule.from_filter_config(config)

        assert rule.relation == "topk"
        assert rule.right == expected_right

        result = rule.filter(df)
        assert sorted(result["Cat"].unique()) == expected_cats

    def test_from_filter_config_include_with_values(self, monkeypatch):
        """Test action='include' with values parameter creates isin rule."""
        from seismometer.configuration.model import FilterConfig

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Cat": ["A", "B", "C", "D", "E"]})
        config = FilterConfig(source="Cat", action="include", values=["A", "B"])
        rule = FilterRule.from_filter_config(config)

        assert rule.relation == "isin"
        assert rule.left == "Cat"
        assert set(rule.right) == {"A", "B"}

        result = rule.filter(df)
        assert sorted(result["Cat"].unique()) == ["A", "B"]

    def test_from_filter_config_exclude_with_values(self, monkeypatch):
        """Test action='exclude' with values parameter creates negated isin rule."""
        from seismometer.configuration.model import FilterConfig

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Cat": ["A", "B", "C", "D", "E"]})
        config = FilterConfig(source="Cat", action="exclude", values=["A", "B"])
        rule = FilterRule.from_filter_config(config)

        assert rule.relation == "notin"
        assert rule.left == "Cat"

        result = rule.filter(df)
        assert sorted(result["Cat"].unique()) == ["C", "D", "E"]

    def test_from_filter_config_include_with_range_both_bounds(self, monkeypatch):
        """Test action='include' with range (min and max) creates compound rule."""
        from seismometer.configuration.model import FilterConfig, FilterRange

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        config = FilterConfig(source="Val", action="include", range=FilterRange(min=3, max=8))
        rule = FilterRule.from_filter_config(config)

        result = rule.filter(df)
        assert list(result["Val"]) == [3, 4, 5, 6, 7]  # min inclusive, max exclusive

    def test_from_filter_config_include_with_range_min_only(self, monkeypatch):
        """Test action='include' with range (min only) creates >= rule."""
        from seismometer.configuration.model import FilterConfig, FilterRange

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Val": [1, 2, 3, 4, 5]})
        config = FilterConfig(source="Val", action="include", range=FilterRange(min=3))
        rule = FilterRule.from_filter_config(config)

        result = rule.filter(df)
        assert list(result["Val"]) == [3, 4, 5]

    def test_from_filter_config_include_with_range_max_only(self, monkeypatch):
        """Test action='include' with range (max only) creates < rule."""
        from seismometer.configuration.model import FilterConfig, FilterRange

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Val": [1, 2, 3, 4, 5]})
        config = FilterConfig(source="Val", action="include", range=FilterRange(max=3))
        rule = FilterRule.from_filter_config(config)

        result = rule.filter(df)
        assert list(result["Val"]) == [1, 2]

    def test_from_filter_config_exclude_with_range(self, monkeypatch):
        """Test action='exclude' with range negates the range rule."""
        from seismometer.configuration.model import FilterConfig, FilterRange

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        config = FilterConfig(source="Val", action="exclude", range=FilterRange(min=3, max=8))
        rule = FilterRule.from_filter_config(config)

        result = rule.filter(df)
        # Should exclude [3,4,5,6,7], keep [1,2,8,9,10]
        assert list(result["Val"]) == [1, 2, 8, 9, 10]

    def test_from_filter_config_invalid_action_raises(self):
        """Test invalid action raises ValidationError from Pydantic."""
        from pydantic import ValidationError

        from seismometer.configuration.model import FilterConfig

        # Pydantic validates action field, so invalid values raise ValidationError at creation
        with pytest.raises(ValidationError, match="Input should be 'include', 'exclude' or 'keep_top'"):
            FilterConfig(source="Col", action="invalid_action")

    def test_from_filter_config_topk_with_none_maximum_returns_all(self, monkeypatch):
        """Test keep_top with MAXIMUM_NUM_COHORTS=None and count=None returns all() rule."""
        from seismometer.configuration.model import FilterConfig

        monkeypatch.setattr(FilterRule, "MAXIMUM_NUM_COHORTS", None)
        config = FilterConfig(source="Cat", action="keep_top", count=None)
        rule = FilterRule.from_filter_config(config)

        assert rule == FilterRule.all()

    def test_from_filter_config_list_with_none(self):
        """Test from_filter_config_list with None returns all() rule."""
        rule = FilterRule.from_filter_config_list(None)
        assert rule == FilterRule.all()

    def test_from_filter_config_list_with_empty_list(self):
        """Test from_filter_config_list with empty list returns all() rule."""
        rule = FilterRule.from_filter_config_list([])
        assert rule == FilterRule.all()

    def test_from_filter_config_list_with_single_config(self, monkeypatch):
        """Test from_filter_config_list with single config creates that rule."""
        from seismometer.configuration.model import FilterConfig

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Cat": ["A", "B", "C"]})
        config = FilterConfig(source="Cat", action="include", values=["A"])
        rule = FilterRule.from_filter_config_list([config])

        result = rule.filter(df)
        assert list(result["Cat"]) == ["A"]

    def test_from_filter_config_list_with_multiple_configs(self, monkeypatch):
        """Test from_filter_config_list with multiple configs combines with AND logic."""
        from seismometer.configuration.model import FilterConfig, FilterRange

        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Cat": ["A", "A", "B", "B", "C"], "Val": [1, 5, 2, 6, 3]})
        config1 = FilterConfig(source="Cat", action="include", values=["A", "B"])
        config2 = FilterConfig(source="Val", action="include", range=FilterRange(min=2, max=6))
        rule = FilterRule.from_filter_config_list([config1, config2])

        result = rule.filter(df)
        # Should keep rows where Cat in ["A","B"] AND Val in [2,6)
        # That's: B,2 and A,5
        assert len(result) == 2
        assert set(result["Cat"]) == {"A", "B"}
        assert all((result["Val"] >= 2) & (result["Val"] < 6))


class TestFilterRuleCombinationLogic:
    @pytest.mark.parametrize(
        "left,right",
        [
            (FilterRule.all() | FilterRule.none(), FilterRule.all()),
            (FilterRule.all() & FilterRule.none(), FilterRule.none()),
            (FilterRule.none() | FilterRule.all(), FilterRule.all()),
            (FilterRule.none() & FilterRule.all(), FilterRule.none()),
        ],
    )
    def test_universal_idempotence(self, left, right):
        assert left == right

    @pytest.mark.parametrize(
        "left,right",
        [
            (FilterRule.all() | FilterRule.eq("Column", "Value"), FilterRule.all()),
            (FilterRule.eq("Column", "Value") | FilterRule.all(), FilterRule.all()),
            (FilterRule.none() | FilterRule.eq("Column", "Value"), FilterRule.eq("Column", "Value")),
            (FilterRule.eq("Column", "Value") | FilterRule.none(), FilterRule.eq("Column", "Value")),
        ],
    )
    def test_universal_conjunction(self, left, right):
        assert left == right

    @pytest.mark.parametrize(
        "left,right",
        [
            (FilterRule.all() & FilterRule.eq("Column", "Value"), FilterRule.eq("Column", "Value")),
            (FilterRule.eq("Column", "Value") & FilterRule.all(), FilterRule.eq("Column", "Value")),
            (FilterRule.none() & FilterRule.eq("Column", "Value"), FilterRule.none()),
            (FilterRule.eq("Column", "Value") & FilterRule.none(), FilterRule.none()),
        ],
    )
    def test_universal_disjunction(self, left, right):
        assert left == right

    def test_conjunction_idempotence(self):
        rule1 = FilterRule("Val", ">=", 20)
        assert rule1 == rule1 | rule1

    def test_disjunction_idempotence(self):
        rule1 = FilterRule("Val", ">=", 20)
        assert rule1 == rule1 & rule1

    def test_filter_conjunction_and(self, test_dataframe):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("T/F", "==", 0)
        rule3 = rule1 & rule2
        assert rule3.mask(test_dataframe).equals((rule1.mask(test_dataframe) & rule2.mask(test_dataframe)))

    def test_filter_disjunction_or(self, test_dataframe):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("T/F", "==", 0)
        rule3 = rule1 | rule2
        assert rule3.mask(test_dataframe).equals((rule1.mask(test_dataframe) | rule2.mask(test_dataframe)))

    @pytest.mark.parametrize(
        "left,right",
        [
            (FilterRule("Val", ">=", 20), ~FilterRule("Val", "<", 20)),
            (FilterRule("Val", ">", 20), ~FilterRule("Val", "<=", 20)),
            (FilterRule("Val", "<=", 20), ~FilterRule("Val", ">", 20)),
            (FilterRule("Val", "<", 20), ~FilterRule("Val", ">=", 20)),
            (FilterRule("Val", "==", 20), ~FilterRule("Val", "!=", 20)),
            (FilterRule("Val", "!=", 20), ~FilterRule("Val", "==", 20)),
            (FilterRule("Val", "isin", [1, 2, 3]), ~FilterRule("Val", "notin", [1, 2, 3])),
            (FilterRule("Val", "notin", [1, 2, 3]), ~FilterRule("Val", "isin", [1, 2, 3])),
            (FilterRule("Val", "isna"), ~FilterRule("Val", "notna")),
            (FilterRule("Cat", "topk", 2), ~FilterRule("Cat", "nottopk", 2)),
            (FilterRule("Cat", "nottopk", 2), ~FilterRule("Cat", "topk", 2)),
            (FilterRule.all(), ~FilterRule.none()),
            (FilterRule.none(), ~FilterRule.all()),
        ],
    )
    def test_negation_logic(self, left, right):
        assert left == right

    def test_de_Morgans_laws_and(self):
        rule1 = FilterRule("Val", "<", 15) & FilterRule("Val", ">=", 30)
        rule2 = ~FilterRule("Val", "<", 15) | ~FilterRule("Val", ">=", 30)
        assert ~rule1 == rule2

    def test_de_Morgans_laws_or(self):
        rule3 = FilterRule("Val", "<", 15) | FilterRule("Val", ">=", 30)
        rule4 = ~FilterRule("Val", "<", 15) & ~FilterRule("Val", ">=", 30)
        assert ~rule3 == rule4

    def test_object_ineq(self):
        assert FilterRule("val", "==", "value") != "NotARule"


class TestFilterRulesAsText:
    @pytest.mark.parametrize(
        "rule,expected",
        [
            (FilterRule.all(), "FilterRule.all()"),
            (FilterRule.none(), "FilterRule.none()"),
            (FilterRule(None, "all", None), "FilterRule.all()"),
            (FilterRule(None, "none", None), "FilterRule.none()"),
        ],
    )
    def test_repr_universal(self, rule, expected):
        assert repr(rule) == expected

    @pytest.mark.parametrize(
        "rule,expected",
        [
            (FilterRule.all(), "Include all"),
            (FilterRule.none(), "Exclude all"),
        ],
    )
    def test_str_universal(self, rule, expected):
        assert str(rule) == expected

    @pytest.mark.parametrize(
        "rule, expected",
        [
            (FilterRule.isna("cat"), "FilterRule.isna('cat')"),
            (FilterRule.notna("cat"), "FilterRule.notna('cat')"),
            (FilterRule("cat", "isna", None), "FilterRule.isna('cat')"),
            (FilterRule("cat", "notna", None), "FilterRule.notna('cat')"),
        ],
    )
    def test_repr_unary(self, rule, expected):
        assert repr(rule) == expected

    @pytest.mark.parametrize(
        "rule, expected", [(FilterRule.isna("cat"), "cat is missing"), (FilterRule.notna("cat"), "cat has a value")]
    )
    def test_str_unary(self, rule, expected):
        assert str(rule) == expected

    @pytest.mark.parametrize(
        "rule, expected",
        [
            (FilterRule("Val", ">=", 20), "FilterRule('Val', '>=', 20)"),
            (FilterRule("Cat", "!=", "A"), "FilterRule('Cat', '!=', 'A')"),
            (FilterRule("Cat", "==", "A"), "FilterRule('Cat', '==', 'A')"),
            (FilterRule("Cat", "isin", ["A", "B"]), "FilterRule('Cat', 'isin', ['A', 'B'])"),
            (FilterRule("Cat", "notin", ["A", "B"]), "FilterRule('Cat', 'notin', ['A', 'B'])"),
            (FilterRule("Cat", "topk", 3), "FilterRule('Cat', 'topk', 3)"),
            (FilterRule("Cat", "nottopk", 2), "FilterRule('Cat', 'nottopk', 2)"),
        ],
    )
    def test_repr_binary(self, rule, expected):
        assert repr(rule) == expected

    @pytest.mark.parametrize(
        "rule, expected",
        [
            (FilterRule("Val", ">=", 20), "Val >= 20"),
            (FilterRule("Cat", "!=", "A"), "Cat is not A"),
            (FilterRule("Cat", "==", "A"), "Cat is A"),
            (FilterRule("Cat", "isin", ["A", "B"]), "Cat is in: A, B"),
            (FilterRule("Cat", "notin", ["A", "B"]), "Cat not in: A, B"),
            (FilterRule("Cat", "topk", 3), "Cat in top 3 values"),
            (FilterRule("Cat", "nottopk", 2), "Cat not in top 2 values"),
        ],
    )
    def test_str_binary(self, rule, expected):
        assert str(rule) == expected

    def test_repr_respects_and_or_logic(self):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("T/F", "==", 0)
        rule3 = FilterRule("Other", "<", 5)
        assert repr(rule1 | rule2) == f"{repr(rule1)} | {repr(rule2)}"
        assert repr(rule1 & rule2) == f"{repr(rule1)} & {repr(rule2)}"
        rule4 = rule1 | rule2 & rule3
        rule5 = rule1 & rule2 | rule3
        assert repr(rule4) == f"{repr(rule1)} | ({repr(rule2)} & {repr(rule3)})"
        assert repr(rule5) == f"({repr(rule1)} & {repr(rule2)}) | {repr(rule3)}"

    def test_str_respects_and_or_logic(self):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("T/F", "==", 0)
        rule3 = FilterRule("Other", "<", 5)
        assert str(rule1 | rule2) == f"{str(rule1)} or {str(rule2)}"
        assert str(rule1 & rule2) == f"{str(rule1)} and {str(rule2)}"
        rule4 = rule1 | rule2 & rule3
        rule5 = rule1 & rule2 | rule3
        assert str(rule4) == f"{str(rule1)} or ({str(rule2)} and {str(rule3)})"
        assert str(rule5) == f"({str(rule1)} and {str(rule2)}) or {str(rule3)}"

    def test_corrupted_rule(self):
        rule = FilterRule("Val", ">=", 20)
        rule.relation = "NotARealRelation"
        with pytest.raises(ValueError):
            str(rule)


class TestFilterRuleFromCohortDictionary:
    def test_matches_cohort(self):
        rule = filter_rule_from_cohort_dictionary(cohort={"-1": (), "One": (1, 2), "two": ("a",), "3": ()})
        assert rule == FilterRule.isin("One", (1, 2)) & FilterRule.isin("two", ("a",))

    def test_matches_default_cohort(self):
        rule = filter_rule_from_cohort_dictionary()
        assert rule == FilterRule.all()


class TestHelperFunctions:
    """Test helper functions that are exported but not directly tested elsewhere."""

    def test_apply_column_comparison_error_handling(self):
        """Test apply_column_comparison error handling with incompatible types."""
        from seismometer.data.filter import apply_column_comparison

        df = pd.DataFrame({"Col": ["a", "b", "c"]})

        # String column compared with integer should raise ValueError
        with pytest.raises(ValueError, match="Values in 'Col' must be comparable to '5'"):
            apply_column_comparison(df, "Col", 5, "<")

    def test_apply_column_comparison_with_valid_comparison(self):
        """Test apply_column_comparison works with valid comparisons."""
        from seismometer.data.filter import apply_column_comparison

        df = pd.DataFrame({"Val": [1, 2, 3, 4, 5]})
        result = apply_column_comparison(df, "Val", 3, "<")

        assert result.equals(df["Val"] < 3)
        assert result.sum() == 2  # Only 1 and 2 are < 3

    def test_apply_topk_filter_with_k_greater_than_unique_values(self):
        """Test topk with k > number of unique values returns all True mask."""
        from seismometer.data.filter import apply_topk_filter

        df = pd.DataFrame({"Cat": ["A", "A", "B", "B", "C"]})
        # Only 3 unique values, but ask for top 5
        mask = apply_topk_filter(df, "Cat", 5)

        assert isinstance(mask, pd.Series)
        assert mask.all()  # All rows should be True
        assert len(mask) == 5

    def test_apply_topk_filter_with_k_equal_to_unique_values(self):
        """Test topk with k == number of unique values returns all True mask."""
        from seismometer.data.filter import apply_topk_filter

        df = pd.DataFrame({"Cat": ["A", "A", "B", "B", "C"]})
        # Exactly 3 unique values, ask for top 3
        mask = apply_topk_filter(df, "Cat", 3)

        assert isinstance(mask, pd.Series)
        assert mask.all()  # All rows should be True

    def test_apply_topk_filter_with_single_unique_value(self):
        """Test topk with DataFrame containing single unique value."""
        from seismometer.data.filter import apply_topk_filter

        df = pd.DataFrame({"Cat": ["A", "A", "A", "A"]})
        mask = apply_topk_filter(df, "Cat", 2)

        assert isinstance(mask, pd.Series)
        assert mask.all()  # All rows should be True since only one unique value
        assert len(mask) == 4

    def test_apply_topk_filter_with_empty_dataframe(self):
        """Test topk with empty DataFrame doesn't crash."""
        from seismometer.data.filter import apply_topk_filter

        df = pd.DataFrame({"Cat": []})
        mask = apply_topk_filter(df, "Cat", 2)

        assert isinstance(mask, pd.Series)
        assert len(mask) == 0


class TestEdgeCases:
    """Test edge cases and error conditions not covered elsewhere."""

    def test_filter_with_missing_column_raises_keyerror(self):
        """Test filtering with non-existent column raises KeyError."""
        df = pd.DataFrame({"Col1": [1, 2, 3]})
        rule = FilterRule("NonExistentColumn", "==", 1)

        with pytest.raises(KeyError):
            rule.filter(df)

    def test_mask_with_missing_column_raises_keyerror(self):
        """Test mask with non-existent column raises KeyError."""
        df = pd.DataFrame({"Col1": [1, 2, 3]})
        rule = FilterRule("NonExistentColumn", "==", 1)

        with pytest.raises(KeyError):
            rule.mask(df)

    def test_filter_with_empty_dataframe(self, monkeypatch):
        """Test filter on empty DataFrame returns empty DataFrame."""
        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Col": []})
        rule = FilterRule("Col", "==", 1)

        result = rule.filter(df)
        assert len(result) == 0
        assert list(result.columns) == ["Col"]

    def test_filter_with_single_row_dataframe(self, monkeypatch):
        """Test filter on single-row DataFrame works correctly."""
        monkeypatch.setattr(FilterRule, "MIN_ROWS", None)
        df = pd.DataFrame({"Col": [1]})
        rule = FilterRule("Col", "==", 1)

        result = rule.filter(df)
        assert len(result) == 1
        assert result["Col"].iloc[0] == 1

    def test_str_with_numeric_isin_values(self):
        """Test __str__ with numeric isin values doesn't crash."""
        rule = FilterRule("Col", "isin", [1, 2, 3])

        # Should not raise AttributeError from .join()
        result = str(rule)
        assert "Col" in result
        assert "is in" in result

    def test_str_with_mixed_type_isin_values(self):
        """Test __str__ with mixed type isin values doesn't crash."""
        rule = FilterRule("Col", "isin", [1, "A", 2.5])

        # Should not raise AttributeError
        result = str(rule)
        assert "Col" in result

    def test_min_rows_boundary_exact_equal(self, monkeypatch):
        """Test MIN_ROWS boundary when len(df) == MIN_ROWS returns empty (exclusive threshold)."""
        monkeypatch.setattr(FilterRule, "MIN_ROWS", 10)
        df = pd.DataFrame({"Col": [1] * 10})  # Exactly 10 rows
        rule = FilterRule("Col", "==", 1)

        result = rule.filter(df)
        # MIN_ROWS uses > comparison, so len == MIN_ROWS returns empty
        assert len(result) == 0

    def test_min_rows_boundary_just_below(self, monkeypatch):
        """Test MIN_ROWS boundary when len(df) < MIN_ROWS returns empty."""
        monkeypatch.setattr(FilterRule, "MIN_ROWS", 10)
        df = pd.DataFrame({"Col": [1] * 9})  # 9 rows, below threshold
        rule = FilterRule("Col", "==", 1)

        result = rule.filter(df)
        # Should return empty because below MIN_ROWS
        assert len(result) == 0

    def test_min_rows_boundary_just_above(self, monkeypatch):
        """Test MIN_ROWS boundary when len(df) > MIN_ROWS returns data."""
        monkeypatch.setattr(FilterRule, "MIN_ROWS", 10)
        df = pd.DataFrame({"Col": [1] * 11})  # 11 rows, above threshold
        rule = FilterRule("Col", "==", 1)

        result = rule.filter(df)
        # Should return the filtered result (11 rows match)
        assert len(result) == 11
