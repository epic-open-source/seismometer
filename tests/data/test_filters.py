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


class TestFilterRules:
    def test_filter_rule_can_hide_small_counts(self, test_dataframe):
        assert len(FilterRule("Filtered", "==", 1).filter(test_dataframe)) == 0
        assert len(FilterRule("Filtered", "==", 0).filter(test_dataframe)) == 45

    def test_filter_base_rule_equals(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("T/F", "==", 0).mask(test_dataframe).equals(test_dataframe["T/F"] == 0)
        assert FilterRule("T/F", "==", 0).filter(test_dataframe).equals(test_dataframe[test_dataframe["T/F"] == 0])

    def test_filter_base_rule_not_equals(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("T/F", "!=", 0).mask(test_dataframe).equals(test_dataframe["T/F"] != 0)
        assert FilterRule("T/F", "!=", 0).filter(test_dataframe).equals(test_dataframe[test_dataframe["T/F"] != 0])

    def test_filter_base_rule_isin(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert (
            FilterRule("Cat", "isin", ["A", "B"]).mask(test_dataframe).equals(test_dataframe["Cat"].isin(["A", "B"]))
        )
        assert (
            FilterRule("Cat", "isin", ["A", "B"])
            .filter(test_dataframe)
            .equals(test_dataframe[test_dataframe["Cat"].isin(["A", "B"])])
        )

    def test_filter_base_rule_notin(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert (
            FilterRule("Cat", "notin", ["A", "B"]).mask(test_dataframe).equals(~test_dataframe["Cat"].isin(["A", "B"]))
        )
        assert (
            FilterRule("Cat", "notin", ["A", "B"])
            .filter(test_dataframe)
            .equals(test_dataframe[~test_dataframe["Cat"].isin(["A", "B"])])
        )

    def test_filter_base_rule_less_than(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("Val", "<", 20).mask(test_dataframe).equals(test_dataframe["Val"] < 20)
        assert FilterRule("Val", "<", 20).filter(test_dataframe).equals(test_dataframe[test_dataframe["Val"] < 20])

    def test_filter_base_rule_greater_then(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("Val", ">", 20).mask(test_dataframe).equals(test_dataframe["Val"] > 20)
        assert FilterRule("Val", ">", 20).filter(test_dataframe).equals(test_dataframe[test_dataframe["Val"] > 20])

    def test_filter_base_rule_less_than_or_eq(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("Val", "<=", 20).mask(test_dataframe).equals(test_dataframe["Val"] <= 20)
        assert FilterRule("Val", "<=", 20).filter(test_dataframe).equals(test_dataframe[test_dataframe["Val"] <= 20])

    def test_filter_base_rule_greater_then_or_eq(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("Val", ">=", 20).mask(test_dataframe).equals(test_dataframe["Val"] >= 20)
        assert FilterRule("Val", ">=", 20).filter(test_dataframe).equals(test_dataframe[test_dataframe["Val"] >= 20])

    def test_filter_base_rule_isna(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("T/F", "isna").mask(test_dataframe).equals(test_dataframe["T/F"].isna())
        assert FilterRule("T/F", "isna").filter(test_dataframe).equals(test_dataframe[test_dataframe["T/F"].isna()])

    def test_filter_base_rule_notna(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        assert FilterRule("T/F", "notna").mask(test_dataframe).equals(~test_dataframe["T/F"].isna())
        assert FilterRule("T/F", "notna").filter(test_dataframe).equals(test_dataframe[~test_dataframe["T/F"].isna()])

    def test_filter_base_bad_relation(self):
        with pytest.raises(ValueError):
            FilterRule("Some", "quasi-equals", "Relation")
        with pytest.raises(TypeError):
            FilterRule(FilterRule("Good", "==", "Rule"), "or", "NotARule")
        with pytest.raises(TypeError):
            FilterRule("NotARule", "or", FilterRule("Good", "==", "Rule"))
        with pytest.raises(TypeError):
            FilterRule("NeedsAList", "notin", "a_string")
        with pytest.raises(TypeError):
            FilterRule("NeedsAList", "isin", "a_string")
        with pytest.raises(TypeError):
            FilterRule("NeedsNone", "notna", "a_string")
        with pytest.raises(TypeError):
            FilterRule("NeedsNone", "isna", "a_string")

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

    def test_repr_respects_and_or_logic(self):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("T/F", "==", 0)
        rule3 = FilterRule("Other", "<", 5)
        assert repr(rule1 | rule2) == f"{rule1} | {rule2}"
        assert repr(rule1 & rule2) == f"{rule1} & {rule2}"
        rule5 = rule1 | rule2 & rule3
        rule6 = rule1 & rule2 | rule3
        assert repr(rule5) == f"{rule1} | ({rule2} & {rule3})"
        assert repr(rule6) == f"({rule1} & {rule2}) | {rule3}"

    def test_repr_simple(self):
        rule1 = FilterRule("Val", ">=", 20)
        rule2 = FilterRule("Cat", "isin", ["A", "B"])
        rule3 = FilterRule("Cat", "!=", "A")
        assert repr(rule1) == "FilterRule('Val', '>=', 20)"
        assert repr(rule2) == "FilterRule('Cat', 'isin', ['A', 'B'])"
        assert repr(rule3) == "FilterRule('Cat', '!=', 'A')"

    def test_negation_logic(self):
        assert FilterRule("Val", ">=", 20) == ~FilterRule("Val", "<", 20)
        assert FilterRule("Val", "<=", 20) == ~FilterRule("Val", ">", 20)
        assert FilterRule("Val", ">", 20) == ~FilterRule("Val", "<=", 20)
        assert FilterRule("Val", "<", 20) == ~FilterRule("Val", ">=", 20)
        assert FilterRule("Val", "==", 20) == ~FilterRule("Val", "!=", 20)
        assert FilterRule("Val", "!=", 20) == ~FilterRule("Val", "==", 20)
        assert FilterRule("Val", "isin", [1, 2, 3]) == ~FilterRule("Val", "notin", [1, 2, 3])
        assert FilterRule("Val", "notin", [1, 2, 3]) == ~FilterRule("Val", "isin", [1, 2, 3])
        assert FilterRule("Val", "isna") == ~FilterRule("Val", "notna")

    def test_de_Morgans_laws_and(self):
        rule1 = FilterRule("Val", "<", 15) & FilterRule("Val", ">=", 30)
        rule2 = ~FilterRule("Val", "<", 15) | ~FilterRule("Val", ">=", 30)
        assert ~rule1 == rule2

    def test_do_Morgans_laws_or(self):
        rule3 = FilterRule("Val", "<", 15) | FilterRule("Val", ">=", 30)
        rule4 = ~FilterRule("Val", "<", 15) & ~FilterRule("Val", ">=", 30)
        assert ~rule3 == rule4

    def test_object_ineq(self):
        assert FilterRule("val", "==", "value") != "NotARule"

    def test_base_convience_methods(self):
        assert FilterRule.isin("Col", ["list"]) == FilterRule("Col", "isin", ["list"])
        assert FilterRule.notin("Col", ["list"]) == FilterRule("Col", "notin", ["list"])

        assert FilterRule.isna("Col") == FilterRule("Col", "isna")
        assert FilterRule.notna("Col") == FilterRule("Col", "notna")

        assert FilterRule.lt("Col", 1) == FilterRule("Col", "<", 1)
        assert FilterRule.gt("Col", 1) == FilterRule("Col", ">", 1)
        assert FilterRule.leq("Col", 1) == FilterRule("Col", "<=", 1)
        assert FilterRule.geq("Col", 1) == FilterRule("Col", ">=", 1)
        assert FilterRule.eq("Col", 1) == FilterRule("Col", "==", 1)
        assert FilterRule.neq("Col", 1) == FilterRule("Col", "!=", 1)
        assert FilterRule.between("Col", lower=0) == FilterRule("Col", ">=", 0)
        assert FilterRule.between("Col", upper=1) == FilterRule("Col", "<", 1)
        assert FilterRule.between("Col", lower=0, upper=1) == FilterRule("Col", ">=", 0) & FilterRule("Col", "<", 1)
        with pytest.raises(ValueError):
            FilterRule.between("Col")


class TestFilterRuleFromCohortDictionary:
    def test_matches_cohort(self):
        rule = filter_rule_from_cohort_dictionary(cohort={"-1": (), "One": (1, 2), "two": ("a",), "3": ()})
        assert rule == FilterRule.isin("One", (1, 2)) & FilterRule.isin("two", ("a",))
