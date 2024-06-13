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

    def test_filter_universal_rule_equals(self, test_dataframe):
        FilterRule.MIN_ROWS = None
        pd.testing.assert_frame_equal(FilterRule.all().filter(test_dataframe),test_dataframe)
        assert len(FilterRule.none().filter(test_dataframe)) == 0
        assert all(FilterRule.none().filter(test_dataframe).columns == test_dataframe.columns)

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
        with pytest.raises(TypeError):
            FilterRule(None, "none", "a_string")
        with pytest.raises(TypeError):
            FilterRule("ShouldBeNone", "all", None)

    def test_universal_idenpotence(self):
        assert (FilterRule.all()  | FilterRule.none()) == FilterRule.all()
        assert (FilterRule.none() | FilterRule.all())  == FilterRule.all()
        assert (FilterRule.none() & FilterRule.all())  == FilterRule.none()
        assert (FilterRule.all()  & FilterRule.none()) == FilterRule.none()

    def test_universal_conjunction(self):
        assert (FilterRule.none() & FilterRule.eq("Column", "Value")) == FilterRule.none()
        assert (FilterRule.all()  & FilterRule.eq("Column", "Value")) == FilterRule.eq("Column", "Value")

        assert (FilterRule.eq("Column", "Value") & FilterRule.none()) == FilterRule.none()
        assert (FilterRule.eq("Column", "Value") & FilterRule.all())  == FilterRule.eq("Column", "Value")

    def test_universal_disjunction(self):
        assert (FilterRule.none() | FilterRule.eq("Column", "Value")) == FilterRule.eq("Column", "Value")
        assert (FilterRule.all()  | FilterRule.eq("Column", "Value")) == FilterRule.all()
        # reverse order
        assert (FilterRule.eq("Column", "Value") | FilterRule.none()) == FilterRule.eq("Column", "Value")
        assert (FilterRule.eq("Column", "Value") | FilterRule.all())  == FilterRule.all()

    def test_conjuction_idenpotence(self):
        rule1 = FilterRule("Val", ">=", 20)
        assert rule1 == rule1 & rule1
        assert rule1 == rule1 | rule1
    
    def test_disjunction_idenpotence(self):
        rule1 = FilterRule("Val", ">=", 20)
        assert rule1 == rule1 | rule1
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

    def test_de_Morgans_laws_or(self):
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


class TestFilterRuleDisplay:
    
    def test_repr_universal(self):
        assert repr(FilterRule.all()) == "FilterRule.all()"
        assert repr(FilterRule.none()) == "FilterRule.none()"
        assert repr(FilterRule(None, "all", None)) == "FilterRule.all()"
        assert repr(FilterRule(None, "none", None)) == "FilterRule.none()"
        
    def test_str_universal(self):
        assert str(FilterRule.all()) == "Include all"
        assert str(FilterRule.none()) == "Exclude all"

    def test_repr_unary(self):
        assert repr(FilterRule.isna("cat")) == "FilterRule.isna('cat')"
        assert repr(FilterRule.notna("cat")) == "FilterRule.notna('cat')"
        assert repr(FilterRule("cat", "isna", None)) == "FilterRule.isna('cat')"
        assert repr(FilterRule("cat", "notna", None)) == "FilterRule.notna('cat')"

    def test_str_unary(self):
        assert str(FilterRule.isna("cat")) == "cat is missing"
        assert str(FilterRule.notna("cat")) == "cat has a value"
    
    def test_repr_binary(self):
        assert repr(FilterRule("Val", ">=", 20)) == "FilterRule('Val', '>=', 20)"
        assert repr(FilterRule("Cat", "!=", "A")) == "FilterRule('Cat', '!=', 'A')"
        assert repr(FilterRule("Cat", "==", "A")) == "FilterRule('Cat', '==', 'A')"
        assert repr(FilterRule("Cat", "isin", ["A", "B"])) == "FilterRule('Cat', 'isin', ['A', 'B'])"
        assert repr(FilterRule("Cat", "notin", ["A", "B"])) == "FilterRule('Cat', 'notin', ['A', 'B'])"

    def test_str_binary(self):
        assert str(FilterRule("Val", ">=", 20)) == "Val >= 20"
        assert str(FilterRule("Cat", "!=", "A")) == "Cat is not A"
        assert str(FilterRule("Cat", "==", "A")) == "Cat is A"
        assert str(FilterRule("Cat", "isin", ["A", "B"])) == "Cat is in: A, B"
        assert str(FilterRule("Cat", "notin", ["A", "B"])) == "Cat not in: A, B"

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

    def test_corruped_rule(self):
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
