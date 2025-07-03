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
