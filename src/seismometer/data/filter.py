from typing import Any, Optional, Union

import pandas as pd


class FilterRule(object):
    """
    A rule is used to build a dataframe mask for selecting rows based on column values.

    Rules can be combined using and/or/not logic.
    Rules can be applied to a dataframe to filter rows or create a mask.
    """

    MIN_ROWS: Optional[int] = 10
    left: Union["FilterRule", str, None]
    relation: str
    right: Any

    method_router = {
        # returns a matching dataframe data, left, right
        "all": lambda x, y, z: pd.Series(True, index=x.index),
        "none": lambda x, y, z: pd.Series(False, index=x.index),
        "isna": lambda x, y, z: x[y].isna(),
        "notna": lambda x, y, z: ~x[y].isna(),
        "isin": lambda x, y, z: x[y].isin(z),
        "notin": lambda x, y, z: ~x[y].isin(z),
        "==": lambda x, y, z: x[y] == z,
        "!=": lambda x, y, z: x[y] != z,
        "<=": lambda x, y, z: x[y] <= z,
        ">=": lambda x, y, z: x[y] >= z,
        "<": lambda x, y, z: x[y] < z,
        ">": lambda x, y, z: x[y] > z,
        "or": lambda x, y, z: (y.mask(x)) | (z.mask(x)),
        "and": lambda x, y, z: (y.mask(x)) & (z.mask(x)),
    }

    inversion = {
        "all": "none",
        "none": "all",
        "isna": "notna",
        "notna": "isna",
        "isin": "notin",
        "notin": "isin",
        "==": "!=",
        "!=": "==",
        "<=": ">",
        ">=": "<",
        "<": ">=",
        ">": "<=",
    }

    def __init__(self, left: Union["FilterRule", str, None], relation: str, right: Any = None):
        """
        A FilterRule is a relationship that can be reused for filtering data frames.

        Parameters
        ----------
        left : string or FilterRule
            Column name for filtering relationships or FilterRule for and/or relationships.
        relation : str
            A relation from FilterRule.method_router.keys().
        right : string or FilterRule
            A value for filtering the column name based on a relation, or a FilterRule for and/or relationships.
        """
        self.left = left
        self.right = right
        if relation not in FilterRule.method_router.keys():
            raise ValueError(f"Relation {relation} not in {FilterRule.method_router.keys()}")

        if relation in ["isin", "notin"] and not isinstance(right, (list, tuple)):
            raise TypeError(f"Containment relation '{relation}' requires list/tuple, right item of type {type(right)}")

        if relation in ["isna", "notna"] and right is not None:
            raise TypeError(
                f"NaN checking relation '{relation}' does not accept right item. Right item is of type {type(right)}"
            )

        if relation in ["all", "none"] and (right is not None or left is not None):
            raise TypeError(f"Universal relation '{relation}' does not accept left/right items")

        if relation in ["and", "or"]:
            if not isinstance(left, FilterRule):
                raise TypeError(
                    f"Relation {relation} only supported between FilterRules, left item of type {type(left)}"
                )
            if not isinstance(right, FilterRule):
                raise TypeError(
                    f"Relation {relation} only supported between FilterRules, right item of type {type(right)}"
                )
        self.relation = relation

    def __repr__(self) -> str:
        """
        String that represents a FilterRule.
        """
        if self.relation in ["all", "none"]:
            return f"FilterRule.{self.relation}()"

        if self.relation in ["and", "or"]:
            assert isinstance(self.left, FilterRule)
            assert isinstance(self.right, FilterRule)
            left = repr(self.left)
            right = repr(self.right)
            if self.left.relation in ["and", "or"]:
                left = f"({left})"
            if self.right.relation in ["and", "or"]:
                right = f"({right})"
            if self.relation == "and":
                return f"{left} & {right}"
            else:  # The "or" case
                return f"{left} | {right}"

        if self.relation in ["isna", "notna"]:
            return f"FilterRule.{self.relation}('{self.left}')"
        if not isinstance(self.right, str):
            return f"FilterRule('{self.left}', '{self.relation}', {self.right})"
        return f"FilterRule('{self.left}', '{self.relation}', '{self.right}')"

    def __str__(self) -> str:
        """
        User readable string that represents a FilterRule.

        >>> rule1 = FilterRule("Val", ">=", 20)
        >>> rule2 = FilterRule("T/F", "==", 0)
        >>> rule3 = FilterRule("Other", "<", 5)
        >>> str(rule1 | (rule2 & rule3))
        'Val >= 20 or (T/F is 0 and Other < 5)'
        """
        match self.relation:
            case "all":
                return "Include all"
            case "none":
                return "Exclude all"
            case "isna":
                return f"{self.left} is missing"
            case "notna":
                return f"{self.left} has a value"
            case "isin":
                return f"{self.left} is in: {', '.join(self.right)}"
            case "notin":
                return f"{self.left} not in: {', '.join(self.right)}"
            case "==":
                return f"{self.left} is {self.right}"
            case "!=":
                return f"{self.left} is not {self.right}"
            case rel if rel in ["<=", "<", ">=", ">"]:
                return f"{self.left} {rel} {self.right}"
            case "and" | "or":
                assert isinstance(self.left, FilterRule)
                assert isinstance(self.right, FilterRule)
                left = str(self.left)
                right = str(self.right)
                if self.left.relation in ["and", "or"]:
                    left = f"({left})"
                if self.right.relation in ["and", "or"]:
                    right = f"({right})"
                if self.relation == "and":
                    return f"{left} and {right}"
                else:  # The "or" case
                    return f"{left} or {right}"
            case _:  # relation is checked in __init__, this should never be reached
                raise ValueError(f"Unknown relation {self.relation}")

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters a dataframe to only the rows matching the FilterRule.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame to filter.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        df = data.loc[self.mask(data)]
        if (not self.MIN_ROWS) or (len(df) > self.MIN_ROWS):
            return df
        else:
            return df[pd.Series(False, index=df.index)]

    def mask(self, data: pd.DataFrame) -> pd.Index:
        """
        Masks the index of a dataframe to only those rows matching the FilterRule.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame whose index is to be masked.

        Returns
        -------
        pd.Index
            Index masked to only the rows that match the FilterRule.
        """
        relation = FilterRule.method_router[self.relation]
        return relation(data, self.left, self.right)

    def __or__(left, right) -> "FilterRule":
        if left == right:
            return left
        if ~left == right:
            return FilterRule.all()

        if left.relation == "all" or right.relation == "all":
            return FilterRule.all()
        if left.relation == "none":
            return right
        if right.relation == "none":
            return left

        return FilterRule(left, "or", right)

    def __and__(left, right) -> "FilterRule":
        if left == right:
            return left
        if ~left == right:
            return FilterRule.none()

        if left.relation == "none" or right.relation == "none":
            return FilterRule.none()
        if left.relation == "all":
            return right
        if right.relation == "all":
            return left

        return FilterRule(left, "and", right)

    def __invert__(self) -> "FilterRule":
        """
        Allows for ~rule to return the negation of a rule.
        Applies De Morgan's laws for conjunctions.
        """
        if self.relation in ["and", "or"]:
            assert isinstance(self.left, FilterRule)
            assert isinstance(self.right, FilterRule)
            if self.relation == "or":
                return FilterRule(~self.left, "and", ~self.right)
            return FilterRule(~self.left, "or", ~self.right)
        return FilterRule(self.left, FilterRule.inversion[self.relation], self.right)

    def __eq__(self, other: object) -> bool:
        """
        Basic check for equality, does not include checks for associativity, distributivity, or commutativity.
        """
        if not isinstance(other, FilterRule):
            return False
        if other.relation != self.relation:
            return False
        if self.left != other.left:
            return False
        if self.right != other.right:
            return False
        return True

    @classmethod
    def isin(cls, column, values) -> "FilterRule":
        """
        FilterRule where the column contains a value in values.
        """
        return cls(column, "isin", values)

    @classmethod
    def notin(cls, column, values) -> "FilterRule":
        """
        FilterRule where the column does not contain any value in values.
        """
        return cls(column, "notin", values)

    @classmethod
    def isna(cls, column) -> "FilterRule":
        """
        FilterRule where the column contains a na value (np.nan or None).
        """
        return cls(column, "isna")

    @classmethod
    def notna(cls, column) -> "FilterRule":
        """
        FilterRule where the column does not contain a na value (np.nan or None).
        """
        return cls(column, "notna")

    @classmethod
    def between(cls, column, lower: Optional[float] = None, upper: Optional[float] = None) -> "FilterRule":
        """
        For ranges, we include the lower range, and exclude the upper range.

        Parameters
        ----------
        column : str
            Name of column to which this rule applies.
        lower : optional float
            Lower range (included).
        upper : optional float
            Upper range (excluded).
        """
        if lower is None and upper is None:
            raise ValueError("Must have either an upper or lower bound.")
        if lower is None:
            return cls(column, "<", upper)
        if upper is None:
            return cls(column, ">=", lower)
        return cls(column, ">=", lower) & cls(column, "<", upper)

    @classmethod
    def lt(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value less than the key value.
        """
        return cls(column, "<", value)

    @classmethod
    def gt(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value greater than the key value.
        """
        return cls(column, ">", value)

    @classmethod
    def leq(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value less than or equal to the key value.
        """
        return cls(column, "<=", value)

    @classmethod
    def geq(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value great than or equal to the key value.
        """
        return cls(column, ">=", value)

    @classmethod
    def eq(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value equal to the key value.
        """
        return cls(column, "==", value)

    @classmethod
    def neq(cls, column, value) -> "FilterRule":
        """
        FilterRule where the column contains a value different from the key value.
        """
        return cls(column, "!=", value)

    @classmethod
    def all(cls):
        """
        FilterRule that selects all rows.
        """
        return cls(None, "all")

    @classmethod
    def none(cls):
        """
        FilterRule that selects no rows.
        """
        return cls(None, "none")

    @classmethod
    def from_cohort_dictionary(cls, cohort_dict: dict[str, tuple[any]] | None = None) -> "FilterRule":
        """
        For a given dictionary, generate a matching FilterRule

        Parameters
        ----------
        cohort_dict : dict[str, tuple[any]], optional
            A dictionary of column names and cohort category labels,
            by default None, in which case FilterRule.all() is returned.

        Returns
        -------
        FilterRule
            A filter rule that verifyes that each column in the keys has a value in the set of selected categories.
        """

        rule = cls.all()
        if not cohort_dict:
            return rule

        for key, value in cohort_dict.items():
            if value:
                rule = rule & cls.isin(key, cohort_dict[key])
        return rule


def filter_rule_from_cohort_dictionary(cohort: dict[str, tuple[any]] | None = None) -> FilterRule:
    """
    For a given dictionary, generate a matching FilterRule

    Parameters
    ----------
    cohort : dict[str, tuple[any]], optional
        A dictionary of column names and cohort category labels,
        by default None, in which case FilterRule.all() is returned.

    Returns
    -------
    FilterRule
        A filter rule that verifyes that each column in the keys has a value in the set of selected categories.
    """
    return FilterRule.from_cohort_dictionary(cohort)
