from typing import Any, Optional, Union

import pandas as pd


class FilterRule(object):
    """
    A rule is used to build a dataframe mask for selecting rows based on column values.

    Rules can be combined using and/or/not logic.
    Rules can be applied to a dataframe to filter rows or create a mask.
    """

    MIN_ROWS: Optional[int] = 10
    left: Union["FilterRule", str]
    relation: str
    right: Any

    method_router = {
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

    def __init__(self, left: Union["FilterRule", str], relation: str, right: Any = None):
        """
        A FilterRule is a relationship that can be reused for filtering data frames.

        Parameters
        ----------
        left : string or FilterRule
            Column name for filtering relatinoships or FilterRule for and/or relationships.
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
            return f"FilterRule('{self.left}', '{self.relation}')"
        if not isinstance(self.right, str):
            return f"FilterRule('{self.left}', '{self.relation}', {self.right})"
        return f"FilterRule('{self.left}', '{self.relation}', '{self.right}')"

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
        df = data[self.mask(data)]
        if not self.MIN_ROWS or len(df) > self.MIN_ROWS:
            return df
        else:
            return df.iloc[0:0]

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
        return FilterRule(left, "or", right)

    def __and__(left, right) -> "FilterRule":
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
            elif self.relation == "and":
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
        FilterRule where the column contains a na value (np.NaN or None).
        """
        return cls(column, "isna")

    @classmethod
    def notna(cls, column) -> "FilterRule":
        """
        FilterRule where the column does not contain a na value (np.NaN or None).
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


def filter_rule_from_cohort_dictionary(cohort=dict[str, tuple[any]]):
    rule = None
    for key in cohort:
        if not cohort[key]:
            continue
        if rule is None:
            rule = FilterRule(key, "isin", cohort[key])
        else:
            rule = rule & FilterRule(key, "isin", cohort[key])
    return rule
