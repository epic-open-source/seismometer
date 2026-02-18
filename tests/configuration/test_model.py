import logging

import pytest
from pydantic import ValidationError

import seismometer.configuration.model as undertest


class TestDictionaryItem:
    @pytest.mark.parametrize(
        "key,value", [["name", "myname"], ["display_name", "myname"], ["dtype", None], ["definition", None]]
    )
    def test_default_values(self, key, value):
        dict_item = undertest.DictionaryItem(name="myname")
        assert getattr(dict_item, key) == value

    def test_disp_name_takes_precedence(self):
        dict_item = undertest.DictionaryItem(name="myname", display_name="a display")
        assert dict_item.name == "myname"
        assert dict_item.display_name == "a display"


class TestEventDictionary:
    def test_no_events_ok(self):
        expected = {"events": []}

        actual = undertest.EventDictionary().model_dump()

        assert expected == actual

    def test_minimal_item(self):
        input = {"name": "evA", "display_name": "event_A"}

        expected_event = input.copy()
        expected_event.update({"dtype": None, "definition": None})
        expected = {"events": [expected_event]}

        event_dict = undertest.EventDictionary(events=[input])

        actual = event_dict.model_dump()
        assert expected == actual

    def test_multiple_events(self):
        input_min = {"name": "evA", "display_name": "event_A"}
        input_full = {"name": "evB", "display_name": "event_B", "dtype": "int", "definition": "a definition"}
        inputs = [input_min.copy(), input_full.copy()]

        expected_event = input_min.copy()
        expected_event.update({"dtype": None, "definition": None})
        expected = {"events": [expected_event.copy(), input_full]}

        event_dict = undertest.EventDictionary(events=inputs)

        actual = event_dict.model_dump()
        assert expected == actual

    @pytest.mark.parametrize("search_key,expected_key", [("evA", "filled"), ("evB", "given"), ("evC", "empty")])
    def test_search_returns_item(self, search_key, expected_key):
        inputs = [
            {"name": "evA", "display_name": "event_A"},
            {"name": "evB", "display_name": "event_B", "dtype": "int", "definition": "a definition"},
        ]

        filled = inputs[0].copy()
        filled.update({"dtype": None, "definition": None})
        expected_dict = {
            "filled": undertest.DictionaryItem(**filled),
            "given": undertest.DictionaryItem(**inputs[1]),
            "empty": "MISSING",
        }
        expected = expected_dict[expected_key]

        event_dict = undertest.EventDictionary(events=inputs)
        actual = event_dict.get(search_key, "MISSING")

        assert actual == expected


class TestPredictDictionary:
    def test_no_predictions_ok(self):
        expected = {"predictions": []}

        actual = undertest.PredictionDictionary().model_dump()

        assert expected == actual

    def test_minimal_item(self):
        input = {"name": "prA", "display_name": "feature_A"}

        expected_prediction = input.copy()
        expected_prediction.update({"dtype": None, "definition": None})
        expected = {"predictions": [expected_prediction]}

        prediction_dict = undertest.PredictionDictionary(predictions=[input])

        actual = prediction_dict.model_dump()
        assert expected == actual

    def test_multiple_predictions(self):
        input_min = {"name": "prA", "display_name": "feature_A"}
        input_full = {"name": "prB", "display_name": "feature_B", "dtype": "int", "definition": "a definition"}
        inputs = [input_min.copy(), input_full.copy()]

        expected_prediction = input_min.copy()
        expected_prediction.update({"dtype": None, "definition": None})
        expected = {"predictions": [expected_prediction.copy(), input_full]}

        prediction_dict = undertest.PredictionDictionary(predictions=inputs)

        actual = prediction_dict.model_dump()
        assert expected == actual

    @pytest.mark.parametrize("search_key,expected_key", [("evA", "filled"), ("evB", "given"), ("evC", "empty")])
    def test_search_returns_item(self, search_key, expected_key):
        inputs = [
            {"name": "evA", "display_name": "event_A"},
            {"name": "evB", "display_name": "event_B", "dtype": "int", "definition": "a definition"},
        ]

        filled = inputs[0].copy()
        filled.update({"dtype": None, "definition": None})
        expected_dict = {
            "filled": undertest.DictionaryItem(**filled),
            "given": undertest.DictionaryItem(**inputs[1]),
            "empty": "MISSING",
        }
        expected = expected_dict[expected_key]

        event_dict = undertest.EventDictionary(events=inputs)
        actual = event_dict.get(search_key, "MISSING")

        assert actual == expected


class TestEvent:
    expectation = {
        "source": ["source"],
        "display_name": "source",
        "type": "binary classification",
        "group_keys": "group_undefined",
        "window_hr": None,
        "offset_hr": 0,
        "impute_val": None,
        "usage": None,
        "aggregation_method": "max",
        "merge_strategy": "forward",
    }

    def test_default_values(self):
        cohort = undertest.Event(source="source")
        assert TestEvent.expectation.copy() == cohort.model_dump()

    @pytest.mark.parametrize(
        "input_dict",
        [
            ({"display_name": "display"}),
            ({"window_hr": 1}),
            ({"offset_hr": 1}),
            ({"impute_val": 1}),
            ({"usage": "usage"}),
            ({"aggregation_method": "min"}),
        ],
    )
    def test_one_valid_attribute_change(self, input_dict):
        expected = TestEvent.expectation.copy()
        for k, v in input_dict.items():
            expected[k] = v

        cohort = undertest.Event(source="source", **input_dict)

        assert expected == cohort.model_dump()

    @pytest.mark.parametrize(
        "input_dict",
        [
            ({"display_name": 1}),
            ({"window_hr": "abc"}),
            ({"offset_hr": "abc"}),
            ({"usage": 1}),
            ({"aggregation_method": "middle"}),
            ({"merge_strategy": "center"}),
        ],
    )
    def test_one_invalid_attribute_change(self, input_dict):
        with pytest.raises(ValidationError, match=f".*{list(input_dict)[0]}.*"):
            _ = undertest.Event(source="source", **input_dict)

    @pytest.mark.parametrize("agg_strategy", undertest.AggregationStrategies.__args__)
    def test_supported_agg_strategies_are_allowed(self, agg_strategy):
        expected = TestEvent.expectation.copy()
        expected["aggregation_method"] = agg_strategy
        cohort = undertest.Event(source="source", aggregation_method=agg_strategy)

        assert expected == cohort.model_dump()

    @pytest.mark.parametrize("merge_strategy", undertest.MergeStrategies.__args__)
    def test_supported_merge_strategies_are_allowed(self, merge_strategy):
        expected = TestEvent.expectation.copy()
        expected["merge_strategy"] = merge_strategy
        cohort = undertest.Event(source="source", merge_strategy=merge_strategy)

        assert expected == cohort.model_dump()

    def test_multiple_sources_is_allowed(self):
        source_list = ["source1", "source2", "source3"]
        expected = TestEvent.expectation.copy()
        expected["source"] = source_list
        expected["display_name"] = "display"

        cohort = undertest.Event(source=source_list, display_name="display")

        assert expected == cohort.model_dump()

    def test_multiple_sources_require_display(self):
        source_list = ["source1", "source2", "source3"]
        with pytest.raises(ValidationError, match=".*display_name.*"):
            _ = undertest.Event(source=source_list)


class TestCohort:
    def test_default_values(self):
        expected = {"source": "source", "display_name": "source", "splits": []}
        cohort = undertest.Cohort(source="source")

        assert expected == cohort.model_dump()

    def test_set_displayname(self):
        expected = {"source": "source", "display_name": "display", "splits": []}
        cohort = undertest.Cohort(source="source", display_name="display")

        assert expected == cohort.model_dump()

    def test_allows_splits(self):
        split_list = ["split1", "split2"]
        expected = {"source": "source", "display_name": "source", "splits": split_list}
        cohort = undertest.Cohort(source="source", splits=split_list)

        assert expected == cohort.model_dump()

    def test_strips_other_keys(self):
        expected = {"source": "source", "display_name": "source", "splits": []}
        cohort = undertest.Cohort(source="source", other="other")

        assert expected == cohort.model_dump()


class TestDataUsage:
    @pytest.mark.parametrize(
        "key,value",
        [
            ["entity_id", "Id"],
            ["context_id", None],
            ["primary_output", "Score"],
            ["primary_target", "Target"],
            ["predict_time", "Time"],
            ["comparison_time", "Time"],
            ["outputs", []],
            ["cohorts", []],
            ["features", []],
            ["censor_min_count", 10],
        ],
    )
    def test_default_values(self, key, value):
        data_usage = undertest.DataUsage()
        assert getattr(data_usage, key) == value

    @pytest.mark.parametrize(
        "key,value",
        [
            ["entity_id", "EntityId"],
            ["context_id", "ContextId"],
            ["primary_output", "Output"],
            ["primary_target", "TargetEvent"],
            ["predict_time", "Timestamp"],
            ["comparison_time", "ComparisonTime"],
            ["outputs", ["Output1", "Output2"]],
            ["features", ["Feature1", "Feature2"]],
            ["censor_min_count", 20],
        ],
    )
    def test_custom_values(self, key, value):
        cohorts = [undertest.Cohort(source="age", display_name="Age")]
        events = [undertest.Event(source="event1", display_name="Event 1")]
        data_usage = undertest.DataUsage(
            entity_id="EntityId",
            context_id="ContextId",
            primary_output="Output",
            primary_target="TargetEvent",
            predict_time="Timestamp",
            comparison_time="ComparisonTime",
            event_table=undertest.EventTableMap(type="EventType", time="EventTime", value="EventValue"),
            outputs=["Output1", "Output2"],
            cohorts=cohorts,
            features=["Feature1", "Feature2"],
            events=events,
            censor_min_count=20,
        )
        assert getattr(data_usage, key) == value

    def test_reduce_events_to_unique_names(self, caplog):
        events = [
            undertest.Event(source="event1", display_name="Event 1"),
            undertest.Event(source="event2", display_name="Event 2"),
            undertest.Event(source="different source", display_name="Event 1"),
        ]
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            data_usage = undertest.DataUsage(events=events)
        assert "Duplicate" in caplog.text

        assert len(data_usage.events) == 2
        data_usage.events[0].source == "event1"
        data_usage.events[0].display_name == "Event 1"
        data_usage.events[1].source == "event2"
        data_usage.events[1].display_name == "Event 2"

    def test_reduce_events_eliminates_source_display_collision(self, caplog):
        events = [
            undertest.Event(source="event1"),
            undertest.Event(source="event2", display_name="event1"),
        ]
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            data_usage = undertest.DataUsage(events=events)
        assert "Duplicate" in caplog.text

        assert len(data_usage.events) == 1
        data_usage.events[0].source == "event1"
        data_usage.events[0].display_name == "event1"

    def test_reduce_cohorts_to_unique_names(self, caplog):
        cohorts = [
            undertest.Cohort(source="cohort1", display_name="Cohort 1"),
            undertest.Cohort(source="cohort2", display_name="Cohort 2"),
            undertest.Cohort(source="different source", display_name="Cohort 1"),
        ]
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            data_usage = undertest.DataUsage(cohorts=cohorts)
        assert "Duplicate" in caplog.text

        assert len(data_usage.cohorts) == 2
        data_usage.cohorts[0].source == "cohort1"
        data_usage.cohorts[0].display_name == "Cohort 1"
        data_usage.cohorts[1].source == "cohort2"
        data_usage.cohorts[1].display_name == "Cohort 2"

    def test_reduce_cohorts_eliminates_source_display_collision(self, caplog):
        cohorts = [
            undertest.Cohort(source="cohort1"),
            undertest.Cohort(source="cohort2", display_name="cohort1"),
        ]

        with caplog.at_level(logging.WARNING, logger="seismometer"):
            data_usage = undertest.DataUsage(cohorts=cohorts)
        assert "Duplicate" in caplog.text

        assert len(data_usage.cohorts) == 1
        data_usage.cohorts[0].source == "cohort1"
        data_usage.cohorts[0].display_name == "cohort1"

    def test_filter_defaults_are_none(self):
        filter_range = undertest.FilterRange()
        assert filter_range.min is None
        assert filter_range.max is None

    def test_filter_can_set_bounds(self):
        filter_range = undertest.FilterRange(min=10, max=100)
        assert filter_range.min == 10
        assert filter_range.max == 100


class TestFilterConfig:
    @pytest.mark.parametrize(
        "action, values, range_, count, should_raise, expected_warning, expected_attributes",
        [
            # (action, values, range_, count, should_raise, expected_warning, expected_attributes)
            ("keep_top", None, None, None, False, None, {}),
            ("keep_top", None, None, 5, False, None, {"count": 5}),
            ("keep_top", None, None, 0, True, None, {"count": 0}),
            ("keep_top", ["A"], undertest.FilterRange(min=1), None, False, "ignores 'values' and 'range'", {}),
            ("include", ["A", "B"], None, None, False, None, {"values": ["A", "B"]}),
            (
                "exclude",
                None,
                undertest.FilterRange(min=0, max=10),
                None,
                False,
                None,
                {"range": undertest.FilterRange(min=0, max=10)},
            ),
            ("include", None, None, None, True, None, {}),
            ("exclude", None, None, None, True, None, {}),
            (
                "include",
                ["A"],
                undertest.FilterRange(min=0),
                None,
                False,
                "both 'values' and 'range'",
                {"values": ["A"]},
            ),
            ("keep_top", ["A"], undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'", {}),
            ("keep_top", ["A"], None, None, False, "ignores 'values' and 'range'", {}),
            ("keep_top", None, undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'", {}),
            ("keep_top", ["A"], undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'", {}),
            ("keep_top", None, None, 5, False, None, {"count": 5}),
            ("keep_top", None, None, 0, True, None, {}),
            ("keep_top", None, None, -1, True, None, {}),
            ("include", ["A"], None, 10, False, "Ignoring 'count=10' for filter on 'some_col'", {"values": ["A"]}),
        ],
    )
    def test_filter_validation_behavior(
        self, caplog, action, values, range_, count, should_raise, expected_warning, expected_attributes
    ):
        kwargs = dict(source="some_col", action=action, values=values, range=range_, count=count)
        if should_raise:
            with pytest.raises(ValueError):
                undertest.FilterConfig(**kwargs)
        else:
            with caplog.at_level("WARNING", logger="seismometer"):
                f = undertest.FilterConfig(**kwargs)
                assert f.action == action
                assert f.source == "some_col"
                assert f.values == expected_attributes.get("values")
                assert f.range == expected_attributes.get("range")
                assert f.count == expected_attributes.get("count")
            if expected_warning:
                assert expected_warning in caplog.text
            else:
                assert "WARNING" not in caplog.text

    def test_from_filter_config_uses_count_when_provided(self):
        from seismometer.data.filter import FilterRule

        config = undertest.FilterConfig(source="col", action="keep_top", count=7)
        rule = FilterRule.from_filter_config(config)
        assert rule.relation == "topk"
        assert rule.right == 7

    def test_from_filter_config_uses_class_default(self, monkeypatch):
        from seismometer.data import filter as filter_module
        from seismometer.data.filter import FilterRule

        monkeypatch.setattr(filter_module.FilterRule, "MAXIMUM_NUM_COHORTS", 42)

        config = undertest.FilterConfig(source="col", action="keep_top")
        rule = FilterRule.from_filter_config(config)

        assert rule.right == 42

    @pytest.mark.parametrize(
        "values, range_, count, expected_action",
        [
            (["a", "b"], None, None, "include"),  # inferred include from values
            (None, undertest.FilterRange(min=0), None, "include"),  # inferred include from range
            (None, None, 2, "keep_top"),  # inferred keep_top from count
        ],
    )
    def test_action_is_inferred_when_none(self, values, range_, count, expected_action):
        config = undertest.FilterConfig(source="demo", values=values, range=range_, count=count)
        assert config.action == expected_action

    def test_action_is_required_if_all_other_fields_missing(self):
        with pytest.raises(ValueError, match="must specify one of 'values', 'range', or 'count'"):
            undertest.FilterConfig(source="demo")


class TestCohortHierarchy:
    def test_valid_hierarchy_is_accepted(self):
        h = undertest.CohortHierarchy(name="Demo", column_order=["location", "department"])
        assert h.name == "Demo"
        assert h.column_order == ["location", "department"]

    @pytest.mark.parametrize(
        "column_order,expected_error",
        [
            (["only_one"], "'Invalid' is invalid: 'column_order' must contain at least two distinct column names."),
            (["a", "b", "a"], "'Invalid' is invalid: 'column_order' contains duplicate columns."),
        ],
    )
    def test_invalid_hierarchy_raises(self, column_order, expected_error):
        with pytest.raises(ValueError, match=expected_error):
            undertest.CohortHierarchy(name="Invalid", column_order=column_order)


# ============================================================================
# ADDITIONAL VALIDATOR AND CONSTRAINT TESTS
# ============================================================================


class TestEventCoerceSourceListValidator:
    """Test Event.coerce_source_list validator with edge cases."""

    def test_coerce_single_string_to_list(self):
        """Test that a single string source is coerced to a list."""
        event = undertest.Event(source="event1")
        assert event.source == ["event1"]
        assert isinstance(event.source, list)

    def test_list_source_remains_list(self):
        """Test that a list source remains a list."""
        event = undertest.Event(source=["event1", "event2"], display_name="Combined")
        assert event.source == ["event1", "event2"]
        assert isinstance(event.source, list)

    def test_empty_string_coerced_to_list(self):
        """Test that an empty string is coerced to a single-item list."""
        event = undertest.Event(source="")
        assert event.source == [""]
        assert isinstance(event.source, list)
        assert len(event.source) == 1

    @pytest.mark.parametrize(
        "source_value",
        [
            123,  # integer
            123.45,  # float
            True,  # boolean
            {"key": "value"},  # dict
        ],
    )
    def test_invalid_source_types_raise_error(self, source_value):
        """Test that invalid source types raise ValidationError."""
        with pytest.raises(ValidationError):
            undertest.Event(source=source_value)

    def test_list_with_empty_strings(self):
        """Test list containing empty strings."""
        event = undertest.Event(source=["event1", "", "event2"], display_name="Combined")
        assert event.source == ["event1", "", "event2"]
        assert len(event.source) == 3


class TestDataUsageValidateHierarchiesDisjoint:
    """Test DataUsage.validate_hierarchies_disjoint validator."""

    def test_disjoint_hierarchies_are_valid(self):
        """Test that disjoint hierarchies are accepted."""
        hierarchies = [
            undertest.CohortHierarchy(name="Geo", column_order=["country", "state", "city"]),
            undertest.CohortHierarchy(name="Org", column_order=["department", "team"]),
        ]
        data_usage = undertest.DataUsage(cohort_hierarchies=hierarchies)
        assert len(data_usage.cohort_hierarchies) == 2

    def test_duplicate_columns_across_hierarchies_raise_error(self):
        """Test that duplicate columns across hierarchies raise ValueError."""
        hierarchies = [
            undertest.CohortHierarchy(name="Geo", column_order=["country", "state"]),
            undertest.CohortHierarchy(name="Org", column_order=["state", "department"]),  # 'state' duplicated
        ]
        with pytest.raises(ValueError, match="must be disjoint.*found duplicates.*state"):
            undertest.DataUsage(cohort_hierarchies=hierarchies)

    def test_multiple_duplicate_columns_across_hierarchies(self):
        """Test that multiple duplicate columns are reported."""
        hierarchies = [
            undertest.CohortHierarchy(name="H1", column_order=["col_a", "col_b"]),
            undertest.CohortHierarchy(name="H2", column_order=["col_b", "col_c"]),
            undertest.CohortHierarchy(name="H3", column_order=["col_a", "col_d"]),
        ]
        with pytest.raises(ValueError, match="must be disjoint.*found duplicates"):
            undertest.DataUsage(cohort_hierarchies=hierarchies)

    def test_empty_hierarchies_list_is_valid(self):
        """Test that an empty hierarchies list is valid."""
        data_usage = undertest.DataUsage(cohort_hierarchies=[])
        assert data_usage.cohort_hierarchies == []


class TestMetricDetails:
    """Test MetricDetails class initialization."""

    def test_default_initialization(self):
        """Test MetricDetails with all defaults."""
        details = undertest.MetricDetails()
        assert details.min is None
        assert details.max is None
        assert details.handle_na is None
        assert details.values is None

    def test_initialization_with_min_max(self):
        """Test MetricDetails with min and max values."""
        details = undertest.MetricDetails(min=0, max=100)
        assert details.min == 0
        assert details.max == 100
        assert details.handle_na is None

    def test_initialization_with_float_values(self):
        """Test MetricDetails with float min and max."""
        details = undertest.MetricDetails(min=0.0, max=1.0)
        assert details.min == 0.0
        assert details.max == 1.0

    def test_initialization_with_handle_na(self):
        """Test MetricDetails with handle_na strategy."""
        details = undertest.MetricDetails(handle_na="drop")
        assert details.handle_na == "drop"

    def test_initialization_with_values_list(self):
        """Test MetricDetails with a list of possible values."""
        values_list = [0, 1, 2, 3, 4]
        details = undertest.MetricDetails(values=values_list)
        assert details.values == values_list

    def test_initialization_with_mixed_type_values(self):
        """Test MetricDetails with mixed type values (int, float, str)."""
        values_list = [0, 1.5, "low", "medium", "high"]
        details = undertest.MetricDetails(values=values_list)
        assert details.values == values_list

    def test_initialization_with_all_fields(self):
        """Test MetricDetails with all fields specified."""
        details = undertest.MetricDetails(min=0, max=10, handle_na="impute", values=[0, 5, 10])
        assert details.min == 0
        assert details.max == 10
        assert details.handle_na == "impute"
        assert details.values == [0, 5, 10]


class TestMetricMetricDetailsField:
    """Test Metric.metric_details field validation."""

    def test_metric_with_default_metric_details(self):
        """Test that Metric initializes with default MetricDetails."""
        metric = undertest.Metric(source="score", display_name="Score")
        assert isinstance(metric.metric_details, undertest.MetricDetails)
        assert metric.metric_details.min is None
        assert metric.metric_details.max is None

    def test_metric_with_custom_metric_details(self):
        """Test Metric with custom MetricDetails."""
        details = undertest.MetricDetails(min=0, max=100, handle_na="drop")
        metric = undertest.Metric(source="score", display_name="Score", metric_details=details)
        assert metric.metric_details.min == 0
        assert metric.metric_details.max == 100
        assert metric.metric_details.handle_na == "drop"

    def test_metric_with_inline_metric_details(self):
        """Test Metric with inline MetricDetails dict."""
        metric = undertest.Metric(
            source="score", display_name="Score", metric_details={"min": 0, "max": 1, "values": [0, 0.5, 1]}
        )
        assert metric.metric_details.min == 0
        assert metric.metric_details.max == 1
        assert metric.metric_details.values == [0, 0.5, 1]


class TestCensorMinCountConstraint:
    """Test censor_min_count constraint (ge=10) validation."""

    def test_censor_min_count_default_is_10(self):
        """Test that default censor_min_count is 10."""
        data_usage = undertest.DataUsage()
        assert data_usage.censor_min_count == 10

    def test_censor_min_count_accepts_valid_values(self):
        """Test that values >= 10 are accepted."""
        data_usage = undertest.DataUsage(censor_min_count=10)
        assert data_usage.censor_min_count == 10

        data_usage = undertest.DataUsage(censor_min_count=20)
        assert data_usage.censor_min_count == 20

        data_usage = undertest.DataUsage(censor_min_count=100)
        assert data_usage.censor_min_count == 100

    @pytest.mark.parametrize("invalid_value", [9, 5, 1, 0, -1, -10])
    def test_censor_min_count_rejects_values_below_10(self, invalid_value):
        """Test that values < 10 raise ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 10"):
            undertest.DataUsage(censor_min_count=invalid_value)


class TestCohortSplitsValidation:
    """Test Cohort.splits validation for continuous data."""

    def test_cohort_with_numeric_splits(self):
        """Test Cohort with numeric splits for continuous data."""
        cohort = undertest.Cohort(source="age", splits=[18, 35, 50, 65])
        assert cohort.splits == [18, 35, 50, 65]

    def test_cohort_with_categorical_splits(self):
        """Test Cohort with categorical splits (list of strings)."""
        cohort = undertest.Cohort(source="region", splits=["North", "South", "East", "West"])
        assert cohort.splits == ["North", "South", "East", "West"]

    def test_cohort_with_empty_splits(self):
        """Test Cohort with empty splits list."""
        cohort = undertest.Cohort(source="category")
        assert cohort.splits == []

    def test_cohort_with_mixed_type_splits(self):
        """Test Cohort with mixed type splits (int, float, str)."""
        cohort = undertest.Cohort(source="mixed", splits=[1, 2.5, "high"])
        assert cohort.splits == [1, 2.5, "high"]

    def test_cohort_with_float_splits(self):
        """Test Cohort with float splits for continuous data."""
        cohort = undertest.Cohort(source="score", splits=[0.0, 0.25, 0.5, 0.75, 1.0])
        assert cohort.splits == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestFilterRangeMinMaxValidation:
    """Test FilterRange with min > max."""

    def test_filter_range_with_valid_min_max(self):
        """Test FilterRange with valid min < max."""
        filter_range = undertest.FilterRange(min=0, max=100)
        assert filter_range.min == 0
        assert filter_range.max == 100

    def test_filter_range_with_equal_min_max(self):
        """Test FilterRange with min == max (edge case, allowed by Pydantic)."""
        filter_range = undertest.FilterRange(min=50, max=50)
        assert filter_range.min == 50
        assert filter_range.max == 50

    def test_filter_range_with_min_greater_than_max(self):
        """Test FilterRange with min > max (no validation, allowed by model)."""
        # Note: The model doesn't validate min < max, so this is allowed
        filter_range = undertest.FilterRange(min=100, max=0)
        assert filter_range.min == 100
        assert filter_range.max == 0

    def test_filter_range_with_only_min(self):
        """Test FilterRange with only min specified."""
        filter_range = undertest.FilterRange(min=10)
        assert filter_range.min == 10
        assert filter_range.max is None

    def test_filter_range_with_only_max(self):
        """Test FilterRange with only max specified."""
        filter_range = undertest.FilterRange(max=100)
        assert filter_range.min is None
        assert filter_range.max == 100

    def test_filter_range_with_negative_values(self):
        """Test FilterRange with negative values."""
        filter_range = undertest.FilterRange(min=-100, max=-10)
        assert filter_range.min == -100
        assert filter_range.max == -10

    def test_filter_range_with_float_values(self):
        """Test FilterRange with float values."""
        filter_range = undertest.FilterRange(min=0.5, max=99.5)
        assert filter_range.min == 0.5
        assert filter_range.max == 99.5


class TestEventWindowOffsetCombinations:
    """Test Event.window_hr / offset_hr invalid combinations."""

    def test_event_with_valid_window_and_offset(self):
        """Test Event with valid window_hr and offset_hr."""
        event = undertest.Event(source="event1", window_hr=24, offset_hr=0)
        assert event.window_hr == 24
        assert event.offset_hr == 0

    def test_event_with_none_window_and_zero_offset(self):
        """Test Event with None window_hr (default) and zero offset_hr (default)."""
        event = undertest.Event(source="event1")
        assert event.window_hr is None
        assert event.offset_hr == 0

    def test_event_with_none_window_and_positive_offset(self):
        """Test Event with None window_hr and positive offset_hr."""
        event = undertest.Event(source="event1", window_hr=None, offset_hr=12)
        assert event.window_hr is None
        assert event.offset_hr == 12

    def test_event_with_window_and_negative_offset(self):
        """Test Event with window_hr and negative offset_hr."""
        event = undertest.Event(source="event1", window_hr=48, offset_hr=-24)
        assert event.window_hr == 48
        assert event.offset_hr == -24

    def test_event_with_zero_window(self):
        """Test Event with zero window_hr (edge case)."""
        event = undertest.Event(source="event1", window_hr=0, offset_hr=0)
        assert event.window_hr == 0
        assert event.offset_hr == 0

    def test_event_with_negative_window(self):
        """Test Event with negative window_hr (edge case, no validation prevents this)."""
        # Note: Model doesn't validate window_hr >= 0, so negative values are allowed
        event = undertest.Event(source="event1", window_hr=-10, offset_hr=0)
        assert event.window_hr == -10
        assert event.offset_hr == 0

    def test_event_with_float_window_and_offset(self):
        """Test Event with float window_hr and offset_hr."""
        event = undertest.Event(source="event1", window_hr=24.5, offset_hr=12.25)
        assert event.window_hr == 24.5
        assert event.offset_hr == 12.25

    @pytest.mark.parametrize(
        "window_hr,offset_hr",
        [
            (24, 0),
            (48, -12),
            (None, 6),
            (168, 24),
            (0.5, 0.25),
        ],
    )
    def test_event_various_window_offset_combinations(self, window_hr, offset_hr):
        """Test Event with various valid window_hr and offset_hr combinations."""
        event = undertest.Event(source="event1", window_hr=window_hr, offset_hr=offset_hr)
        assert event.window_hr == window_hr
        assert event.offset_hr == offset_hr
