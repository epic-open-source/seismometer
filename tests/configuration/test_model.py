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
        "action, values, range_, count, should_raise, expected_warning",
        [
            # Valid: keep_top with nothing
            ("keep_top", None, None, None, False, None),
            # Valid: keep_top with explicit count
            ("keep_top", None, None, 5, False, None),
            # Invalid count <= 0 → raises
            ("keep_top", None, None, 0, True, None),
            # Valid, hit warning: keep_top with values
            ("keep_top", ["A"], undertest.FilterRange(min=1), None, False, "ignores 'values' and 'range'"),
            # Valid: include with values
            ("include", ["A", "B"], None, None, False, None),
            # Valid: exclude with range
            ("exclude", None, undertest.FilterRange(min=0, max=10), None, False, None),
            # Invalid: include with neither values nor range
            ("include", None, None, None, True, None),
            # Invalid: exclude with neither values nor range
            ("exclude", None, None, None, True, None),
            # Warning: include with both values and range
            ("include", ["A"], undertest.FilterRange(min=0), None, False, "both 'values' and 'range'"),
            # Warning: keep_top with values and range
            ("keep_top", ["A"], undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'"),
            # One of values or range
            ("keep_top", ["A"], None, None, False, "ignores 'values' and 'range'"),
            ("keep_top", None, undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'"),
            # Both values and range
            ("keep_top", ["A"], undertest.FilterRange(min=0), None, False, "ignores 'values' and 'range'"),
            # Valid: keep_top with explicit count
            ("keep_top", None, None, 5, False, None),
            # Edge case: count=0 is invalid
            ("keep_top", None, None, 0, True, None),
            # Invalid: count < 0
            ("keep_top", None, None, -1, True, None),
            # Irrelevant count should be ignored
            ("include", ["A"], None, 10, False, None),
        ],
    )
    def test_filter_validation_behavior(self, caplog, action, values, range_, count, should_raise, expected_warning):
        kwargs = dict(source="some_col", action=action, values=values, range=range_, count=count)
        if should_raise:
            with pytest.raises(ValueError):
                undertest.FilterConfig(**kwargs)
        else:
            with caplog.at_level("WARNING", logger="seismometer"):
                f = undertest.FilterConfig(**kwargs)
                assert f.action == action
                assert f.source == "some_col"
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
