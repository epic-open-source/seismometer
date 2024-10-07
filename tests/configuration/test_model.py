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
