import logging
from pathlib import Path

from pydantic import BaseModel

from seismometer.core.io import load_yaml

from .model import DataUsage, Event, EventDictionary, Metric, OtherInfo, PredictionDictionary


class ConfigProvider:
    """
    The base configuration provider.

    The configuration provider is a layer between the configuration data model and the consuming class.
    It is responsible for loading the config files, template notebook, and data and streamlining the access across
    multiple files.

    Parameters
    ----------
    config_config : str | Path
        Specifies the path to the primary configuration file with a top level key of "other_info".
        The primary configuration file is largely focused on describing where to find other pieces and includes several
        file names and paths.
    usage_config : Optional[str] | Path, optional
        Specifies the path to the usage configuration file with a top level key of "data_usage", by default None; it
        uses usage_config from the primary config file,
        which specifies details about what kind of data is used and how it should be used.
    info_dir : Optional[str | Path], optional
        Specifies the path to the information *directory*. Not used, by default None;
        Configured in the primary config file
    data_dir : Optional[str | Path], optional
        Specifies the path to the data directory, by default None; it uses data_dir from the primary config file,
        which is where data dictionaries are written/read.
    template_notebook : Optional[object], optional
        Unused.
    definitions : Optional[dict], optional
        A dictionary of definitions to use instead of loading those specified by configuration, by default None.
    output_path : Optional[str | Path], optional
        Specifies the path to the output directory or file, by default None;
        if a directory, the template notebook will be used with the prefix gen.
    """

    def __init__(
        self,
        config_config: str | Path,
        *,
        usage_config: str | Path = None,
        info_dir: str | Path = None,
        data_dir: str | Path = None,
        template_notebook: object = None,
        definitions: dict = None,
        output_path: str | Path = None,
    ):
        self._config: OtherInfo = None
        self._usage: DataUsage = None
        self._output_dir: Path = None
        self._output_notebook: str = ""
        self._event_defs: EventDictionary = None
        self._prediction_defs: PredictionDictionary = None
        self._metrics: dict[str, Metric] = None
        self._metric_groups: dict = None
        self._metric_types: dict = None

        if definitions is not None:
            self._prediction_defs = PredictionDictionary(predictions=definitions.pop("predictions", []))
            self._event_defs = EventDictionary(events=definitions.pop("events", None))

        self._load_config_config(config_config)
        self._resolve_other_paths(usage_config, info_dir, data_dir, output_path)
        self._load_output_as_metric()
        self._load_metrics()

    def _load_config_config(self, config_config: str | Path) -> None:
        """
        Loads the configuration file containing "other_info".
        """
        config_path = Path.cwd() / "data" if config_config is None else Path(config_config)

        if config_path.is_dir():
            self.config_dir: Path = config_path
            self.config_file: str = "config.yml"
        else:
            self.config_dir: Path = config_path.parent
            self.config_file: str = config_path.name

        raw_config = load_yaml(self.config_file, self.config_dir)

        self.config = OtherInfo(**raw_config["other_info"])

    def _resolve_other_paths(
        self,
        usage_config: str | Path = None,
        info_dir: str | Path = None,
        data_dir: str | Path = None,
        output_path: str | Path = None,
    ) -> None:
        """
        Identifies the paths to all other configuration between primary configuration and specified values.
        """
        # Coerce to Path or override with Path
        self.config.usage_config = Path(usage_config) if usage_config is not None else Path(self.config.usage_config)
        self.config.info_dir = Path(info_dir) if info_dir is not None else Path(self.config.info_dir)
        self.config.data_dir = Path(data_dir) if data_dir is not None else Path(self.config.data_dir)
        self.set_output(output_path)

    def _load_metrics(self):
        """Load metrics data defined in configuration."""
        self._metrics = {}
        for metric in self.usage.metrics:
            if metric.source in self._metrics:
                logging.warning(
                    f"{metric.source} is also the source for {self._metrics[metric.source].display_name}. "
                    + "Skipping {metric.display_name}."
                )
                continue
            self._metrics[metric.source] = metric
        self._metric_groups = {}
        self._metric_types = {}
        for metric_source in self._metrics:
            metric = self._metrics[metric_source]
            group_keys = [metric.group_keys] if isinstance(metric.group_keys, str) else metric.group_keys
            for group in group_keys:
                self._metric_groups[group] = self._metric_groups.get(group, []) + [metric.source]
            self._metric_types[metric.type] = self._metric_types.get(metric.type, []) + [metric.source]
        for group in self._metric_groups:
            self._metric_groups[group] = sorted(list(set(self._metric_groups[group])))
        for metric_type in self._metric_types:
            self._metric_types[metric_type] = sorted(list(set(self.metric_types[metric_type])))

    def _load_output_as_metric(self):
        # For backward compatibility, outputs are loaded as Metrics with metric type of "binary classification".
        for output in self.output_list:
            metric = Metric(
                source=output,
                display_name=output,
                type="binary classification",
                group_keys="binary classification scores",
            )
            self.usage.metrics.append(metric)

    # region Config
    @property
    def config(self) -> OtherInfo:
        """
        The configuration definition.

        Usually from config.yml, this is primarily used during initial loading to know
        where other pieces are located.
        """
        return self._config

    @config.setter
    def config(self, config_def: OtherInfo):
        self._config = config_def

        # reset any cached values
        self._template = None

    @property
    def template(self) -> None:
        """The template used for building a model-specific seismograph notebook."""
        raise NotImplementedError("Template building is not implemented")

    @property
    def info_dir(self) -> Path:
        """The directory for output information."""
        return self.config.info_dir

    @property
    def prediction_defs(self) -> PredictionDictionary:
        """The dictionary for data in the prediction frame."""
        if self._prediction_defs is None:
            self._prediction_defs = self._load_definitions(
                self.config.prediction_definition, "predictions", PredictionDictionary
            )
        return self._prediction_defs

    @property
    def event_defs(self) -> EventDictionary:
        """The dictionary for event data."""
        if self._event_defs is None:
            self._event_defs = self._load_definitions(self.config.event_definition, "events", EventDictionary)
        return self._event_defs

    def _load_definitions(self, def_path: Path, def_key: str, data_model: BaseModel) -> dict:
        try:
            raw_defs = load_yaml(def_path, resource_dir=self.config_dir)
        except FileNotFoundError:
            logging.info(f"No dictionary file found at {def_path}. Update config config.")
            raw_defs = None

        if raw_defs is None:
            raw_defs = {def_key: []}
        return data_model(**raw_defs)

    @property
    def usage(self) -> DataUsage:
        """
        The configuration on data usage.

        This is loaded from usage_config and maps the data to how it should be used.
        """
        if self._usage is None:
            self._usage = self._load_usage()
        return self._usage

    def _load_usage(self) -> DataUsage:
        raw_usage = load_yaml(self.config.usage_config, resource_dir=self.config_dir)
        return DataUsage(**raw_usage.pop("data_usage", {}))

    @property
    def data_dir(self) -> Path:
        """The parent directory for data files."""
        return self.config.data_dir

    @property
    def prediction_path(self) -> Path:
        """The path to the prediction data file."""
        return self.config.data_dir / self.config.prediction_path

    @property
    def event_path(self) -> Path:
        """The path to the event data file."""
        return self.config.data_dir / self.config.event_path

    @property
    def metadata_path(self) -> Path:
        """The path to the metadata json."""
        return self.config.data_dir / self.config.metadata_path

    # endregion
    # region Usage
    @property
    def entity_id(self) -> str:
        """Accessor for the entity_id key."""
        return self.usage.entity_id

    @property
    def context_id(self) -> str:
        """Accessor for the context_id key."""
        return self.usage.context_id

    @property
    def entity_keys(self) -> list:
        """List of entity and context ids."""
        return [k for k in [self.entity_id, self.context_id] if k is not None]

    @property
    def metrics(self) -> dict:
        """Collection of metric objects to use during analysis."""
        return self._metrics

    @property
    def metric_groups(self) -> dict:
        """Collection of metrics that are members of each metric group."""
        return self._metric_groups

    @property
    def metric_types(self) -> dict:
        """Collection of the metric type associated with each metric."""
        return self._metric_types

    # region EventTableMap
    @property
    def ev_time(self) -> str:
        """The time column in the event table."""
        return self.usage.event_table.time

    @property
    def ev_type(self) -> str:
        """The type column in the event table."""
        return self.usage.event_table.type

    @property
    def ev_value(self) -> str:
        """The value column in the event table."""
        return self.usage.event_table.value

    # endregion

    @property
    def target(self) -> str:
        """
        The primary target to use during evaluation.

        Configured in usage_data as primary_target.
        """
        return self.usage.primary_target

    @property
    def output(self) -> str:
        """
        The primary output of the model.

        Configured in usage_data as primary_output.
        """
        return self.usage.primary_output

    @property
    def predict_time(self) -> str:
        """
        The time column for predictions.

        Configured in usage_data as predict_time.
        """
        return self.usage.predict_time

    @property
    def output_list(self) -> list:
        """
        The list of all columns to consider as outputs.

        Configured in usage_data as outputs or primary_output.
        """
        if self.output in self.usage.outputs:
            return self.usage.outputs
        return [self.output] + self.usage.outputs

    @property
    def features(self) -> list:
        """
        An explicit list of features to use in analysis.

        Optionally configured in usage_data as features, if None, all columns are used.
        Used to focus analysis on a subset of columns when the full set is large.
        """
        return self.usage.features

    @property
    def cohorts(self) -> dict:
        """
        List of cohort objects to use during analysis.

        Configured in usage_data as cohorts.
        """
        return self.usage.cohorts

    @property
    def comparison_time(self) -> str:
        """The column name of the timestamp to use as reference for comparison across events."""
        return self.usage.comparison_time

    @property
    def events(self) -> dict[str, Event]:
        """
        Dictionary of all event objects indexed by column name.

        Configured in usage_data as events, contains target, outcome, and intervention events with any windowing
        information.
        """
        return {ev.display_name: ev for ev in self.usage.events}

    def event_group(self, usage_group: str) -> dict[str, Event]:
        """
        Returns a dictionary of events indexed by column name and restricted to the specified usage group

        Configured in usage_data as events with usage 'group'.
        """
        return {ev.display_name: ev for ev in self.usage.events if ev.usage == usage_group}

    @property
    def targets(self) -> dict[str, Event]:
        """
        Dictionary of events to use as targets, keyed off of event name.

        Configured in usage_data as events with usage 'target'.
        """
        return self.event_group("target")

    @property
    def outcomes(self) -> dict[str, Event]:
        """
        Dictionary of events to use as outcomes, keyed off of event name.

        Configured in usage_data as events with usage 'outcome'.
        """
        return self.event_group("outcome")

    @property
    def interventions(self) -> dict[str, Event]:
        """
        Dictionary of events to use as interventions, keyed off of event name.

        Configured in usage_data as events with usage 'intervention'.
        """
        return self.event_group("intervention")

    @property
    def prediction_columns(self) -> list:
        """List of all columns referenced in usage configuration."""
        # Use set to remove duplicates
        col_set = set(
            self.entity_keys
            + [self.predict_time]
            + self.features
            + self.output_list
            + [c.source for c in self.cohorts]
            + [m.source for m in self.metrics.values()]
        )
        return sorted(col_set)

    def event_types(self) -> list:
        """List of all event types referenced in usage configuration."""
        if not self.events:
            return self.usage.primary_target

        events = [source for event in self.events for source in event.source]
        return list(set(events))

    @property
    def censor_min_count(self) -> int:
        """
        The minimum count needed for a cohort to be included in analysis.

        Configured in usage_data as censor_min_count.
        """
        return self.usage.censor_min_count

    # endregion
    # region Builder
    def set_output(self, output: Path, nb_prefix: str = ""):
        """
        Resolves the location of the outputs based on config and specified values.

        The output directory has the following precedence:
            - value specified by output argument,
            - value specified by info_dir in configuration,
            - the current working directory.

        Parameters
        ----------
        output : Path
            The path to the output directory.
        nb_prefix : str, optional
            string to prepend to an output notebook, by default ''.
        """
        if output is None:
            output = Path(self.info_dir) or Path.cwd() / "output"
        else:
            output = Path(output)
        output = output.resolve()
        if not output.suffix:  # no suffix implies directory
            self._output_dir = output
        else:  # file specified
            if str(output.parent) == ".":  # No path given, use config
                self._output_dir = self.info_dir or output.parent
            else:
                self._output_dir = output.parent

    @property
    def output_dir(self) -> Path:
        """The directory for output files."""
        if self._output_dir is None:
            self._output_dir = Path.cwd()
        # Make directory on first access
        self._output_dir.mkdir(exist_ok=True, parents=True)
        return Path(self._output_dir)

    @property
    def output_notebook(self):
        """The name of the output notebook."""
        if self._output_notebook is None:
            raise ValueError("No notebook was set")
        return self._output_notebook

    # endregion
