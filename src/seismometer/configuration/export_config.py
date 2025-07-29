from typing import Any


class ExportConfig:
    otel_files: list[str]
    """Which files we are sending our OTel data to."""
    otel_ports: list[int]
    """Which ports we are exporting our OTel data over."""
    stdout: bool
    """Whether we are dumping our data to standard out."""
    hostname: str
    """Where to send data over ports. By default, otel-collector."""

    def __init__(self, raw_config: dict):
        """Real in all relevant sections of the raw config from config.yml.

        Parameters
        ----------
        raw_config : dict
            The parsed YAML.
        """
        if "log" not in raw_config:
            self.otel_ports = self.otel_files = []
            self.otel_stdout = False
            self.hostname = ""
            return

        log_config = raw_config["log"]

        self.otel_ports = self._parse_to_list(log_config, "ports")
        self.otel_files = self._parse_to_list(log_config, "files")

        if "stdout" in log_config and log_config["stdout"]:
            self.otel_stdout = True
        else:
            self.otel_stdout = False

        if "hostname" in log_config:
            self.hostname = log_config["hostname"]
        else:
            self.hostname = "otel-collector"

    def _parse_to_list(self, config: dict, section_header: str) -> list[Any]:
        """If we are exporting to a list of foos, the YAML will look like:

        foos:
            a
            b
            c

        We want to get the list of foos if it exists, and normalize a single
        `foo` into a list anyway.

        Parameters
        ----------
        config : dict
            The config we are passing in.
        section_header : str
            Which section we are reading.

        Returns
        -------
        list
            The parsed list of objects / export targets.
        """
        if section_header in config:
            section = config[section_header]  # Either a single object, or a list of them
            return section if isinstance(section, list) else [section]
        else:
            return []

    def is_exporting_possible(self) -> bool:
        """Whether there are any export targets

        Returns
        -------
        bool

        """
        return self.otel_stdout or self.otel_files != [] or self.otel_ports != []
