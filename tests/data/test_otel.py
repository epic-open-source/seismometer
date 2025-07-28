import pytest

from seismometer.data.otel import RealExportManager


class TestExportManager:
    def test_unusable_create(self):
        """Creating an export manager with nowhere to export should error."""
        with pytest.raises(Exception):
            # Force use to actually make a new class instance.
            RealExportManager.kill()
            _ = RealExportManager()

    def test_class_activate(self):
        r = RealExportManager(file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.activate_exports()
        assert r.active

    def test_class_deactivate(self):
        r = RealExportManager(file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.deactivate_exports()
        assert not r.active
