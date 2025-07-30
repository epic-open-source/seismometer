from seismometer.data.otel import RealExportManager


class TestExportManager:
    def test_class_activate(self):
        r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.activate_exports()
        assert r.active

    def test_class_deactivate(self):
        r = RealExportManager(hostname="", file_output_paths=[], export_ports=[], dump_to_stdout=True)
        r.deactivate_exports()
        assert not r.active
