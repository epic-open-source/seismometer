import seismometer.plot.mpl._lines as undertest


class TestFindThresholds:
    def test_thresholds_increasing_labels(self):
        labels = [0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [0.2, 0.6]
        expected_output = [0, 2]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_decreasing_labels(self):
        labels = [0.9, 0.7, 0.5, 0.3, 0.1]
        thresholds = [0.2, 0.6]
        expected_output = [3, 1]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_ascending_out_of_range(self):
        labels = [0.1, 0.3, 0.5, 0.7, 0.9]
        thresholds = [-0.1, 1.0]
        expected_output = [0, 4]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output

    def test_thresholds_descending_out_of_range(self):
        labels = [0.9, 0.7, 0.5, 0.3, 0.1]
        thresholds = [-0.1, 1.0]
        expected_output = [4, 0]

        actual = undertest._find_thresholds(labels, thresholds)

        assert actual == expected_output
