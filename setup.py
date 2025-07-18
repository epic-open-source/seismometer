#!/usr/bin/env python
# This shim enables editable installs
import site
import sys

import setuptools

if __name__ == "__main__":
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
    setuptools.setup(
        name="runmetrics",
        version="0.1",
        packages=setuptools.find_packages(),
        entry_points={
            "console_scripts": [
                "runmetrics = seismometer.runmetrics:main",
            ],
        },
    )
