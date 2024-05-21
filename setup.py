#!/usr/bin/env python
# This shim enables editable installs
import site
import sys

import setuptools

if __name__ == "__main__":
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]
    setuptools.setup()
