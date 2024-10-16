# Seismometer

[![Docs](https://img.shields.io/badge/docs-stable-blue)](https://epic-open-source.github.io/seismometer/)
[![PyPI](https://img.shields.io/pypi/v/seismometer)](https://pypi.org/project/seismometer/)
[![GitHub License](https://img.shields.io/github/license/epic-open-source/seismometer)](https://github.com/epic-open-source/seismometer/blob/main/LICENSE.txt)
![CodeQL](https://github.com/epic-open-source/seismometer/workflows/CodeQL/badge.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/seismometer)

Healthcare organizations are seeing a proliferation of artificial intelligence (AI) features that have the potential to improve patient outcomes and clinician efficiency. These organizations need convenient, effective tools and processes to evaluate AI models’ accuracy and ensure they support equitable care for diverse patient populations.

Standards developed in cooperation among healthcare systems, health IT software developers, third-party experts, and the government help can establish evaluation criteria for AI. However, to be meaningful, the standards need to be implemented correctly and consistently and used to test models on the local patient populations and embedded in the specific workflows that will be used.

**seismometer** is a suite of tools that allows you to evaluate AI model performance using these standardized evaluation criteria to help you make decisions based on your own local data and workflows. You can use it to validate a model’s initial performance and continue to monitor its performance over time. Although it can be used for models in any field, it was designed with a focus on validation for healthcare AI models where local validation requires cross-referencing data about patients (such as demographics, clinical interventions, and patient outcomes) and model performance.

## Features

The package includes templates to analyze model statistical performance, fairness across different cohorts, and the application and impact of interventions on outcomes for commonly used model types within healthcare.

We expect these templates to continually evolve as new validation and analysis techniques and approaches emerge, and seismometer is designed to make it easy to incorporate these improvements.  

## Installation

Install the base package with:

`pip install seismometer`

## Getting Help

For general usage questions, refer to our [User Guide](https://epic-open-source.github.io/seismometer/user_guide/index.html#user-guide).

Report any bugs or enhancement suggestions using our [Issues page](https://github.com/epic-open-source/seismometer/issues).  

If you have questions or feedback, e-mail <OpenSourceContributions-Python@epic.com>.

## Contributing to Seismometer

We welcome contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas.

Refer to the [Contribution Guide](https://epic-open-source.github.io/seismometer/development/index.html) for more details.
