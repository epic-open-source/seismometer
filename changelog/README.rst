This directory contains changelog `fragments`; small per-fix or per-PR files **ReST**-formatted text that will 
be added to ``CHANGES.rst`` by `towncrier <https://towncrier.readthedocs.io/en/latest/>`_.

The result is a documentation page meant for **users**. With this focus in mind, describe the change in the user 
experience over the internal implementation detail.

Use full sentences, in the past tense, with proper punctuation, examples::

    Added support for displaying a fairness audit visualization.

    Upgraded event merging to be compatible with pandas v2.

Each file should be named ``<ISSUE>.<TYPE>.rst``, where ``<ISSUE>`` is an issue number, and ``<TYPE>`` is one of 
the five towncrier types: feature, bugfix, doc, removal, or misc.

Such as ``1234.bugfix.rst`` or ``2345.doc.rst``

If a pull request fixes an issue, use that number in its file name. If there is no issue, then use the pull 
request instead.

If your change does not deserve a changelog entry, apply the `skip changelog` GitHub label to your pull request.
