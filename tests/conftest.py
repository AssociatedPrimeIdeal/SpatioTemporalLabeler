"""Release closed Qt windows between tests that do not run app.exec()."""

import sys

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    qtcore = sys.modules.get("PySide6.QtCore")
    if qtcore is not None:
        application = qtcore.QCoreApplication.instance()
        if application is not None:
            application.sendPostedEvents(None, qtcore.QEvent.Type.DeferredDelete)
