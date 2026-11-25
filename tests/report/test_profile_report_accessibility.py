"""
Accessibility checks of the profile reports in a browser: axe-core's WCAG 2.2 A/AA and best-practice rules, and
keyboard use. Needs the accessibility extra, skipped without it, and a browser (``playwright install chromium``).
"""

import sys
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
axe_playwright = pytest.importorskip("axe_playwright_python.sync_playwright")

from seismometer.data.profile import profile_dataframe  # noqa: E402
from seismometer.report.profile_report import ProfileReport  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "data"))
from profile_cases import make_profile_frame  # noqa: E402

AXE_RULES = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]
# axe only checks what is shown, so open every "More details" section and tab panel first
SHOW_ALL = """() => {
  document.querySelectorAll("details").forEach((details) => { details.open = true; });
  document.querySelectorAll(".tabs > .panel").forEach((panel) => { panel.style.display = "block"; });
}"""


@pytest.fixture(scope="module")
def reports():
    df = make_profile_frame()
    single = ProfileReport([profile_dataframe(df, "Profile")], title="Profile")
    left, right = profile_dataframe(df.iloc[:600], "Target=0"), profile_dataframe(df.iloc[600:], "Target=1")
    compare = ProfileReport([left, right], title="Compare", labels=["Target=0", "Target=1"])
    return {"single": single.to_html(), "compare": compare.to_html()}


@pytest.fixture(scope="module")
def browser():
    # a missing browser fails rather than skips, so the CI job can't pass without running the checks
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.mark.parametrize("name", ["single", "compare"])
def test_reports_pass_axe(browser, reports, name):
    page = browser.new_page()
    page.set_content(reports[name])
    page.evaluate(SHOW_ALL)

    results = axe_playwright.Axe().run(page, options={"runOnly": {"type": "tag", "values": AXE_RULES}})

    violations = [(v["id"], v["help"], len(v["nodes"])) for v in results.response["violations"]]
    assert violations == []


def test_tabs_work_from_the_keyboard(browser, reports):
    page = browser.new_page()
    page.set_content(reports["single"])
    for _ in range(50):
        page.keyboard.press("Tab")
        if page.evaluate("() => document.activeElement.matches('.tabs > input')"):
            break
    else:
        pytest.fail("Tab never reaches the report's tabs")

    shown = """() => [...document.querySelector(".tabs").querySelectorAll(":scope > .panel")]
        .map((panel) => getComputedStyle(panel).display !== "none")"""
    assert page.evaluate(shown) == [True, False, False]
    page.keyboard.press("ArrowRight")
    assert page.evaluate(shown) == [False, True, False]
