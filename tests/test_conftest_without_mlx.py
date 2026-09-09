"""On a box without MLX the suite must not be red at collection (plan §16.12).

Proved with a nested in-process run whose conftest imports the collector body from `runlock`
and says "not installed", the way the window collector's proof is built: this box has MLX and
cannot be made not to, and pytester's subprocess runner forks, which the fork guard refuses.
"""

from __future__ import annotations

pytest_plugins = ["pytester"]


def test_files_that_can_reach_mlx_are_left_uncollected_and_named(pytester) -> None:
    from conftest import TESTS_THAT_CAN_REACH_MLX

    reaching = sorted(TESTS_THAT_CAN_REACH_MLX)[0]
    pytester.makeconftest(
        f"""
        from local_llm_lab.runlock import ignore_when_mlx_absent, report_mlx_absent

        def pytest_ignore_collect(collection_path, config):
            return ignore_when_mlx_absent(
                collection_path.name, ({reaching!r},), config, installed=False
            )

        def pytest_terminal_summary(terminalreporter, exitstatus, config):
            report_mlx_absent(terminalreporter, config)
        """
    )
    pytester.makepyfile(**{reaching[:-3]: "import mlx.core\n\ndef test_x():\n    assert True\n"})
    pytester.makepyfile(test_plain="def test_y():\n    assert True\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*1 test files not collected: MLX is not installed*"])
    result.stdout.fnmatch_lines([f"*not a pass here: {reaching}*"])
