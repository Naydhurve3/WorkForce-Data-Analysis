from wf_analysis.utils.helpers import timer as context_timer, chunk_list
from wf_analysis.utils.logging import setup_logging
from wf_analysis.utils.decorators import log_call, timer, cache_to_disk


class TestHelpers:
    def test_chunk_list_yields_chunks(self):
        chunks = list(chunk_list([1, 2, 3, 4, 5], 2))
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_chunk_list_handles_empty_list(self):
        chunks = list(chunk_list([], 3))
        assert chunks == []

    def test_timer_context_manager_runs(self, capsys):
        with context_timer("test_block"):
            pass
        captured = capsys.readouterr()
        assert "test_block" in captured.out


class TestLogging:
    def test_setup_logging_runs_without_error(self):
        setup_logging(level="DEBUG")


class TestDecorators:
    def test_log_call_decorator(self):
        @log_call
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_timer_decorator(self):
        @timer
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_cache_to_disk_decorator(self, tmp_path):
        call_count = 0

        @cache_to_disk(cache_dir=str(tmp_path))
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3
        assert call_count == 1
