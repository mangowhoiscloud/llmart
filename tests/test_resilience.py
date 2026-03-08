"""Tests for llmart.resilience."""

from __future__ import annotations

import pytest

from llmart.resilience import llm_retry


class TestLlmRetry:
    def test_success_on_first_attempt(self) -> None:
        call_count = 0

        @llm_retry
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self) -> None:
        call_count = 0

        @llm_retry
        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("transient")
            return "recovered"

        assert fail_twice() == "recovered"
        assert call_count == 3

    def test_raises_after_max_retries(self) -> None:
        @llm_retry
        def always_fail() -> str:
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError, match="permanent"):
            always_fail()

    def test_no_retry_on_non_transient(self) -> None:
        call_count = 0

        @llm_retry
        def value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("not transient")

        with pytest.raises(ValueError, match="not transient"):
            value_error()
        assert call_count == 1
