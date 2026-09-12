# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The per-session model-call budget.

Written because the thing it replaces was imaginary. `agent.build.steps` is a real key in opencode's
schema and is simply not honoured -- measured against a fake engine that always asks for one more
tool call, `steps=3`, `maxSteps=3` and no setting at all each produced 61 model calls. So the tests
that matter here are the two that distinguish a real cap from a decorative one: that the (n+1)th call
is never forwarded, and that it never enters the capture graph.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from openenv.core.harness.capture.server import create_app


class _CountingEngine:
    """Stands in for vLLM. Records what it was asked to do, so 'not forwarded' is observable."""

    def __init__(self) -> None:
        self.calls = 0
        self.served_model = "test-model"
        self.param_fixes: dict[str, Any] = {}
        self.capture_level = "text"

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "id": f"c{self.calls}",
            "object": "chat.completion",
            "model": self.served_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"turn {self.calls}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@pytest.fixture
def app_and_engine():
    app = create_app(
        llm_url="http://engine.invalid/v1", model="test-model", capture_level="text"
    )
    engine = _CountingEngine()
    app.state.inference = engine
    # The pool captured the real client at create_app time, so replacing `app.state.inference` alone
    # leaves every request going to `engine.invalid`. Sessions here name no upstream, so they take the
    # pool's default and this is the hook that matters.
    app.state.upstreams._default = (engine, "text")
    return app, engine


def _chat(client: TestClient, session_id: str) -> Any:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {session_id}"},
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_budget_stops_forwarding_at_the_cap(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=3)
    with TestClient(app) as client:
        for _ in range(5):
            assert _chat(client, session.session_id).status_code == 200
    # Five requests, three forwarded. Without the cap the engine would see all five.
    assert engine.calls == 3
    assert session.model_calls == 3


def test_the_capped_turn_is_terminal_and_never_captured(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create(max_model_calls=1)
    with TestClient(app) as client:
        _chat(client, session.session_id)
        over = _chat(client, session.session_id).json()

    # Terminal: this is what actually ends the agent's loop. opencode exits 0 on it.
    assert over["choices"][0]["finish_reason"] == "stop"
    assert not over["choices"][0]["message"].get("tool_calls")
    # Empty, not an explanation: the model did not say anything, so nothing may be attributed to it.
    assert over["choices"][0]["message"]["content"] == ""
    # And it is not in the graph. A synthetic turn in the training data is the failure this guards.
    assert session.graph.stats()["n_turns"] == 1


def test_zero_means_unlimited(app_and_engine):
    app, engine = app_and_engine
    session = app.state.registry.create()
    assert session.max_model_calls == 0
    with TestClient(app) as client:
        for _ in range(6):
            _chat(client, session.session_id)
    assert engine.calls == 6
    assert not session.over_budget


def test_budget_is_per_session_not_per_server(app_and_engine):
    """One deployment serves a capped training run and an uncapped eval run at the same time."""
    app, engine = app_and_engine
    capped = app.state.registry.create(max_model_calls=2)
    uncapped = app.state.registry.create()
    with TestClient(app) as client:
        for _ in range(4):
            _chat(client, capped.session_id)
            _chat(client, uncapped.session_id)
    assert capped.model_calls == 2
    assert uncapped.model_calls == 4
