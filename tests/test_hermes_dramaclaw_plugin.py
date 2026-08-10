from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_plugin_module():
    tools_module = types.ModuleType("tools")
    registry_module = types.ModuleType("tools.registry")
    registry_module.tool_error = lambda value: value
    registry_module.tool_result = lambda value: value
    sys.modules["tools"] = tools_module
    sys.modules["tools.registry"] = registry_module

    path = Path(__file__).resolve().parents[1] / ".hermes" / "plugins" / "dramaclaw" / "__init__.py"
    spec = importlib.util.spec_from_file_location("test_dramaclaw_plugin", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dramaclaw_plugin_adds_chat_error_without_replacing_task_error():
    plugin = _load_plugin_module()
    raw_error = "Content filter triggered. Finish reason: 'content_filter'"

    result = plugin._with_chat_error_hints(
        {
            "ok": True,
            "data": [
                {
                    "status": "failed",
                    "error": raw_error,
                    "metadata": {"provider_response_id": "resp_123"},
                }
            ],
        }
    )

    task = result["data"][0]
    assert task["error"] == raw_error
    assert task["chat_error"] == plugin.TEXT_CONTENT_FILTER_CHAT_ERROR
    assert "Do not quote the raw provider JSON" in task["agent_instruction"]


def test_dramaclaw_plugin_adds_voice_prereq_chat_error():
    plugin = _load_plugin_module()
    raw_error = "Beat 03 解说声线缺失：项目解说人声线未配置，请上传或录制解说人音频"

    result = plugin._with_chat_error_hints(
        {
            "status_code": 200,
            "ok": False,
            "code": "voice_prereq_required",
            "error": raw_error,
        }
    )

    assert result["error"] == raw_error
    assert "配音任务没有成功启动" in result["chat_error"]
    assert "虾塘" in result["chat_error"]
    assert raw_error in result["chat_error"]
    assert "Do not start another tool" in result["agent_instruction"]


def test_dramaclaw_plugin_stops_before_tts_when_pipeline_requires_voice_setup():
    plugin = _load_plugin_module()

    result = plugin._with_chat_error_hints(
        {
            "ok": True,
            "data": {
                "next_step": "voice_setup",
                "next_step_name": "准备配音声线",
                "audio_prerequisites": {
                    "checked": True,
                    "ready": False,
                    "errors": ["Beat 01 解说声线缺失：请上传或录制解说人音频"],
                },
            },
        }
    )

    data = result["data"]
    assert "下一步需要先准备配音声线" in result["chat_notice"]
    assert "虾塘" in result["agent_instruction"]
    assert "Do not claim TTS started" in result["agent_instruction"]
    assert "chat_error" not in data["audio_prerequisites"]


def test_dramaclaw_plugin_adds_render_prereq_chat_error():
    plugin = _load_plugin_module()
    raw_error = (
        "Render 重生未生成可用图片（mode=1x1_2-3, beats=[1, 2, 3]）："
        "Render 模式需要草图但未找到覆盖 beat 1-1 的草图"
    )

    result = plugin._with_chat_error_hints(
        {
            "ok": True,
            "data": [
                {
                    "status": "failed",
                    "error": raw_error,
                }
            ],
        }
    )

    task = result["data"][0]
    assert task["error"] == raw_error
    assert "Render 任务没有生成可用图片" in task["chat_error"]
    assert "虾塘" in task["chat_error"]
    assert raw_error in task["chat_error"]
    assert "Do not start another tool" in task["agent_instruction"]


def test_dramaclaw_plugin_registers_freezone_canvas_tools():
    plugin = _load_plugin_module()

    names = {name for name, _schema, _handler in plugin.TOOLS}

    assert "dramaclaw_list_freezone_skills" in names
    assert "dramaclaw_run_freezone_skill" in names
    assert "dramaclaw_get_freezone_skill_result" in names
    assert "dramaclaw_list_freezone_canvases" in names
    assert "dramaclaw_get_freezone_canvas" in names
    assert "dramaclaw_save_freezone_canvas" in names
    assert "dramaclaw_delete_freezone_canvas" in names
    assert "dramaclaw_create_freezone_canvas_from_preset" in names
    assert "dramaclaw_prepare_system_voices" in names
    assert "dramaclaw_start_video_batch" in names


def test_start_video_batch_starts_up_to_nine_beats(monkeypatch):
    plugin = _load_plugin_module()
    calls = []
    dispatched = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        if path.endswith("/tasks/limits"):
            return {
                "ok": True,
                "data": {"video": {"remaining": 2, "user_remaining": 1}},
            }
        return {"ok": True, "task_id": f"task-{len(calls)}"}

    monkeypatch.setattr(plugin, "_request", fake_request)
    monkeypatch.setattr(
        plugin,
        "_launch_capacity_aware_batch_dispatcher",
        lambda **kwargs: dispatched.append(list(kwargs["pending"])),
    )

    result = plugin._handle_start_video_batch(
        {"project_id": "proj-1", "episode": 2, "beats": [9, 3, 1, 7, 2, 8, 4, 6, 5]}
    )

    assert result["ok"] is True
    assert result["started"] == [1]
    assert result["waiting"] == list(range(2, 10))
    assert dispatched == [list(range(2, 10))]
    post_calls = [call for call in calls if call[0] == "POST"]
    batch_ids = {call[2]["batch_id"] for call in post_calls}
    assert batch_ids == {result["batch_id"]}
    assert all(call[2]["batch_size"] == 9 for call in post_calls)
    assert [call[1] for call in post_calls] == [
        "/api/v1/projects/proj-1/episodes/2/beats/1/video",
    ]


def test_start_video_batch_rejects_more_than_nine_beats():
    plugin = _load_plugin_module()

    result = plugin._handle_start_video_batch(
        {"project_id": "proj-1", "episode": 1, "beats": list(range(1, 11))}
    )

    assert "at most 9 beats" in result


def test_capacity_aware_dispatcher_refills_waiting_items(monkeypatch):
    plugin = _load_plugin_module()
    pending = [4, 5]
    submitted = []
    available = iter([0, 1, 1])

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            self.target()

    monkeypatch.setattr(plugin.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(plugin.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(plugin, "_batch_available_slots", lambda *_args: next(available))

    plugin._launch_capacity_aware_batch_dispatcher(
        batch_id="first-frame-test",
        project="proj-1",
        queue_kind="default",
        pending=pending,
        submit_item=lambda beat: submitted.append(beat) or {"ok": True},
    )

    assert pending == []
    assert submitted == [4, 5]
    assert "first-frame-test" not in plugin._AGENT_BATCH_DISPATCHERS


def test_start_video_batch_auto_fills_short_request_to_nine(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        if path.endswith("/tasks/limits"):
            return {
                "ok": True,
                "data": {"video": {"remaining": 2, "user_remaining": 1}},
            }
        if method == "GET":
            return {
                "ok": True,
                "data": [
                    {
                        "beat_number": beat,
                        "frame_url": f"/frames/{beat}.png",
                        "video_url": "/videos/1.mp4" if beat == 1 else "",
                    }
                    for beat in range(1, 11)
                ],
            }
        return {"ok": True, "task_id": f"task-{len(calls)}"}

    monkeypatch.setattr(plugin, "_request", fake_request)
    monkeypatch.setattr(plugin, "_launch_capacity_aware_batch_dispatcher", lambda **_kwargs: None)

    result = plugin._handle_start_video_batch(
        {"project_id": "proj-1", "episode": 2, "beats": [2, 3, 4]}
    )

    assert result["ok"] is True
    assert result["started"] == [2]
    assert result["waiting"] == list(range(3, 11))
    post_calls = [call for call in calls if call[0] == "POST"]
    assert len(post_calls) == 1
    assert all(call[2]["batch_size"] == 9 for call in post_calls)


def test_start_video_batch_exact_subset_disables_auto_fill(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        if path.endswith("/tasks/limits"):
            return {
                "ok": True,
                "data": {"video": {"remaining": 2, "user_remaining": 1}},
            }
        return {"ok": True, "task_id": f"task-{len(calls)}"}

    monkeypatch.setattr(plugin, "_request", fake_request)
    monkeypatch.setattr(plugin, "_launch_capacity_aware_batch_dispatcher", lambda **_kwargs: None)

    result = plugin._handle_start_video_batch(
        {
            "project_id": "proj-1",
            "episode": 2,
            "beats": [2, 3],
            "auto_fill": False,
        }
    )

    assert result["ok"] is True
    assert result["started"] == [2]
    assert result["waiting"] == [3]
    assert [call[0] for call in calls] == ["GET", "POST"]


def test_get_final_video_displays_all_existing_finals_in_one_tool_call(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path))
        if path.endswith("/episodes"):
            return {"ok": True, "data": [{"number": episode} for episode in range(1, 5)]}
        episode = int(path.split("/")[-2])
        return {
            "ok": True,
            "data": {
                "exists": episode != 4,
                "video_url": f"/static/finals/ep{episode:03d}.mp4" if episode != 4 else "",
            },
        }

    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._handle_get_final_video({"project_id": "proj-1"})

    assert result["ok"] is True
    assert result["count"] == 3
    assert result["episodes"] == [1, 2, 3]
    assert result["has_more"] is False
    root = result["ui_spec"]["elements"]["root"]
    assert len(root["children"]) == 3
    assert calls == [
        ("GET", "/api/v1/projects/proj-1/episodes"),
        ("GET", "/api/v1/projects/proj-1/episodes/1/final"),
        ("GET", "/api/v1/projects/proj-1/episodes/2/final"),
        ("GET", "/api/v1/projects/proj-1/episodes/3/final"),
        ("GET", "/api/v1/projects/proj-1/episodes/4/final"),
    ]


def test_render_first_frames_starts_three_independent_tasks(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        if path.endswith("/tasks/limits"):
            return {
                "ok": True,
                "data": {"default": {"remaining": 3, "user_remaining": 3}},
            }
        return {"ok": True, "task_id": f"task-{len(calls)}"}

    monkeypatch.setattr(plugin, "_request", fake_request)
    monkeypatch.setattr(plugin, "_launch_capacity_aware_batch_dispatcher", lambda **_kwargs: None)

    result = plugin._handle_render_first_frames(
        {"project_id": "proj-1", "episode": 3, "beat_indices": [3, 1, 2]}
    )

    assert result["ok"] is True
    assert result["started"] == [1, 2, 3]
    post_calls = [call for call in calls if call[0] == "POST"]
    assert all(call[1].endswith("/episodes/3/beats/regenerate") for call in post_calls)
    assert [call[2]["beat_indices"] for call in post_calls] == [[1], [2], [3]]
    assert {call[2]["batch_id"] for call in post_calls} == {result["batch_id"]}
    assert all(call[2]["batch_size"] == 3 for call in post_calls)


def test_render_first_frames_omits_completed_beats_and_limits_batch(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        if path.endswith("/tasks/limits"):
            return {
                "ok": True,
                "data": {"default": {"remaining": 3, "user_remaining": 3}},
            }
        if method == "GET":
            return {
                "ok": True,
                "data": [
                    {"beat_number": 1, "frame_url": "/frame-1.png"},
                    {"beat_number": 2, "frame_url": ""},
                    {"beat_number": 3, "frame_url": ""},
                    {"beat_number": 4, "frame_url": ""},
                    *[
                        {"beat_number": beat, "frame_url": ""}
                        for beat in range(5, 12)
                    ],
                ],
            }
        return {"ok": True, "task_id": f"task-{len(calls)}"}

    monkeypatch.setattr(plugin, "_request", fake_request)
    monkeypatch.setattr(plugin, "_launch_capacity_aware_batch_dispatcher", lambda **_kwargs: None)

    result = plugin._handle_render_first_frames({"project_id": "proj-1", "episode": 3})

    assert result["started"] == [2, 3, 4]
    assert result["waiting"] == list(range(5, 11))
    assert result["remaining"] == 1
    assert [call[2]["beat_indices"] for call in calls if call[0] == "POST"] == [
        [2], [3], [4]
    ]


def test_render_first_frames_rejects_more_than_nine_beats():
    plugin = _load_plugin_module()

    result = plugin._handle_render_first_frames(
        {"project_id": "proj-1", "episode": 3, "beat_indices": list(range(1, 11))}
    )

    assert "at most 9 first-frame beats" in result


def test_render_first_frames_does_not_restart_completed_episode(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append((method, path, body))
        return {
            "ok": True,
            "data": [
                {"beat_number": 1, "frame_url": "/frame-1.png"},
                {"beat_number": 2, "frame_url": "/frame-2.png"},
            ],
        }

    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._handle_render_first_frames({"project_id": "proj-1", "episode": 3})

    assert result["code"] == "first_frames_complete"
    assert result["started"] == []
    assert [call[0] for call in calls] == ["GET"]


def test_compose_episode_sends_canonical_body(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_episode_post(args, suffix, *, body=None):
        calls.append((args, suffix, body))
        return {"ok": True, "task_type": "compose_episode", "status": "queued"}

    monkeypatch.setattr(plugin, "_episode_post", fake_episode_post)

    result = plugin._handle_compose_episode({"project_id": "proj-1", "episode": 2})

    assert result["ok"] is True
    assert calls == [
        (
            {"project_id": "proj-1", "episode": 2},
            "videos/compose",
            {"add_subtitles": True, "add_bgm": False},
        )
    ]


def test_prepare_system_voices_tool_requires_explicit_confirmation():
    plugin = _load_plugin_module()

    result = plugin._handle_prepare_system_voices(
        {"episode": 1, "confirmed": False}
    )

    assert result["code"] == "system_voice_confirmation_required"


def test_dramaclaw_run_freezone_skill_uses_typed_endpoint(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append({"method": method, "path": path, "query": query, "body": body})
        return {"ok": True, "data": {"task_type": "freezone_gen"}}

    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._handle_run_freezone_skill(
        {
            "project_id": "demo",
            "skill_id": "freezone.sketch_from_context",
            "skill_node_id": "skill_1",
            "canvas_id": "canvas_a",
            "parameters": {"aspect_ratio": "2:3"},
            "resolved_inputs": [{"role": "beat_context", "node_id": "beat_1"}],
        }
    )

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "path": "/api/v1/projects/demo/freezone/skills/freezone.sketch_from_context/run",
            "query": None,
            "body": {
                "schema_version": "skill.v1",
                "skill_node_id": "skill_1",
                "canvas_id": "canvas_a",
                "idempotency_key": None,
                "parameters": {"aspect_ratio": "2:3"},
                "resolved_inputs": [{"role": "beat_context", "node_id": "beat_1"}],
            },
        }
    ]


def test_dramaclaw_save_freezone_canvas_puts_complete_payload(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append({"method": method, "path": path, "query": query, "body": body})
        return {"ok": True, "saved": True}

    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._handle_save_freezone_canvas(
        {
            "project_id": "demo",
            "canvas_id": "canvas_a",
            "payload": {
                "nodes": [],
                "edges": [],
                "base_revision": 3,
                "client_save_id": "save-1",
            },
        }
    )

    assert result["saved"] is True
    assert calls == [
        {
            "method": "PUT",
            "path": "/api/v1/projects/demo/freezone/canvases/canvas_a",
            "query": None,
            "body": {
                "nodes": [],
                "edges": [],
                "base_revision": 3,
                "client_save_id": "save-1",
                "canvas_id": "canvas_a",
                "project_id": "demo",
            },
        }
    ]


def test_dramaclaw_create_freezone_canvas_from_preset_uses_preset_endpoint(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append({"method": method, "path": path, "query": query, "body": body})
        return {"ok": True, "data": {"canvas_id": "beat-1"}}

    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._handle_create_freezone_canvas_from_preset(
        {
            "project_id": "demo",
            "scope": "beat",
            "episode": 1,
            "beat": 2,
            "primary_slot": "sketch",
        }
    )

    assert result["ok"] is True
    assert calls == [
        {
            "method": "POST",
            "path": "/api/v1/projects/demo/freezone/canvases:from-preset",
            "query": None,
            "body": {
                "scope": "beat",
                "episode": 1,
                "beat": 2,
                "primary_slot": "sketch",
            },
        }
    ]


def test_dramaclaw_freezone_mode_denies_mainline_writes(monkeypatch):
    plugin = _load_plugin_module()

    def fail_request(*_args, **_kwargs):
        raise AssertionError("_request should not be called")

    monkeypatch.setenv("DRAMACLAW_TOOL_MODE", "freezone_canvas")
    monkeypatch.setattr(plugin, "_request", fail_request)
    denied = plugin._guard_freezone_mainline_write(
        "dramaclaw_generate_sketches",
        plugin._handle_generate_sketches,
    )({"project_id": "demo", "episode": 1})

    assert denied["ok"] is False
    assert denied["code"] == "freezone_mainline_write_denied"
    assert "虾画画布" in denied["chat_error"]


def test_dramaclaw_freezone_mode_allows_canvas_writes(monkeypatch):
    plugin = _load_plugin_module()
    calls = []

    def fake_request(method, path, *, query=None, body=None):
        calls.append({"method": method, "path": path, "query": query, "body": body})
        return {"ok": True}

    monkeypatch.setenv("DRAMACLAW_TOOL_MODE", "freezone_canvas")
    monkeypatch.setattr(plugin, "_request", fake_request)

    result = plugin._guard_freezone_mainline_write(
        "dramaclaw_save_freezone_canvas",
        plugin._handle_save_freezone_canvas,
    )(
        {
            "project_id": "demo",
            "canvas_id": "canvas_a",
            "payload": {"nodes": [], "edges": []},
        }
    )

    assert result["ok"] is True
    assert calls[0]["method"] == "PUT"
    assert calls[0]["path"] == "/api/v1/projects/demo/freezone/canvases/canvas_a"
