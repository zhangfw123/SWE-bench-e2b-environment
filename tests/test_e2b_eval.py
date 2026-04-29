import json

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness import e2b_eval
from swebench.harness.e2b_eval import (
    _build_multi_swe_bench_report,
    _build_swebench_pro_report,
    _resolve_swebench_pro_scripts,
    get_swebench_docker_image_name,
    validate_e2b_credentials,
)
from swebench.harness.utils import get_predictions_from_file, load_swebench_dataset


def test_get_e2b_image_name_for_swebench_pro():
    instance = {
        "instance_id": "instance_NodeBB__NodeBB-123",
        "dockerhub_tag": "nodebb.nodebb-NodeBB__NodeBB-123",
    }

    assert get_swebench_docker_image_name(instance) == "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-123"


def test_get_e2b_image_name_for_multi_swe_bench():
    instance = {"instance_id": "burntsushi__rip-grep-1294", "_dataset_type": "multi-swe-bench"}

    assert get_swebench_docker_image_name(instance) == "mswebench/burntsushi_m_rip-grep:pr-1294"


def test_swebench_pro_report_from_parser_output():
    instance = {
        "instance_id": "instance_NodeBB__NodeBB-123",
        "fail_to_pass": '["test_a"]',
        "pass_to_pass": '["test_b"]',
    }
    pred = {"model_patch": "diff --git a/a b/a\n"}
    parser_output = {
        "tests": [
            {"name": "test_a", "status": "PASSED"},
            {"name": "test_b", "status": "PASSED"},
        ]
    }

    report = _build_swebench_pro_report(instance, pred, parser_output)

    assert report["instance_NodeBB__NodeBB-123"]["resolved"] is True
    assert report["instance_NodeBB__NodeBB-123"]["tests_status"][FAIL_TO_PASS]["success"] == ["test_a"]
    assert report["instance_NodeBB__NodeBB-123"]["tests_status"][PASS_TO_PASS]["success"] == ["test_b"]


def test_swebench_pro_scripts_are_cached_when_dir_is_omitted(tmp_path, monkeypatch):
    calls = []

    def fake_download(instance_id, script_name):
        calls.append((instance_id, script_name))
        return f"# {script_name}\n"

    monkeypatch.setattr(e2b_eval, "SWE_BENCH_PRO_SCRIPTS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(e2b_eval, "_download_swebench_pro_script", fake_download)

    run_script, parser_script = _resolve_swebench_pro_scripts("instance_a", scripts_dir=None)

    assert run_script == tmp_path / "instance_a" / "run_script.sh"
    assert parser_script == tmp_path / "instance_a" / "parser.py"
    assert run_script.read_text() == "# run_script.sh\n"
    assert parser_script.read_text() == "# parser.py\n"
    assert calls == [("instance_a", "run_script.sh"), ("instance_a", "parser.py")]


def test_validate_e2b_credentials_requires_key(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)

    try:
        validate_e2b_credentials(None)
    except ValueError as e:
        assert "E2B evaluation requires an API key" in str(e)
    else:
        raise AssertionError("Expected ValueError when E2B_API_KEY is missing")

    validate_e2b_credentials("e2b_test")


def test_multi_swe_bench_report_uses_fix_run_exit_code():
    instance = {
        "instance_id": "jqlang__jq-3238",
        "f2p_tests": {"tests/onigtest": {"run": "PASS", "test": "FAIL", "fix": "PASS"}},
        "p2p_tests": {"tests/jqtest": {"run": "PASS", "test": "PASS", "fix": "PASS"}},
    }
    pred = {"model_patch": "diff --git a/a b/a\n"}

    report = _build_multi_swe_bench_report(instance, pred, 0)

    assert report["jqlang__jq-3238"]["resolved"] is True
    assert report["jqlang__jq-3238"]["tests_status"][FAIL_TO_PASS]["success"] == ["tests/onigtest"]


def test_predictions_file_accepts_dict_preds_and_patch_key(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "i1": {
                    "instance_id": "i1",
                    "patch": "diff --git a/a b/a\n",
                    "prefix": "model-a",
                }
            }
        )
    )

    preds = get_predictions_from_file(str(preds_path), "pro", "test")

    assert preds == [
        {
            "instance_id": "i1",
            "patch": "diff --git a/a b/a\n",
            "prefix": "model-a",
            "model_name_or_path": "model-a",
            "model_patch": "diff --git a/a b/a\n",
        }
    ]


def test_predictions_file_accepts_multi_swe_bench_fields(tmp_path):
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text(
        json.dumps(
            {
                "org": "facebook",
                "repo": "zstd",
                "number": 3362,
                "fix_patch": "diff --git a/a b/a\n",
                "model_name": "model-b",
            }
        )
        + "\n"
    )

    preds = get_predictions_from_file(str(preds_path), "multi-swe-bench", "train")

    assert preds == [
        {
            "org": "facebook",
            "repo": "zstd",
            "number": 3362,
            "fix_patch": "diff --git a/a b/a\n",
            "model_name": "model-b",
            "instance_id": "facebook__zstd-3362",
            "model_name_or_path": "model-b",
            "model_patch": "diff --git a/a b/a\n",
        }
    ]


def test_local_multi_swe_bench_dataset_is_marked(tmp_path):
    data_dir = tmp_path / "multi"
    c_dir = data_dir / "c"
    c_dir.mkdir(parents=True)
    (c_dir / "data.jsonl").write_text('{"instance_id":"facebook__zstd-3362"}\n')

    instances = load_swebench_dataset(str(data_dir), "dev")

    assert instances == [{"instance_id": "facebook__zstd-3362", "_dataset_type": "multi-swe-bench"}]
