from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

from tqdm.auto import tqdm

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_INSTANCE,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    PASS_TO_PASS,
    RUN_EVALUATION_LOG_DIR,
)
from swebench.harness.docker_build import close_logger, setup_logger
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec
from swebench.harness.utils import EvaluationError, run_threadpool


GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]
E2B_EXIT_SENTINEL = "__SWEBENCH_E2B_EXIT_CODE__="
SWE_BENCH_PRO_SCRIPTS_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main/run_scripts"
)
SWE_BENCH_PRO_SCRIPTS_CACHE_DIR = RUN_EVALUATION_LOG_DIR / "swebench_pro_scripts"
_E2B_TEMPLATE_CACHE_LOCK = threading.Lock()
_E2B_TEMPLATE_READY: set[str] = set()
_E2B_TEMPLATE_BUILD_LOCKS: dict[str, threading.Lock] = {}


def _get_e2b_classes():
    try:
        from e2b import Sandbox, Template
    except ImportError as e:
        msg = (
            "E2B evaluation requires the optional dependency `e2b`. "
            "Install it with `pip install e2b` or `pip install 'swebench[e2b]'`."
        )
        raise ImportError(msg) from e
    return Sandbox, Template


def _get_e2b_api_kwargs(api_key: str | None, request_timeout: float | None) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if api_key:
        opts["api_key"] = api_key
    if request_timeout is not None:
        opts["request_timeout"] = request_timeout
    return opts


def validate_e2b_credentials(api_key: str | None = None) -> None:
    if api_key or os.getenv("E2B_API_KEY"):
        return
    raise ValueError(
        "E2B evaluation requires an API key. Set E2B_API_KEY in the environment "
        "or pass --e2b_api_key."
    )


def get_swebench_docker_image_name(
    instance: dict[str, Any],
    *,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
) -> str:
    """Return the Docker image that should be converted into an E2B template."""
    image_name = instance.get("image_name") or instance.get("docker_image")
    if image_name:
        return image_name

    if dockerhub_tag := instance.get("dockerhub_tag"):
        return f"jefzda/sweap-images:{dockerhub_tag}"

    if instance.get("_dataset_type") == "multi-swe-bench" or _looks_like_multi_swe_bench(instance):
        parts = instance["instance_id"].split("__", maxsplit=1)
        if len(parts) == 2:
            org = parts[0]
            repo_and_number = parts[1]
            last_hyphen_idx = repo_and_number.rfind("-")
            if last_hyphen_idx > 0:
                repo = repo_and_number[:last_hyphen_idx]
                number = repo_and_number[last_hyphen_idx + 1 :]
                return f"mswebench/{org}_m_{repo}:pr-{number}".lower()

    iid = instance["instance_id"].replace("__", "_1776_")
    image = f"sweb.eval.x86_64.{iid}:{instance_image_tag}".lower()
    if namespace:
        image = f"{namespace}/{image}"
    if namespace == "swebench":
        image = "docker.io/" + image
    return image


def get_swebench_e2b_template_name(
    instance: dict[str, Any],
    image_name: str,
    *,
    cpu_count: int = 2,
    memory_mb: int = 1024,
) -> str:
    instance_slug = re.sub(r"[^a-z0-9]+", "-", instance["instance_id"].lower()).strip("-")
    instance_slug = instance_slug[:40].rstrip("-") or "instance"
    image_hash = hashlib.sha1(image_name.encode("utf-8")).hexdigest()[:10]
    resource_suffix = "" if (cpu_count, memory_mb) == (2, 1024) else f"-c{cpu_count}-m{memory_mb}"
    return f"mswea-{instance_slug}-{image_hash}{resource_suffix}"


def _get_e2b_template_build_lock(template_name: str) -> threading.Lock:
    with _E2B_TEMPLATE_CACHE_LOCK:
        return _E2B_TEMPLATE_BUILD_LOCKS.setdefault(template_name, threading.Lock())


def _e2b_template_exists(template_name: str, *, api_key: str | None, request_timeout: float | None) -> bool:
    _, Template = _get_e2b_classes()
    api_kwargs = _get_e2b_api_kwargs(api_key, request_timeout)
    exists = getattr(Template, "exists", None)
    if exists is not None:
        return exists(template_name, **api_kwargs)
    return Template.alias_exists(template_name, **api_kwargs)


def prepare_swebench_e2b_template(
    instance: dict[str, Any],
    image_name: str,
    *,
    template: str | None = None,
    api_key: str | None = None,
    request_timeout: float | None = None,
    cpu_count: int = 2,
    memory_mb: int = 1024,
    logger: Any = None,
) -> tuple[str, str]:
    """Resolve or build the reusable E2B template for an evaluation image."""
    if template:
        return template, "configured"

    template_name = get_swebench_e2b_template_name(
        instance,
        image_name,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    with _E2B_TEMPLATE_CACHE_LOCK:
        if template_name in _E2B_TEMPLATE_READY:
            return template_name, "cached"

    template_lock = _get_e2b_template_build_lock(template_name)
    with template_lock:
        with _E2B_TEMPLATE_CACHE_LOCK:
            if template_name in _E2B_TEMPLATE_READY:
                return template_name, "cached"

        if _e2b_template_exists(template_name, api_key=api_key, request_timeout=request_timeout):
            if logger:
                logger.info(f"Reusing existing E2B template '{template_name}'")
            status = "existing"
        else:
            if logger:
                logger.info(
                    f"Building E2B template '{template_name}' from image '{image_name}' "
                    f"with {cpu_count} CPU(s), {memory_mb} MB memory"
                )
            _, Template = _get_e2b_classes()
            Template.build(
                Template().from_image(image_name),
                name=template_name,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                **_get_e2b_api_kwargs(api_key, request_timeout),
            )
            if logger:
                logger.info(f"Built E2B template '{template_name}'")
            status = "built"

        with _E2B_TEMPLATE_CACHE_LOCK:
            _E2B_TEMPLATE_READY.add(template_name)
        return template_name, status


class E2BSandboxRuntime:
    def __init__(
        self,
        *,
        template: str,
        sandbox_timeout: int,
        api_key: str | None = None,
        request_timeout: float | None = None,
        user: str | None = DOCKER_USER,
        cwd: str = "",
        secure: bool = True,
        allow_internet_access: bool = True,
    ):
        Sandbox, _ = _get_e2b_classes()
        self.api_key = api_key
        self.request_timeout = request_timeout
        self.user = user
        self.cwd = cwd
        self.sandbox = Sandbox.create(
            template=template,
            timeout=sandbox_timeout,
            secure=secure,
            allow_internet_access=allow_internet_access,
            **_get_e2b_api_kwargs(api_key, request_timeout),
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        self.sandbox.files.write(path, content, user=self.user, request_timeout=self.request_timeout)

    def read_file(self, path: str) -> str:
        return self.sandbox.files.read(path, user=self.user, request_timeout=self.request_timeout)

    def exec(self, command: str, *, cwd: str | None = None, timeout: int | None = None) -> tuple[str, int]:
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        wrapped_command = self._wrap_command(command)
        try:
            result = self.sandbox.commands.run(
                wrapped_command,
                cwd=cwd or self.cwd or None,
                user=self.user,
                timeout=timeout,
                request_timeout=self.request_timeout,
                on_stdout=stdout_chunks.append,
                on_stderr=stderr_chunks.append,
            )
            if not stdout_chunks and getattr(result, "stdout", None):
                stdout_chunks.append(result.stdout)
            if not stderr_chunks and getattr(result, "stderr", None):
                stderr_chunks.append(result.stderr)
            stdout = "".join(stdout_chunks)
            output, returncode = self._split_output_and_returncode(stdout)
            return output + "".join(stderr_chunks), returncode
        except Exception as e:
            output = "".join(stdout_chunks) + "".join(stderr_chunks)
            return f"{output}\nE2B command failed: {type(e).__name__}: {e}\n", -1

    def cleanup(self) -> None:
        sandbox = getattr(self, "sandbox", None)
        if sandbox is None:
            return
        self.sandbox = None
        try:
            sandbox.kill(**_get_e2b_api_kwargs(self.api_key, self.request_timeout))
        except Exception:
            pass

    def _wrap_command(self, command: str) -> str:
        inner = "\n".join(
            [
                command,
                "__sweb_e2b_rc=$?",
                f"printf '\\n{E2B_EXIT_SENTINEL}%s\\n' \"$__sweb_e2b_rc\"",
                "exit 0",
            ]
        )
        return shlex.join(["bash", "-lc", inner])

    def _split_output_and_returncode(self, stdout: str) -> tuple[str, int]:
        match = re.search(rf"(?:^|\n){re.escape(E2B_EXIT_SENTINEL)}(?P<code>\d+)\n?\Z", stdout)
        if match is None:
            raise RuntimeError("E2B command output did not contain an exit code sentinel.")
        return stdout[: match.start()], int(match.group("code"))


def run_instances_e2b(
    predictions: dict[str, dict[str, Any]],
    instances: list[dict[str, Any]],
    full_dataset: list[dict[str, Any]],
    *,
    run_id: str,
    timeout: int,
    max_workers: int,
    namespace: str | None = "swebench",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    rewrite_reports: bool = False,
    api_key: str | None = None,
    template: str | None = None,
    sandbox_timeout: int = 3600,
    request_timeout: float | None = None,
    cpu_count: int = 2,
    memory_mb: int = 1024,
    user: str | None = DOCKER_USER,
    swebench_pro_scripts_dir: str | None = None,
    multi_swe_fix_patch_path: str = "/home/fix.patch",
    multi_swe_fix_patch_run_cmd: str = "bash /home/fix-run.sh",
) -> None:
    print(f"Running {len(instances)} instances on E2B...")
    payloads = [
        (
            instance,
            predictions[instance[KEY_INSTANCE_ID]],
            run_id,
            timeout,
            namespace,
            instance_image_tag,
            env_image_tag,
            rewrite_reports,
            api_key,
            template,
            sandbox_timeout,
            request_timeout,
            cpu_count,
            memory_mb,
            user,
            swebench_pro_scripts_dir,
            multi_swe_fix_patch_path,
            multi_swe_fix_patch_run_cmd,
        )
        for instance in instances
    ]

    stats = {"resolved": 0, "unresolved": 0, "error": 0}
    pbar = tqdm(total=len(payloads), desc="E2B evaluation", postfix=stats)
    lock = threading.Lock()

    def run_with_progress(*args):
        result = run_instance_e2b(*args)
        with lock:
            if result["completed"]:
                if result["resolved"]:
                    stats["resolved"] += 1
                else:
                    stats["unresolved"] += 1
            else:
                stats["error"] += 1
            pbar.set_postfix(stats)
            pbar.update()
        return result

    try:
        run_threadpool(run_with_progress, payloads, max_workers)
    finally:
        pbar.close()
    print("All E2B instances run.")


def run_instance_e2b(
    instance: dict[str, Any],
    pred: dict[str, Any],
    run_id: str,
    timeout: int,
    namespace: str | None,
    instance_image_tag: str,
    env_image_tag: str,
    rewrite_reports: bool,
    api_key: str | None,
    template: str | None,
    sandbox_timeout: int,
    request_timeout: float | None,
    cpu_count: int,
    memory_mb: int,
    user: str | None,
    swebench_pro_scripts_dir: str | None,
    multi_swe_fix_patch_path: str,
    multi_swe_fix_patch_run_cmd: str,
) -> dict[str, Any]:
    if instance.get("dockerhub_tag"):
        return _run_swebench_pro_instance_e2b(
            instance,
            pred,
            run_id,
            timeout,
            rewrite_reports,
            api_key,
            template,
            sandbox_timeout,
            request_timeout,
            cpu_count,
            memory_mb,
            user,
            swebench_pro_scripts_dir,
        )
    if instance.get("_dataset_type") == "multi-swe-bench" or _looks_like_multi_swe_bench(instance):
        return _run_multi_swe_bench_instance_e2b(
            instance,
            pred,
            run_id,
            timeout,
            rewrite_reports,
            api_key,
            template,
            sandbox_timeout,
            request_timeout,
            cpu_count,
            memory_mb,
            user,
            namespace,
            instance_image_tag,
            multi_swe_fix_patch_path,
            multi_swe_fix_patch_run_cmd,
        )
    return _run_standard_swebench_instance_e2b(
        instance,
        pred,
        run_id,
        timeout,
        namespace,
        instance_image_tag,
        env_image_tag,
        rewrite_reports,
        api_key,
        template,
        sandbox_timeout,
        request_timeout,
        cpu_count,
        memory_mb,
        user,
    )


def _run_standard_swebench_instance_e2b(
    instance: dict[str, Any],
    pred: dict[str, Any],
    run_id: str,
    timeout: int,
    namespace: str | None,
    instance_image_tag: str,
    env_image_tag: str,
    rewrite_reports: bool,
    api_key: str | None,
    template: str | None,
    sandbox_timeout: int,
    request_timeout: float | None,
    cpu_count: int,
    memory_mb: int,
    user: str | None,
) -> dict[str, Any]:
    test_spec = make_test_spec(
        instance,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    return _run_test_spec_on_e2b(
        test_spec,
        instance,
        pred,
        run_id,
        timeout,
        rewrite_reports,
        api_key,
        template,
        sandbox_timeout,
        request_timeout,
        cpu_count,
        memory_mb,
        user,
    )


def _run_test_spec_on_e2b(
    test_spec: TestSpec,
    instance: dict[str, Any],
    pred: dict[str, Any],
    run_id: str,
    timeout: int,
    rewrite_reports: bool,
    api_key: str | None,
    template: str | None,
    sandbox_timeout: int,
    request_timeout: float | None,
    cpu_count: int,
    memory_mb: int,
    user: str | None,
) -> dict[str, Any]:
    instance_id = test_spec.instance_id
    log_dir = _get_log_dir(pred, run_id, instance_id)
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        return _rewrite_report(test_spec, pred, log_dir)
    if report_path.exists():
        return _completed_from_report(report_path, instance_id)

    logger = _get_logger(instance_id, log_dir)
    runtime: E2BSandboxRuntime | None = None
    report: dict[str, Any] = {}
    eval_completed = False
    try:
        image_name = get_swebench_docker_image_name(instance, namespace=test_spec.namespace, instance_image_tag=test_spec.instance_image_tag)
        template_name, status = prepare_swebench_e2b_template(
            instance,
            image_name,
            template=template,
            api_key=api_key,
            request_timeout=request_timeout,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            logger=logger,
        )
        logger.info(f"Using E2B template '{template_name}' ({status}) for image '{image_name}'")
        runtime = E2BSandboxRuntime(
            template=template_name,
            sandbox_timeout=sandbox_timeout,
            api_key=api_key,
            request_timeout=request_timeout,
            user=user,
            cwd=DOCKER_WORKDIR,
        )

        patch_file = log_dir / "patch.diff"
        patch_file.write_text(pred.get(KEY_PREDICTION) or "")
        runtime.write_file(DOCKER_PATCH, patch_file.read_text())

        applied_patch = False
        apply_output = ""
        for git_apply_cmd in GIT_APPLY_CMDS:
            apply_output, returncode = runtime.exec(f"{git_apply_cmd} {DOCKER_PATCH}", cwd=DOCKER_WORKDIR, timeout=timeout)
            if returncode == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{apply_output}")
                applied_patch = True
                break
            logger.info(f"Failed to apply patch in E2B sandbox: {git_apply_cmd}\n{apply_output}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{apply_output}")
            raise EvaluationError(instance_id, f"{APPLY_PATCH_FAIL}:\n{apply_output}", logger)

        git_diff_output_before, _ = runtime.exec("git -c core.fileMode=false diff", cwd=DOCKER_WORKDIR, timeout=timeout)
        logger.info(f"Git diff before:\n{git_diff_output_before}")

        eval_file = log_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script)
        runtime.write_file("/eval.sh", eval_file.read_text())

        start_time = time.time()
        test_output, returncode = runtime.exec("/bin/bash /eval.sh", cwd=DOCKER_WORKDIR, timeout=timeout)
        total_runtime = time.time() - start_time
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output)
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        logger.info(f"Test output for {instance_id} written to {test_output_path}")
        if returncode == -1:
            raise EvaluationError(instance_id, f"Test timed out or failed to execute after {timeout} seconds.", logger)

        git_diff_output_after, _ = runtime.exec("git -c core.fileMode=false diff", cwd=DOCKER_WORKDIR, timeout=timeout)
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        report_path.write_text(json.dumps(report, indent=4))
        eval_completed = True
    except (EvaluationError, Exception) as e:
        logger.error(f"Error in E2B evaluation for {instance_id}: {e}\n{traceback.format_exc()}")
    finally:
        if runtime is not None:
            runtime.cleanup()
        close_logger(logger)
    return {"completed": eval_completed, "resolved": report.get(instance_id, {}).get("resolved", False)}


def _run_swebench_pro_instance_e2b(
    instance: dict[str, Any],
    pred: dict[str, Any],
    run_id: str,
    timeout: int,
    rewrite_reports: bool,
    api_key: str | None,
    template: str | None,
    sandbox_timeout: int,
    request_timeout: float | None,
    cpu_count: int,
    memory_mb: int,
    user: str | None,
    swebench_pro_scripts_dir: str | None,
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    log_dir = _get_log_dir(pred, run_id, instance_id)
    report_path = log_dir / LOG_REPORT
    if report_path.exists() and not rewrite_reports:
        return _completed_from_report(report_path, instance_id)

    logger = _get_logger(instance_id, log_dir)
    runtime: E2BSandboxRuntime | None = None
    report: dict[str, Any] = {}
    eval_completed = False
    try:
        run_script, parser_script = _resolve_swebench_pro_scripts(
            instance_id,
            scripts_dir=swebench_pro_scripts_dir,
            logger=logger,
        )

        image_name = get_swebench_docker_image_name(instance)
        template_name, status = prepare_swebench_e2b_template(
            instance,
            image_name,
            template=template,
            api_key=api_key,
            request_timeout=request_timeout,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            logger=logger,
        )
        logger.info(f"Using E2B template '{template_name}' ({status}) for image '{image_name}'")
        runtime = E2BSandboxRuntime(
            template=template_name,
            sandbox_timeout=sandbox_timeout,
            api_key=api_key,
            request_timeout=request_timeout,
            user=user,
            cwd="/app",
        )

        workspace = "/workspace"
        patch_text = _strip_binary_hunks(pred.get(KEY_PREDICTION) or "")
        _write_e2b_workspace(runtime, workspace, {
            "patch.diff": patch_text,
            "run_script.sh": run_script.read_text(),
            "parser.py": parser_script.read_text(),
            "entryscript.sh": _make_swebench_pro_entryscript(instance),
        })
        (log_dir / "patch.diff").write_text(patch_text)
        (log_dir / "entryscript.sh").write_text(_make_swebench_pro_entryscript(instance))

        start_time = time.time()
        command_output, returncode = runtime.exec("bash /workspace/entryscript.sh", cwd=workspace, timeout=timeout)
        total_runtime = time.time() - start_time
        logger.info(f"Entryscript return code: {returncode}")
        logger.info(f"Entryscript output:\n{command_output}")
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")

        stdout_log = _read_optional_file(runtime, f"{workspace}/stdout.log")
        stderr_log = _read_optional_file(runtime, f"{workspace}/stderr.log")
        output_json_text = _read_optional_file(runtime, f"{workspace}/output.json")
        (log_dir / "stdout.log").write_text(stdout_log)
        (log_dir / "stderr.log").write_text(stderr_log)
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(stdout_log + stderr_log + command_output)

        output_json = json.loads(output_json_text) if output_json_text.strip() else None
        report = _build_swebench_pro_report(instance, pred, output_json)
        report_path.write_text(json.dumps(report, indent=4))
        eval_completed = output_json is not None
    except Exception as e:
        logger.error(f"Error in SWE-bench Pro E2B evaluation for {instance_id}: {e}\n{traceback.format_exc()}")
    finally:
        if runtime is not None:
            runtime.cleanup()
        close_logger(logger)
    return {"completed": eval_completed, "resolved": report.get(instance_id, {}).get("resolved", False)}


def _run_multi_swe_bench_instance_e2b(
    instance: dict[str, Any],
    pred: dict[str, Any],
    run_id: str,
    timeout: int,
    rewrite_reports: bool,
    api_key: str | None,
    template: str | None,
    sandbox_timeout: int,
    request_timeout: float | None,
    cpu_count: int,
    memory_mb: int,
    user: str | None,
    namespace: str | None,
    instance_image_tag: str,
    fix_patch_path: str,
    fix_patch_run_cmd: str,
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    log_dir = _get_log_dir(pred, run_id, instance_id)
    report_path = log_dir / LOG_REPORT
    if report_path.exists() and not rewrite_reports:
        return _completed_from_report(report_path, instance_id)

    logger = _get_logger(instance_id, log_dir)
    runtime: E2BSandboxRuntime | None = None
    report: dict[str, Any] = {}
    eval_completed = False
    try:
        image_name = get_swebench_docker_image_name(instance, namespace=namespace, instance_image_tag=instance_image_tag)
        template_name, status = prepare_swebench_e2b_template(
            instance,
            image_name,
            template=template,
            api_key=api_key,
            request_timeout=request_timeout,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            logger=logger,
        )
        logger.info(f"Using E2B template '{template_name}' ({status}) for image '{image_name}'")
        runtime = E2BSandboxRuntime(
            template=template_name,
            sandbox_timeout=sandbox_timeout,
            api_key=api_key,
            request_timeout=request_timeout,
            user=user,
            cwd="/home",
        )

        patch_text = pred.get(KEY_PREDICTION) or ""
        runtime.write_file(fix_patch_path, patch_text)
        (log_dir / "patch.diff").write_text(patch_text)

        start_time = time.time()
        test_output, returncode = runtime.exec(fix_patch_run_cmd, cwd="/home", timeout=timeout)
        total_runtime = time.time() - start_time
        test_output_path = log_dir / LOG_TEST_OUTPUT
        test_output_path.write_text(test_output)
        logger.info(f"Test runtime: {total_runtime:_.2f} seconds")
        logger.info(f"Multi-SWE-bench command return code: {returncode}")
        logger.info(f"Test output for {instance_id} written to {test_output_path}")

        report = _build_multi_swe_bench_report(instance, pred, returncode)
        report_path.write_text(json.dumps(report, indent=4))
        eval_completed = returncode != -1
    except Exception as e:
        logger.error(f"Error in Multi-SWE-bench E2B evaluation for {instance_id}: {e}\n{traceback.format_exc()}")
    finally:
        if runtime is not None:
            runtime.cleanup()
        close_logger(logger)
    return {"completed": eval_completed, "resolved": report.get(instance_id, {}).get("resolved", False)}


def _build_swebench_pro_report(instance: dict[str, Any], pred: dict[str, Any], output_json: dict[str, Any] | None) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    passed_tests = set()
    if output_json:
        passed_tests = {x["name"] for x in output_json.get("tests", []) if x.get("status") == "PASSED"}
    f2p = set(_jsonish_list(instance.get("fail_to_pass", instance.get(FAIL_TO_PASS, []))))
    p2p = set(_jsonish_list(instance.get("pass_to_pass", instance.get(PASS_TO_PASS, []))))
    resolved = bool(output_json) and (f2p | p2p) <= passed_tests
    return {
        instance_id: {
            "patch_is_None": pred.get(KEY_PREDICTION) is None,
            "patch_exists": pred.get(KEY_PREDICTION) is not None,
            "patch_successfully_applied": bool(output_json),
            "resolved": resolved,
            "tests_status": {
                FAIL_TO_PASS: {
                    "success": sorted(f2p & passed_tests),
                    "failure": sorted(f2p - passed_tests),
                },
                PASS_TO_PASS: {
                    "success": sorted(p2p & passed_tests),
                    "failure": sorted(p2p - passed_tests),
                },
            },
        }
    }


def _build_multi_swe_bench_report(instance: dict[str, Any], pred: dict[str, Any], returncode: int) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    f2p = set(_test_keys(instance.get("f2p_tests", instance.get(FAIL_TO_PASS, []))))
    p2p = set(_test_keys(instance.get("p2p_tests", instance.get(PASS_TO_PASS, []))))
    resolved = returncode == 0
    return {
        instance_id: {
            "patch_is_None": pred.get(KEY_PREDICTION) is None,
            "patch_exists": pred.get(KEY_PREDICTION) is not None,
            "patch_successfully_applied": returncode != -1,
            "resolved": resolved,
            "tests_status": {
                FAIL_TO_PASS: {
                    "success": sorted(f2p) if resolved else [],
                    "failure": [] if resolved else sorted(f2p),
                },
                PASS_TO_PASS: {
                    "success": sorted(p2p) if resolved else [],
                    "failure": [] if resolved else sorted(p2p),
                },
            },
        }
    }


def _make_swebench_pro_entryscript(instance: dict[str, Any]) -> str:
    before_repo_set_cmd = str(instance.get("before_repo_set_cmd", "")).strip().split("\n")[-1]
    selected_test_files = ",".join(_jsonish_list(instance.get("selected_test_files_to_run", [])))
    base_commit = instance["base_commit"]
    lines = [
        "set -eu",
        "cd /app",
        f"git reset --hard {shlex.quote(base_commit)}",
        f"git checkout {shlex.quote(base_commit)}",
        "git apply -v /workspace/patch.diff",
    ]
    if before_repo_set_cmd:
        lines.append(before_repo_set_cmd)
    lines.extend(
        [
            "set +e",
            f"bash /workspace/run_script.sh {shlex.quote(selected_test_files)} > /workspace/stdout.log 2> /workspace/stderr.log",
            "run_rc=$?",
            "set -e",
            "python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
            "exit $run_rc",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_swebench_pro_scripts(
    instance_id: str,
    *,
    scripts_dir: str | None,
    logger: Any = None,
) -> tuple[Path, Path]:
    if scripts_dir:
        script_dir = Path(scripts_dir) / instance_id
        run_script = script_dir / "run_script.sh"
        parser_script = script_dir / "parser.py"
        if run_script.exists() and parser_script.exists():
            return run_script, parser_script
        raise FileNotFoundError(f"Missing run_script.sh or parser.py under {script_dir}")

    cache_dir = SWE_BENCH_PRO_SCRIPTS_CACHE_DIR / instance_id
    run_script = cache_dir / "run_script.sh"
    parser_script = cache_dir / "parser.py"
    if run_script.exists() and parser_script.exists():
        return run_script, parser_script

    cache_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Downloading SWE-bench Pro run scripts for {instance_id} into {cache_dir}")
    try:
        if not run_script.exists():
            run_script.write_text(_download_swebench_pro_script(instance_id, "run_script.sh"))
        if not parser_script.exists():
            parser_script.write_text(_download_swebench_pro_script(instance_id, "parser.py"))
    except Exception as e:
        raise FileNotFoundError(
            "Could not load SWE-bench Pro scripts. Provide --swebench_pro_scripts_dir "
            "with run_scripts/{instance_id}/run_script.sh and parser.py, or ensure "
            "network access to the official SWE-bench_Pro-os repository."
        ) from e
    return run_script, parser_script


def _download_swebench_pro_script(instance_id: str, script_name: str) -> str:
    quoted_instance_id = quote(instance_id, safe="")
    quoted_script_name = quote(script_name, safe="")
    url = f"{SWE_BENCH_PRO_SCRIPTS_RAW_BASE_URL}/{quoted_instance_id}/{quoted_script_name}"
    try:
        with urlopen(url, timeout=30) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise FileNotFoundError(f"Unexpected HTTP status {status} for {url}")
            return response.read().decode("utf-8")
    except (OSError, URLError) as e:
        raise FileNotFoundError(f"Failed to download {url}: {e}") from e


def _write_e2b_workspace(runtime: E2BSandboxRuntime, workspace: str, files: dict[str, str]) -> None:
    runtime.exec(f"mkdir -p {shlex.quote(workspace)}", timeout=60)
    for name, content in files.items():
        runtime.write_file(f"{workspace}/{name}", content)


def _read_optional_file(runtime: E2BSandboxRuntime, path: str) -> str:
    try:
        return runtime.read_file(path)
    except Exception:
        return ""


def _rewrite_report(test_spec: TestSpec, pred: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    test_output_path = log_dir / LOG_TEST_OUTPUT
    if not test_output_path.exists():
        raise ValueError(f"Test output file {test_output_path} does not exist")
    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=test_output_path,
        include_tests_status=True,
    )
    (log_dir / LOG_REPORT).write_text(json.dumps(report, indent=4))
    return {"completed": True, "resolved": report[test_spec.instance_id]["resolved"]}


def _completed_from_report(report_path: Path, instance_id: str) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    return {"completed": True, "resolved": report[instance_id]["resolved"]}


def _get_log_dir(pred: dict[str, Any], run_id: str, instance_id: str) -> Path:
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    return RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id


def _get_logger(instance_id: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    return setup_logger(instance_id, log_dir / LOG_INSTANCE)


def _looks_like_multi_swe_bench(instance: dict[str, Any]) -> bool:
    return "f2p_tests" in instance or "p2p_tests" in instance or "fix_patch_result" in instance


def _jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            import ast

            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return []


def _test_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    return _jsonish_list(value)


def _strip_binary_hunks(patch: str) -> str:
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)
