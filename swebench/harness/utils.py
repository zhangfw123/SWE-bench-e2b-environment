import json
import re
import requests
import traceback
from importlib import resources
import swebench.resources

from argparse import ArgumentTypeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from pathlib import Path
from typing import cast
from swebench.harness.constants import (
    SWEbenchInstance,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)
from unidiff import PatchSet

load_dotenv()


DATASET_NAME_ALIASES = {
    "swe-bench": "SWE-bench/SWE-bench",
    "swebench": "SWE-bench/SWE-bench",
    "swe_bench": "SWE-bench/SWE-bench",
    "full": "SWE-bench/SWE-bench",
    "swe-bench-lite": "SWE-bench/SWE-bench_Lite",
    "swebench-lite": "SWE-bench/SWE-bench_Lite",
    "swe_bench_lite": "SWE-bench/SWE-bench_Lite",
    "swe-bench_lite": "SWE-bench/SWE-bench_Lite",
    "lite": "SWE-bench/SWE-bench_Lite",
    "swe-bench-pro": "ScaleAI/SWE-bench_Pro",
    "swebench-pro": "ScaleAI/SWE-bench_Pro",
    "swe_bench_pro": "ScaleAI/SWE-bench_Pro",
    "pro": "ScaleAI/SWE-bench_Pro",
    "multi-swe-bench": "ByteDance-Seed/Multi-SWE-bench",
    "multi_swe_bench": "ByteDance-Seed/Multi-SWE-bench",
}

MULTI_LANGUAGE_DATASETS = {
    "ByteDance-Seed/Multi-SWE-bench": ["c", "cpp", "go", "java", "js", "python", "rust", "ts"],
}

DEFAULT_DATASET_SPLITS = {
    "ScaleAI/SWE-bench_Pro": "test",
    "ByteDance-Seed/Multi-SWE-bench": "train",
}


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        log_msg = traceback.format_exc()
        self.logger.info(log_msg)
        return (
            f"{self.instance_id}: {super().__str__()}\n"
            f"Check ({self.log_file}) for more information."
        )


def get_predictions_from_file(predictions_path: str, dataset_name: str, split: str):
    if predictions_path == "gold":
        print("Using gold predictions")
        dataset = load_swebench_dataset(dataset_name, split)
        return [
            {
                KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
                KEY_PREDICTION: datum.get("patch", datum.get("fix_patch", "")),
                KEY_MODEL: "gold",
            }
            for datum in dataset
        ]
    if predictions_path.endswith(".json"):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
            if isinstance(predictions, dict):
                predictions = list(
                    predictions.values()
                )  # compatible with SWE-agent predictions
            if not isinstance(predictions, list):
                raise ValueError(
                    "Predictions must be a list[prediction] or a dictionary[instance_id: prediction]"
                )
    elif predictions_path.endswith(".jsonl"):
        with open(predictions_path, "r") as f:
            predictions = [json.loads(line) for line in f]
    else:
        raise ValueError("Predictions path must be .json or .jsonl")

    # Validate that each prediction has an instance_id
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError(f"Each prediction must be a dictionary, got {type(pred)}")
        if KEY_INSTANCE_ID not in pred:
            if all(key in pred for key in ("org", "repo", "number")):
                pred[KEY_INSTANCE_ID] = f"{pred['org']}__{pred['repo']}-{pred['number']}"
            else:
                raise ValueError(f"Each prediction must contain '{KEY_INSTANCE_ID}'")
        if KEY_MODEL not in pred:
            pred[KEY_MODEL] = pred.get("model", pred.get("model_name", pred.get("prefix", "unknown")))
        if KEY_PREDICTION not in pred:
            pred[KEY_PREDICTION] = pred.get("patch", pred.get("fix_patch", pred.get("prediction", "")))

    return predictions


def run_threadpool(func, payloads, max_workers):
    """
    Run a function with a list of payloads using ThreadPoolExecutor.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process
        max_workers: Maximum number of worker threads

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    if max_workers <= 0:
        return run_sequential(func, payloads)
    succeeded, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for running each instance
        futures = {executor.submit(func, *payload): payload for payload in payloads}
        # Wait for each future to complete
        for future in as_completed(futures):
            try:
                # Check if instance ran successfully
                future.result()
                succeeded.append(futures[future])
            except Exception as e:
                print(f"{type(e)}: {e}")
                traceback.print_exc()
                failed.append(futures[future])
    return succeeded, failed


def run_sequential(func, payloads):
    """
    Run a function with a list of payloads sequentially.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    succeeded, failed = [], []
    for payload in payloads:
        try:
            func(*payload)
            succeeded.append(payload)
        except Exception:
            traceback.print_exc()
            failed.append(payload)
    return succeeded, failed


def load_swebench_dataset(
    name="SWE-bench/SWE-bench", split="test", instance_ids=None
) -> list[SWEbenchInstance]:
    """
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    original_name = name
    name = DATASET_NAME_ALIASES.get(name.lower(), name)
    fallback_split = DEFAULT_DATASET_SPLITS.get(name)

    # Load from local .json/.jsonl file
    if name.endswith(".json"):
        dataset = json.loads(Path(name).read_text())
    elif name.endswith(".jsonl"):
        dataset = [json.loads(line) for line in Path(name).read_text().splitlines()]
    elif name.endswith(".parquet"):
        dataset = cast(Dataset, load_dataset("parquet", data_files=name, split="train"))
    elif _is_local_dataset_dict(Path(name)):
        ds = load_from_disk(name)
        resolved_split = _resolve_dataset_split(split, list(ds.keys()), fallback_split=fallback_split, dataset_name=name)
        dataset = ds[resolved_split]
    elif _is_local_multi_language_dataset(Path(name)):
        dataset = _load_local_multi_language_dataset(Path(name))
        name = "ByteDance-Seed/Multi-SWE-bench"
    else:
        # Load from Hugging Face Datasets
        parquet_path = Path(name) / f"{split}.parquet"
        if parquet_path.exists():
            dataset = cast(Dataset, load_dataset("parquet", data_files=str(parquet_path), split="train"))
        elif (Path(name) / split / "dataset_info.json").exists():
            dataset = cast(Dataset, load_from_disk(Path(name) / split))
        elif name in MULTI_LANGUAGE_DATASETS:
            dataset = _load_multi_language_dataset(name)
        else:
            try:
                dataset = cast(Dataset, load_dataset(name, split=split))
            except Exception as e:
                if not fallback_split or split == fallback_split or not _is_missing_split_error(e):
                    raise
                print(f"{original_name} does not provide split {split}; loading {fallback_split} instead.")
                dataset = cast(Dataset, load_dataset(name, split=fallback_split))
    dataset = _mark_dataset_type_if_needed(list(dataset), name)
    dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [
            instance
            for instance in dataset
            if instance[KEY_INSTANCE_ID] in instance_ids
        ]
    return [cast(SWEbenchInstance, instance) for instance in dataset]


def _is_local_dataset_dict(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_dict.json").exists()


def _is_local_multi_language_dataset(path: Path) -> bool:
    return path.is_dir() and any((path / language).is_dir() for languages in MULTI_LANGUAGE_DATASETS.values() for language in languages)


def _resolve_dataset_split(
    requested_split: str,
    available_splits: list[str],
    *,
    fallback_split: str | None,
    dataset_name: str,
) -> str:
    if requested_split in available_splits:
        return requested_split
    if fallback_split and fallback_split in available_splits:
        print(f"{dataset_name} does not provide split {requested_split}; loading {fallback_split} instead.")
        return fallback_split
    if len(available_splits) == 1:
        only_split = available_splits[0]
        print(f"{dataset_name} only provides split {only_split}; loading {only_split} instead of {requested_split}.")
        return only_split
    return requested_split


def _is_missing_split_error(error: Exception) -> bool:
    message = str(error).lower()
    return "unknown split" in message or "bad split" in message or "corresponds to no data" in message


def _load_multi_language_dataset(name: str) -> list[dict]:
    languages = MULTI_LANGUAGE_DATASETS[name]
    all_instances = []
    for language in languages:
        try:
            dataset = load_dataset(name, data_files=f"{language}/*.jsonl", split="train")
            all_instances.extend(list(dataset))
        except Exception as load_error:
            try:
                from huggingface_hub import HfApi, hf_hub_download

                api = HfApi()
                jsonl_files = [
                    file_name
                    for file_name in api.list_repo_files(name, repo_type="dataset")
                    if file_name.startswith(f"{language}/") and file_name.endswith(".jsonl")
                ]
                for jsonl_file in jsonl_files:
                    local_path = hf_hub_download(name, jsonl_file, repo_type="dataset")
                    with Path(local_path).open(encoding="utf-8") as f:
                        all_instances.extend(json.loads(line) for line in f if line.strip())
            except Exception as fallback_error:
                print(
                    f"Failed to load {language} subset from {name}: "
                    f"{type(load_error).__name__}: {load_error}; "
                    f"fallback failed with {type(fallback_error).__name__}: {fallback_error}"
                )
    return all_instances


def _load_local_multi_language_dataset(path: Path) -> list[dict]:
    all_instances = []
    languages = next(iter(MULTI_LANGUAGE_DATASETS.values()))
    for language in languages:
        language_dir = path / language
        if not language_dir.is_dir():
            continue
        for jsonl_file in sorted(language_dir.glob("*.jsonl")):
            with jsonl_file.open(encoding="utf-8") as f:
                all_instances.extend(json.loads(line) for line in f if line.strip())
    return all_instances


def _mark_dataset_type_if_needed(dataset: list[dict], name: str) -> list[dict]:
    if name != "ByteDance-Seed/Multi-SWE-bench":
        return dataset
    marked = []
    for instance in dataset:
        instance_dict = dict(instance)
        instance_dict["_dataset_type"] = "multi-swe-bench"
        marked.append(instance_dict)
    return marked


### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    # should leave one space for empty context lines
    new_lines = [line if line.strip() else " " for line in new_lines]
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta


def extract_minimal_patch(model_patch):
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def has_attribute_or_import_error(log_before):
    """
    Check to see if Attribute/Import-prefix is in log text

    Args:
        log_before (str): Validation log text before patch application
    """
    log_before = log_before.lower()

    if any([x in log_before for x in ["attribute", "import"]]):

        def get_lines_with_word(text, target_word):
            # Function to extract line(s) that contains target_word
            text, target_word = text.lower(), target_word.lower()
            lines, hits = text.split("\n")[::-1], []
            for line in lines:
                if target_word in line:
                    hits.append(line)
            return hits

        # Get line with Attribute/Import error
        lines_1 = get_lines_with_word(log_before, "attribute")
        lines_2 = get_lines_with_word(log_before, "import")
        lines_1 = " ".join(lines_1)
        lines_2 = " ".join(lines_2)

        if any([(x in lines_1 or x in lines_2) for x in ["error", "fail"]]):
            return True
    return False


def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def optional_str(value: str) -> str | None:
    """
    Convert special string values to None, otherwise return the string as-is.
    """
    if value.lower() in ("none", "null", ""):
        return None
    return value


def get_repo_file(repo, commit, filepath):
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except:
        return None


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch (excludes new files).
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    source_files = [x[2:] for x in source_files if x.startswith("a/")]
    return source_files


def get_new_files(patch: str) -> list[str]:
    """
    Get the list of new files in a patch (source is /dev/null).
    """
    new_files = []
    for file in PatchSet(patch):
        if file.source_file == "/dev/null":
            target = file.target_file
            if target.startswith("b/"):
                target = target[2:]
            new_files.append(target)
    return new_files


def ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from text
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


def load_cached_environment_yml(instance_id: str) -> str:
    """
    Load environment.yml from cache
    """
    try:
        repo, number = instance_id.rsplit("-", 1)
    except ValueError:
        return None
    try:
        return (
            resources.files(swebench.resources)
            .joinpath(f"swebench-og/{repo}/{number}/environment.yml")
            .read_text()
        )
    except FileNotFoundError:
        return None
