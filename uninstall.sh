#!/usr/bin/env bash
set -euo pipefail

readonly launcher_marker="# Managed by LowParLLMCapacityIncV0 installer."
readonly home_dir="${HOME:?HOME must be set}"
purge_data=false
if (($# > 1)); then
    printf 'Usage: ./uninstall.sh [--purge-data]\n' >&2
    exit 2
fi
if (($# == 1)); then
    if [[ "$1" != "--purge-data" ]]; then
        printf 'Usage: ./uninstall.sh [--purge-data]\n' >&2
        exit 2
    fi
    purge_data=true
fi
data_home="${XDG_DATA_HOME:-$home_dir/.local/share}"
if [[ "$data_home" != /* ]]; then
    data_home="$home_dir/.local/share"
fi
config_home="${XDG_CONFIG_HOME:-$home_dir/.config}"
if [[ "$config_home" != /* ]]; then
    config_home="$home_dir/.config"
fi
cache_home="${XDG_CACHE_HOME:-$home_dir/.cache}"
if [[ "$cache_home" != /* ]]; then
    cache_home="$home_dir/.cache"
fi
readonly venv_dir="${data_home%/}/lowp/venv"
readonly marker_file="$venv_dir/.lowp-installed"
readonly launcher="$home_dir/.local/bin/lowp"
readonly config_dir="${config_home%/}/lowp"
readonly user_data_dir="${data_home%/}/lowp"
readonly cache_dir="${cache_home%/}/lowp"
readonly training_jobs_dir="$user_data_dir/training/jobs"
readonly searxng_dir="${data_home%/}/lowp/searxng"
readonly searxng_marker="$searxng_dir/.lowp-searxng-managed"
readonly searxng_name="lowp-searxng"
readonly searxng_backup_name="${searxng_name}-previous"

runtime_owned=false
launcher_owned=false
searxng_config_owned=false
searxng_container_owned=false
searxng_runtime=""
searxng_backup_owned=false
searxng_backup_runtime=""
searxng_cleanup_safe=true

for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" container inspect "$searxng_name" >/dev/null 2>&1; then
        label="$("$candidate" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' \
            "$searxng_name" 2>/dev/null || true)"
        if [[ "$label" == "true" ]]; then
            searxng_container_owned=true
            searxng_runtime="$candidate"
            break
        fi
    fi
done
for candidate in docker podman; do
    if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" container inspect "$searxng_backup_name" >/dev/null 2>&1; then
        label="$("$candidate" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' \
            "$searxng_backup_name" 2>/dev/null || true)"
        if [[ "$label" == "true" ]]; then
            searxng_backup_owned=true
            searxng_backup_runtime="$candidate"
            break
        fi
    fi
done
if [[ -d "$searxng_dir" && ! -L "$searxng_dir" \
    && -f "$searxng_marker" && ! -L "$searxng_marker" ]] \
    && grep -Fqx -- "$launcher_marker" "$searxng_marker"; then
    searxng_config_owned=true
fi

if [[ -d "$venv_dir" && ! -L "$venv_dir" && -f "$marker_file" && ! -L "$marker_file" ]] \
    && grep -Fqx -- "$launcher_marker" "$marker_file"; then
    runtime_owned=true
fi
if [[ -f "$launcher" && ! -L "$launcher" ]]; then
    if grep -Fqx -- "$launcher_marker" "$launcher"; then
        launcher_owned=true
    fi
fi

active_training_jobs=""
if [[ "$runtime_owned" == true && -d "$training_jobs_dir" ]]; then
    if [[ -L "$training_jobs_dir" ]]; then
        printf 'Refusing to inspect training processes through a symbolic-link jobs directory.\n' >&2
        exit 1
    fi
    if [[ ! -x "$venv_dir/bin/python" ]]; then
        printf 'Training job records exist, but the LowP Python runtime is unavailable; stop jobs manually before uninstalling.\n' >&2
        exit 1
    fi
    if ! active_training_jobs="$("$venv_dir/bin/python" - "$training_jobs_dir" <<'PY'
import json
from pathlib import Path
import re
import sys

jobs_dir = Path(sys.argv[1])
if jobs_dir.is_symlink() or not jobs_dir.is_dir():
    raise SystemExit("Training jobs directory is not a safe directory.")
for job_dir in sorted(jobs_dir.iterdir()):
    if not re.fullmatch(r"[a-f0-9]{12}", job_dir.name) or job_dir.is_symlink():
        continue
    status_path = job_dir / "status.json"
    if status_path.is_symlink() or not status_path.is_file():
        continue
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not inspect training status {status_path}: {exc}")
    if not isinstance(status, dict):
        raise SystemExit(f"Training status is not a JSON object: {status_path}")
    pid = status.get("pid")
    if status.get("state") not in {"starting", "running", "stopping"}:
        continue
    if not isinstance(pid, int) or pid <= 1:
        continue
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        continue
    if b"lowparllm.ai.lora.worker" in command and job_dir.name.encode() in command:
        print(job_dir.name)
PY
    )"; then
        printf 'Could not safely inspect LowP training workers; no installation files were removed.\n' >&2
        exit 1
    fi
fi

if [[ "$runtime_owned" == false && "$launcher_owned" == false \
    && "$searxng_config_owned" == false && "$searxng_container_owned" == false \
    && "$searxng_backup_owned" == false && "$purge_data" == false ]]; then
    printf 'No LowParLLM-owned installation files were found. Nothing was removed.\n'
    exit 0
fi

printf 'The following LowParLLM-owned installation files will be removed:\n'
if [[ "$runtime_owned" == false && "$launcher_owned" == false ]]; then
    printf '  No application runtime or launcher was found.\n'
fi
if [[ "$runtime_owned" == true ]]; then
    printf '  Application virtual environment: %s\n' "$venv_dir"
fi
if [[ "$launcher_owned" == true ]]; then
    printf '  Launcher: %s\n' "$launcher"
fi
if [[ -n "$active_training_jobs" ]]; then
    printf '  Verified active LowP training jobs to stop before uninstall: %s\n' \
        "${active_training_jobs//$'\n'/, }"
fi
printf 'User configuration, application data, the project repository, and Ollama will be preserved.\n'
read -r -p "Continue? [y/N] " answer || answer=""
if [[ ! "$answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
    printf 'Uninstallation cancelled. Nothing was removed.\n'
    exit 0
fi

if [[ -n "$active_training_jobs" ]]; then
    if ! "$venv_dir/bin/python" - "$active_training_jobs" <<'PY'
import sys
import time

from lowparllm.ai.lora.training import TrainingManager

manager = TrainingManager()
job_ids = tuple(item for item in sys.argv[1].splitlines() if item)
for job_id in job_ids:
    manager.stop(job_id)
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    states = {
        job["job_id"]: job["state"]
        for job_id in job_ids
        for job in manager.status(job_id)
    }
    if len(states) != len(job_ids):
        raise SystemExit("Training job status disappeared while stopping workers.")
    if all(state not in {"starting", "running", "stopping"} for state in states.values()):
        break
    time.sleep(0.25)
else:
    raise SystemExit("A verified LowP training worker did not stop within 15 seconds.")
PY
    then
        printf 'LowP training workers were not stopped; the application runtime was preserved.\n' >&2
        exit 1
    fi
    printf 'Verified LowP training workers stopped; training data and outputs were preserved.\n'
fi

if [[ "$runtime_owned" == true ]]; then
    rm -r -- "$venv_dir" \
        || { printf 'Could not remove the verified LowP runtime: %s\n' "$venv_dir" >&2; exit 1; }
fi
if [[ "$launcher_owned" == true ]]; then
    rm -- "$launcher" \
        || { printf 'Could not remove the verified LowP launcher: %s\n' "$launcher" >&2; exit 1; }
fi
printf 'LowParLLM application files were removed. User data and configuration were preserved.\n'

if [[ "$searxng_config_owned" == true || "$searxng_container_owned" == true ]]; then
    printf '\nLowP-owned SearXNG resources are separate from the application files.\n'
    if [[ "$searxng_container_owned" == true ]]; then
        printf '  Container: %s (%s)\n' "$searxng_name" "$searxng_runtime"
    fi

    if [[ "$searxng_backup_owned" == true ]]; then
        printf '  Previous container: %s (%s)\n' "$searxng_backup_name" "$searxng_backup_runtime"
    fi
    if [[ "$searxng_config_owned" == true ]]; then
        printf '  Settings directory: %s\n' "$searxng_dir"
    fi
    read -r -p "Also remove these verified LowP-owned SearXNG resources? [y/N] " searxng_answer || searxng_answer=""
    if [[ "$searxng_answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
        if [[ "$searxng_container_owned" == true ]]; then
            current_label="$("$searxng_runtime" container inspect \
                --format '{{ index .Config.Labels "io.lowp.managed" }}' \
                "$searxng_name" 2>/dev/null || true)"
            if [[ "$current_label" == "true" ]]; then
                "$searxng_runtime" container rm --force "$searxng_name" \
                    || { printf 'Could not remove the verified SearXNG container; it was left in place.\n' >&2; searxng_cleanup_safe=false; }
            else
                printf 'SearXNG container ownership changed; it was left in place.\n' >&2
                searxng_cleanup_safe=false
            fi
        fi
        if [[ "$searxng_backup_owned" == true ]]; then
            current_label="$("$searxng_backup_runtime" container inspect \
                --format '{{ index .Config.Labels "io.lowp.managed" }}' \
                "$searxng_backup_name" 2>/dev/null || true)"
            if [[ "$current_label" == "true" ]]; then
                "$searxng_backup_runtime" container rm --force "$searxng_backup_name" \
                    || { printf 'Could not remove the verified previous SearXNG container; it was left in place.\n' >&2; searxng_cleanup_safe=false; }
            else
                printf 'Previous SearXNG container ownership changed; it was left in place.\n' >&2
                searxng_cleanup_safe=false
            fi
        fi
        if [[ "$searxng_config_owned" == true && "$searxng_cleanup_safe" == true \
            && -d "$searxng_dir" \
            && ! -L "$searxng_dir" && -f "$searxng_marker" && ! -L "$searxng_marker" ]] \
            && grep -Fqx -- "$launcher_marker" "$searxng_marker"; then
            rm -r -- "$searxng_dir"
        elif [[ "$searxng_config_owned" == true && "$searxng_cleanup_safe" != true ]]; then
            printf 'SearXNG settings were preserved because an owned container could not be removed.\n' >&2
        else
            printf 'SearXNG settings ownership could not be reverified; they were preserved.\n' >&2
        fi
    else
        printf 'LowP-owned SearXNG resources were preserved.\n'
    fi
fi

if [[ "$purge_data" == true ]]; then
    printf '\nThe following LowP user-data directories are candidates for permanent removal:\n'
    printf '  Configuration (settings): %s\n' "$config_dir"
    purge_directories=("$config_dir" "$cache_dir")
    if [[ -L "$user_data_dir" || -L "$config_dir" || -L "$cache_dir" ]]; then
        printf 'Refusing data cleanup because a target directory is a symbolic link.\n' >&2
        exit 1
    fi
    if [[ -e "$searxng_dir" ]]; then
        printf '  Persistent data: preserved at %s because SearXNG data remains there.\n' "$user_data_dir"
    else
        printf '  Persistent data (knowledge, adapters, datasets, and sessions): %s\n' "$user_data_dir"
        purge_directories+=("$user_data_dir")
    fi
    printf '  Cache: %s\n' "$cache_dir"
    printf 'This does not remove Ollama models, user documents/projects, or shared container data.\n'
    read -r -p "Permanently remove these exact LowP directories? [y/N] " purge_answer || purge_answer=""
    if [[ "$purge_answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]; then
        for directory in "${purge_directories[@]}"; do
            if [[ -L "$directory" ]]; then
                printf 'Refusing to remove symbolic-link data path: %s\n' "$directory" >&2
                exit 1
            fi
            if [[ -e "$directory" ]]; then
                [[ -d "$directory" ]] \
                    || { printf 'Refusing to remove non-directory data path: %s\n' "$directory" >&2; exit 1; }
            fi
        done
        for directory in "${purge_directories[@]}"; do
            if [[ -e "$directory" ]]; then
                rm -r -- "$directory" \
                    || { printf 'Could not remove LowP data directory: %s\n' "$directory" >&2; exit 1; }
            fi
        done
        if [[ -e "$searxng_dir" ]]; then
            printf 'Confirmed LowP configuration and cache were removed; persistent data containing SearXNG was preserved.\n'
        else
            printf 'Confirmed LowP configuration, persistent data, and cache were removed.\n'
        fi
    else
        printf 'LowP user data was preserved.\n'
    fi
fi
