#!/usr/bin/env bash
set -euo pipefail

readonly launcher_marker="# Managed by LowParLLMCapacityIncV0 installer."
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly python_bin="${PYTHON:-python3}"
readonly home_dir="${HOME:?HOME must be set}"

fail() {
    printf 'Installation failed: %s\n' "$*" >&2
    exit 1
}

mode="install"
if (($# > 1)); then
    fail "usage: ./install.sh [--repair|--update]"
fi
if (($# == 1)); then
    case "$1" in
        --repair) mode="repair" ;;
        --update) mode="update" ;;
        *) fail "usage: ./install.sh [--repair|--update]" ;;
    esac
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "this installer currently supports Linux only."
fi
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    fail "run this installer as your normal user; root permissions are not required."
fi
command -v "$python_bin" >/dev/null 2>&1 || fail "Python 3 was not found (looked for $python_bin)."
"$python_bin" -m pip --version >/dev/null 2>&1 || fail "pip is unavailable for $python_bin."

project_file="$script_dir/pyproject.toml"
[[ -f "$project_file" && -f "$script_dir/lowparllm/main.py" ]] \
    || fail "project files are missing; run this script from a complete project checkout."
required_python="$(sed -n 's/^requires-python = ">=\([^"]*\)".*/\1/p' "$project_file" | head -n 1)"
[[ -n "$required_python" ]] || fail "could not read requires-python from pyproject.toml."
if ! "$python_bin" - "$required_python" <<'PY'
import sys

required = tuple(int(part) for part in sys.argv[1].split("."))
if sys.version_info[: len(required)] < required:
    raise SystemExit(1)
PY
then
    fail "Python $required_python or newer is required; found $("$python_bin" --version 2>&1)."
fi

data_home="${XDG_DATA_HOME:-$home_dir/.local/share}"
if [[ "$data_home" != /* ]]; then
    data_home="$home_dir/.local/share"
fi
readonly venv_dir="${data_home%/}/lowp/venv"
readonly bin_dir="$home_dir/.local/bin"
readonly launcher="$bin_dir/lowp"
readonly marker_file="$venv_dir/.lowp-installed"

if [[ -L "$venv_dir" ]]; then
    fail "refusing to use a symbolic link at the application runtime path: $venv_dir"
fi
if [[ -e "$venv_dir" ]]; then
    [[ -d "$venv_dir" && -f "$marker_file" && ! -L "$marker_file" ]] \
        || fail "runtime path already exists and is not marked as LowParLLM-owned: $venv_dir"
    grep -Fqx -- "$launcher_marker" "$marker_file" \
        || fail "runtime ownership marker is not recognized at $venv_dir"
fi
if [[ -e "$launcher" ]]; then
    [[ -f "$launcher" && ! -L "$launcher" ]] \
        || fail "refusing to overwrite a non-regular launcher path: $launcher"
    grep -Fqx -- "$launcher_marker" "$launcher" \
        || fail "refusing to overwrite an unrelated file at $launcher"
fi

if command -v df >/dev/null 2>&1; then
    available_kib="$(df -Pk "$home_dir" | awk 'NR == 2 { print $4 }')"
    if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
        printf 'Available space on home filesystem: %s KiB\n' "$available_kib"
        if (( available_kib < 100000 )); then
            printf 'Warning: less than 100 MiB is available; installation may fail.\n' >&2
        fi
    fi
fi

if [[ ! -e "$venv_dir" ]]; then
    mkdir -p -- "$(dirname -- "$venv_dir")"
    "$python_bin" -m venv "$venv_dir" || fail "could not create the application virtual environment."
    printf '%s\n' "$launcher_marker" > "$marker_file"
elif [[ ! -x "$venv_dir/bin/python" ]]; then
    fail "the marked application environment is incomplete; refusing to replace it."
fi

"$venv_dir/bin/python" -m pip --version >/dev/null 2>&1 \
    || fail "pip is unavailable in the dedicated application environment."
printf 'Installing LowParLLM into %s\n' "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade "$script_dir" \
    || fail "package installation failed; rerun this script after resolving the pip error."
"$venv_dir/bin/python" - <<'PY' \
    || fail "could not safely remove obsolete media-creation dependencies."
import importlib.metadata
import re
import subprocess
import sys

obsolete = ("diffusers", "reportlab", "Pillow")
installed = {
    re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(): distribution
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
requirements = tuple(
    (distribution.metadata["Name"], requirement)
    for distribution in installed.values()
    for requirement in distribution.requires or ()
)
for package in obsolete:
    normalized = re.sub(r"[-_.]+", "-", package).lower()
    if normalized not in installed:
        continue
    dependents = tuple(
        dependent
        for dependent, requirement in requirements
        if (match := re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement))
        and re.sub(r"[-_.]+", "-", match.group(1)).lower() == normalized
    )
    if dependents:
        print(
            f"Preserving obsolete media package {package}; "
            f"installed packages still declare it: {', '.join(dependents)}"
        )
        continue
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", package],
        check=True,
    )
PY

mkdir -p -- "$bin_dir"
temporary_launcher="$(mktemp "$bin_dir/.lowp.XXXXXX")"
trap 'rm -f -- "$temporary_launcher"' EXIT
printf '#!/usr/bin/env bash\n%s\nexport LOWP_SOURCE_DIR=%q\nexec ' \
    "$launcher_marker" "$script_dir" > "$temporary_launcher"
printf '%q "$@"\n' "$venv_dir/bin/lowp" >> "$temporary_launcher"
chmod 755 "$temporary_launcher"
mv -f -- "$temporary_launcher" "$launcher"
trap - EXIT

[[ -x "$launcher" && -x "$venv_dir/bin/lowp" ]] \
    || fail "launcher verification failed."
"$venv_dir/bin/python" - "$venv_dir" <<'PY'
import importlib.metadata
import pathlib
import sys

environment = pathlib.Path(sys.argv[1])
distribution = importlib.metadata.distribution("LowParLLMCapacityIncV0")
entry_points = {
    (entry.group, entry.name, entry.value)
    for entry in distribution.entry_points
}
expected = ("console_scripts", "lowp", "lowparllm.main:main")
if expected not in entry_points or not (environment / "bin/lowp").is_file():
    raise SystemExit("installed package or console entry point could not be verified")
PY
if ! "$venv_dir/bin/python" - <<'PY'
from lowparllm.core.config import load_config
from lowparllm.core.paths import ensure_cache_dir, ensure_data_dir

load_config()
ensure_data_dir()
ensure_cache_dir()
PY
then
    fail "could not initialize LowP configuration and user directories."
fi

"$launcher" --check || fail "the installed application health check failed."

printf '\nLowParLLM installed successfully.\n'
case ":${PATH:-}:" in
    *":$bin_dir:"*)
        printf 'Launch from any directory with: lowp\n'
        ;;
    *)
        printf '\nThe launcher is installed at %s but is not currently in PATH.\n' "$launcher"
        printf 'Add %s for future shells:\n' "$bin_dir"
        case "$(basename -- "${SHELL:-}")" in
            bash) printf '  echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.bashrc\n' ;;
            zsh) printf '  echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.zshrc\n' ;;
            fish) printf '  fish_add_path "$HOME/.local/bin"\n' ;;
            *) printf '  Add "$HOME/.local/bin" to PATH in your shell startup configuration.\n' ;;
        esac
        printf 'No shell startup files were changed.\n'
        ;;
esac

confirm() {
    local answer
    read -r -p "$1 [y/N] " answer || answer=""
    [[ "$answer" =~ ^([Yy]|[Yy][Ee][Ss])$ ]]
}

model_config_status="deferred"
model_installed_status="unknown"
searxng_installed_status="not installed"
searxng_running_status="no"
searxng_health_status="not verified"

persist_model() {
    "$venv_dir/bin/python" - "$1" <<'PY'
import sys
from lowparllm.ai.llm.model_manager import persist_model_name
from lowparllm.core.config import load_config

load_config()
persist_model_name(sys.argv[1])
PY
}

select_model() {
    local current_model="" selected="" choice="" model_tag="" index=""
    local -a installed_models=()
    if ! command -v ollama >/dev/null 2>&1; then
        printf '\nOllama: not installed. Model configuration is deferred.\n'
        return
    fi
    printf '\nOllama: installed\n'
    if ! ollama list >/dev/null 2>&1; then
        printf 'Ollama service: unavailable. Start it and rerun install.sh to select a model.\n'
        return
    fi
    mapfile -t installed_models < <(ollama list | awk 'NR > 1 && NF { print $1 }')
    current_model="$("$venv_dir/bin/python" -c \
        'from lowparllm.core.config import load_config; print(load_config().base_model)')"
    printf 'Ollama service: running\n'
    if [[ -n "${LOWP_BASE_MODEL:-}" ]]; then
        printf 'LOWP_BASE_MODEL is set and will override the persisted model selection in this shell.\n'
    fi
    if [[ -n "$current_model" ]] && printf '%s\n' "${installed_models[@]}" \
        | grep -Fxq -- "$current_model"; then
        model_config_status="configured"
        model_installed_status="installed"
        printf 'Current default model: %s [Installed]\n' "$current_model"
    fi
    printf '\n========================================\n'
    printf '              LOWP MODEL SETUP\n'
    printf '========================================\n'
    if ((${#installed_models[@]})); then
        printf 'Installed Ollama models:\n'
        for index in "${!installed_models[@]}"; do
            if [[ "${installed_models[$index]}" == "$current_model" ]]; then
                printf '  %d. %s [Installed] [Current]\n' "$((index + 1))" "${installed_models[$index]}"
            else
                printf '  %d. %s [Installed]\n' "$((index + 1))" "${installed_models[$index]}"
            fi
        done
    else
        printf 'No models are currently installed in Ollama.\n'
    fi
    printf '  %d. Enter another Ollama model tag\n' "$((${#installed_models[@]} + 1))"
    printf '  %d. Configure model later\n' "$((${#installed_models[@]} + 2))"
    if [[ -n "$current_model" ]] && printf '%s\n' "${installed_models[@]}" \
        | grep -Fxq -- "$current_model"; then
        printf '  %d. Keep current model\n' "$((${#installed_models[@]} + 3))"
    fi
    read -r -p "Select an option: " choice || choice=""
    if [[ "$choice" =~ ^[0-9]+$ ]] \
        && (( choice >= 1 && choice <= ${#installed_models[@]} )); then
        selected="${installed_models[$((choice - 1))]}"
        persist_model "$selected" || fail "could not save the selected Ollama model."
        model_config_status="configured"
        model_installed_status="installed"
        printf 'Default model configured: %s\n' "$selected"
        return
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] \
        && (( choice == ${#installed_models[@]} + 1 )); then
        read -r -p "Enter the exact Ollama model tag: " model_tag || model_tag=""
        if [[ -z "$model_tag" ]]; then
            printf 'No model selected; model configuration is deferred.\n'
            return
        fi
        if ! "$venv_dir/bin/python" - "$model_tag" <<'PY'
import sys
from lowparllm.ai.llm.model_manager import validate_model_name

try:
    validate_model_name(sys.argv[1])
except ValueError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
        then
            printf 'Invalid Ollama model tag; model configuration is deferred.\n' >&2
            return
        fi
        if ! printf '%s\n' "${installed_models[@]}" | grep -Fxq -- "$model_tag"; then
            printf 'Model %s is not installed locally.\n' "$model_tag"
            if ! confirm "Download it now with ollama pull?"; then
                printf 'No model was downloaded; model configuration is deferred.\n'
                return
            fi
            if ! ollama pull "$model_tag"; then
                printf 'Model download failed or was interrupted. Existing models were not changed.\n' >&2
                return
            fi
            if ! ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq -- "$model_tag"; then
                printf 'Ollama did not report %s as installed after the pull; it was not selected.\n' "$model_tag" >&2
                return
            fi
        fi
        persist_model "$model_tag" || fail "could not save the selected Ollama model."
        model_config_status="configured"
        model_installed_status="installed"
        printf 'Default model configured: %s\n' "$model_tag"
        return
    fi
    if [[ "$choice" =~ ^[0-9]+$ ]] \
        && (( choice == ${#installed_models[@]} + 3 )) \
        && [[ -n "$current_model" ]] \
        && printf '%s\n' "${installed_models[@]}" | grep -Fxq -- "$current_model"; then
        model_config_status="configured"
        model_installed_status="installed"
        printf 'Default model preserved: %s\n' "$current_model"
        return
    fi
    printf 'Model configuration deferred. Use /model after Ollama is ready.\n'
}

install_optional_training() {
    local cuda_available=false
    if ! confirm "Install optional Transformers/PEFT training dependencies? They may use several GiB."; then
        training_status="not installed (user declined optional dependencies)"
        printf 'Optional training dependencies: not installed.\n'
        return
    fi
    if ! "$venv_dir/bin/python" -m pip install --upgrade "${script_dir}[training]"; then
        training_status="installation failed"
        printf 'Optional training dependencies: installation failed; LowP chat remains installed.\n' >&2
        return
    fi
    training_status="LoRA dependency set installed; training still requires compatible local checkpoint and hardware"
    if "$venv_dir/bin/python" -c \
        'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
        >/dev/null 2>&1; then
        cuda_available=true
    fi
    if [[ "$cuda_available" == true ]] && confirm \
        "Install optional bitsandbytes for CUDA QLoRA? Compatibility depends on your driver and PyTorch build."; then
        if ! "$venv_dir/bin/python" -m pip install --upgrade "${script_dir}[qlora]"; then
            qlora_status="installation failed"
            printf 'Optional QLoRA package installation failed; LoRA remains available if compatible.\n' >&2
        else
            qlora_status="installed; runtime compatibility still requires verification"
        fi
    else
        printf 'QLoRA package not selected or no working CUDA runtime was detected.\n'
    fi
}

check_searxng_json() {
    "$venv_dir/bin/python" - "$1" <<'PY'
import json
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

url = sys.argv[1].rstrip("/") + "/search?" + urlencode({"q": "lowp health check", "format": "json"})
try:
    with urlopen(url, timeout=5) as response:
        payload = json.loads(response.read())
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not any(
        isinstance(item, dict)
        and isinstance(item.get("title"), str)
        and isinstance(item.get("url"), str)
        for item in results
    ):
        raise ValueError("JSON response contained no valid search results")
except Exception as exc:
    print(f"SearXNG JSON search failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

start_managed_searxng() {
    "$container_runtime" run --detach \
        --name "$service_name" \
        --label io.lowp.managed=true \
        --restart unless-stopped \
        --security-opt no-new-privileges \
        --env SEARXNG_BASE_URL=http://127.0.0.1:8080/ \
        --publish 127.0.0.1:8080:8080 \
        --volume "$settings_file:/etc/searxng/settings.yml:rw,Z" \
        "$image"
}

setup_searxng() {
    local endpoint="http://127.0.0.1:8080"
    local container_runtime="" image="searxng/searxng@sha256:c750ee1d859b26d3aeae51c07f6a48c3a0bfbaa9195d71a18b258862d88c4c3c"
    local service_name="lowp-searxng" config_dir settings_file marker existing_label running
    local runtime_candidate=""
    local new_settings=false
    local backup_name="" backup_label="" new_label="" recovery_name="" current_image=""
    config_dir="${data_home%/}/lowp/searxng"
    settings_file="$config_dir/settings.yml"
    marker="$config_dir/.lowp-searxng-managed"
    recovery_name="${service_name}-previous"

    if check_searxng_json "$endpoint"; then
        for runtime_candidate in docker podman; do
            if command -v "$runtime_candidate" >/dev/null 2>&1 \
                && "$runtime_candidate" container inspect "$recovery_name" >/dev/null 2>&1; then
                existing_label="$("$runtime_candidate" container inspect \
                    --format '{{ index .Config.Labels "io.lowp.managed" }}' \
                    "$recovery_name" 2>/dev/null || true)"
                if [[ "$existing_label" == "true" ]]; then
                    "$runtime_candidate" container rm --force "$recovery_name" >/dev/null 2>&1 \
                        || printf 'SearXNG is healthy; a LowP-owned previous container remains as %s.\n' \
                            "$recovery_name" >&2
                fi
            fi
        done
        searxng_installed_status="existing service"
        searxng_running_status="yes"
        searxng_health_status="verified"
        printf '\nSearXNG: reusing healthy existing service at %s\n' "$endpoint"
        return
    fi
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        runtime_candidate="docker"
    fi
    if command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
        if [[ -z "$runtime_candidate" ]]; then
            runtime_candidate="podman"
        else
            printf 'Docker and Podman are operational; using Docker predictably.\n'
        fi
    fi
    if [[ -z "$runtime_candidate" ]]; then
        printf '\nSearXNG: NOT installed or health-verified.\n'
        printf 'Automated SearXNG deployment needs a working Docker or Podman runtime; LowP will not install one.\n'
        printf 'Install and enable Docker/Podman as your system administrator, then rerun ./install.sh.\n'
        return
    fi
    container_runtime="$runtime_candidate"
    printf '\nSearXNG container runtime: %s\n' "$container_runtime"
    if [[ -e "$config_dir" && ( ! -d "$config_dir" || -L "$config_dir" ) ]]; then
        printf 'SearXNG: configuration path is not a safe directory: %s\n' "$config_dir" >&2
        return
    fi
    if [[ -d "$config_dir" && ( ! -f "$marker" || -L "$marker" ) ]]; then
        printf 'SearXNG: refusing to take over an unmarked configuration directory: %s\n' "$config_dir" >&2
        return
    fi
    if [[ -L "$settings_file" ]]; then
        printf 'SearXNG: refusing to use a symbolic link for settings: %s\n' "$settings_file" >&2
        return
    fi
    if ! "$container_runtime" container inspect "$service_name" >/dev/null 2>&1 \
        && "$container_runtime" container inspect "$recovery_name" >/dev/null 2>&1; then
        existing_label="$("$container_runtime" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' \
            "$recovery_name" 2>/dev/null || true)"
        if [[ "$existing_label" != "true" || ! -f "$marker" || -L "$marker" ]]; then
            printf 'SearXNG: a previous container name is occupied by unverified resources; leaving it untouched.\n' >&2
            return
        fi
        if ! "$container_runtime" container rename "$recovery_name" "$service_name" \
            || ! "$container_runtime" start "$service_name" >/dev/null; then
            printf 'SearXNG: could not restore the previous LowP-owned container for repair.\n' >&2
            return
        fi
        printf 'SearXNG: restored the LowP-owned container left by an interrupted update.\n'
    fi
    if ! "$container_runtime" container inspect "$service_name" >/dev/null 2>&1; then
        if ! confirm "Download the pinned SearXNG image and create a loopback-only LowP service?"; then
            printf 'SearXNG: not installed; no container was created.\n'
            return
        fi
        if [[ ! -d "$config_dir" ]]; then
            mkdir -p -- "$config_dir"
            chmod 755 "$config_dir"
        fi
        if [[ ! -f "$marker" ]]; then
            printf '%s\n' "$launcher_marker" > "$marker"
            chmod 600 "$marker"
        fi
        if [[ ! -e "$settings_file" ]]; then
            new_settings=true
            "$venv_dir/bin/python" - "$settings_file" <<'PY'
import os
import secrets
import sys
import tempfile

target = sys.argv[1]
content = (
    "use_default_settings: true\n"
    "server:\n"
    f'  secret_key: "{secrets.token_urlsafe(48)}"\n'
    "  limiter: false\n"
    "search:\n"
    "  formats:\n"
    "    - html\n"
    "    - json\n"
    "  safe_search: 1\n"
)
fd, temporary = tempfile.mkstemp(prefix=".settings-", dir=os.path.dirname(target))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    os.chmod(temporary, 0o644)
    os.replace(temporary, target)
except OSError:
    try:
        os.unlink(temporary)
    except OSError:
        pass
    raise
PY
        fi
        if [[ "$new_settings" == true ]]; then
            chmod 644 "$settings_file"
        fi
        if ! start_managed_searxng; then
            printf 'SearXNG: container creation failed. LowP-owned settings were preserved for repair.\n' >&2
            return
        fi
        searxng_installed_status="installed"
        searxng_running_status="yes"
    else
        searxng_installed_status="container exists"
        existing_label="$("$container_runtime" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' "$service_name" 2>/dev/null || true)"
        if [[ "$existing_label" != "true" ]]; then
            running="$("$container_runtime" container inspect \
                --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
            [[ "$running" == "true" ]] && searxng_running_status="yes"
            printf 'SearXNG: a container named %s exists but is not marked LowP-owned; leaving it untouched.\n' "$service_name" >&2
            return
        fi
        if [[ ! -f "$marker" || -L "$marker" ]]; then
            printf 'SearXNG: existing LowP container has no matching owned configuration marker; refusing to alter it.\n' >&2
            return
        fi
        if "$container_runtime" container inspect "$recovery_name" >/dev/null 2>&1; then
            existing_label="$("$container_runtime" container inspect \
                --format '{{ index .Config.Labels "io.lowp.managed" }}' \
                "$recovery_name" 2>/dev/null || true)"
            if [[ "$existing_label" == "true" ]]; then
                backup_name="$recovery_name"
            fi
        fi
        searxng_installed_status="installed"
        running="$("$container_runtime" container inspect \
            --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
        [[ "$running" == "true" ]] && searxng_running_status="yes"
        current_image="$("$container_runtime" container inspect \
            --format '{{.Config.Image}}' "$service_name" 2>/dev/null || true)"
        if [[ "$current_image" != "$image" ]]; then
            if [[ -n "$backup_name" ]]; then
                printf 'SearXNG: a verified previous container is available; checking the current service before another update.\n'
            elif ! confirm "Update the verified LowP-owned SearXNG container to the pinned image?"; then
                searxng_installed_status="installed (update declined)"
                running="$("$container_runtime" container inspect \
                    --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
                [[ "$running" == "true" ]] && searxng_running_status="yes"
                printf 'SearXNG: existing LowP-owned container was preserved without an image update.\n'
                return
            fi
            if ! "$container_runtime" image pull "$image"; then
                printf 'SearXNG: pinned image update failed; the existing container was preserved.\n' >&2
                return
            fi
            if [[ -z "$backup_name" ]]; then
                backup_name="$recovery_name"
                if "$container_runtime" container inspect "$backup_name" >/dev/null 2>&1; then
                    printf 'SearXNG: update backup name is occupied; refusing to replace the service.\n' >&2
                    return
                fi
                "$container_runtime" stop "$service_name" \
                    || { printf 'SearXNG: could not stop the old container; it was preserved.\n' >&2; return; }
                if ! "$container_runtime" container rename "$service_name" "$backup_name"; then
                    "$container_runtime" start "$service_name" >/dev/null 2>&1 || true
                    printf 'SearXNG: could not stage the old container for update; it was preserved.\n' >&2
                    return
                fi
                if ! start_managed_searxng; then
                    "$container_runtime" container rename "$backup_name" "$service_name" \
                        && "$container_runtime" start "$service_name" >/dev/null 2>&1 || true
                    printf 'SearXNG: update failed; the prior LowP-owned container was restored.\n' >&2
                    return
                fi
                searxng_installed_status="installed (updating)"
                searxng_running_status="yes"
            fi
        fi
        running="$("$container_runtime" container inspect \
            --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
        if [[ "$running" != "true" ]]; then
            "$container_runtime" start "$service_name" \
                || { printf 'SearXNG: could not start the existing LowP service.\n' >&2; return; }
        fi
        searxng_installed_status="installed"
        searxng_running_status="yes"
    fi
    for _attempt in {1..30}; do
        if check_searxng_json "$endpoint" >/dev/null 2>&1; then
            if [[ -n "$backup_name" ]]; then
                backup_label="$("$container_runtime" container inspect \
                    --format '{{ index .Config.Labels "io.lowp.managed" }}' \
                    "$backup_name" 2>/dev/null || true)"
                if [[ "$backup_label" == "true" ]] \
                    && ! "$container_runtime" container rm --force "$backup_name" >/dev/null; then
                    printf 'SearXNG: updated service is healthy; the stopped previous container remains as %s.\n' "$backup_name" >&2
                elif [[ "$backup_label" != "true" ]]; then
                    printf 'SearXNG: previous container ownership changed; it was left untouched.\n' >&2
                fi
            fi
            searxng_installed_status="installed"
            searxng_health_status="verified"
            printf 'SearXNG: running and JSON search verified at %s\n' "$endpoint"
            return
        fi
        sleep 2
    done
    if [[ -n "$backup_name" ]]; then
        new_label="$("$container_runtime" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' \
            "$service_name" 2>/dev/null || true)"
        backup_label="$("$container_runtime" container inspect \
            --format '{{ index .Config.Labels "io.lowp.managed" }}' \
            "$backup_name" 2>/dev/null || true)"
        if [[ "$new_label" == "true" && "$backup_label" == "true" ]]; then
            "$container_runtime" container rm --force "$service_name" >/dev/null 2>&1 || true
            "$container_runtime" container rename "$backup_name" "$service_name" \
                && "$container_runtime" start "$service_name" >/dev/null 2>&1 || true
        else
            printf 'SearXNG: rollback ownership could not be verified; both containers were preserved.\n' >&2
        fi
        searxng_installed_status="installed (rollback attempted)"
        running="$("$container_runtime" container inspect \
            --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
        if [[ "$running" == "true" ]]; then
            searxng_running_status="yes"
        else
            searxng_running_status="no"
        fi
        if check_searxng_json "$endpoint" >/dev/null 2>&1; then
            searxng_running_status="yes"
            searxng_health_status="verified"
            printf 'SearXNG: failed update rolled back; the previous service is healthy.\n' >&2
            return
        fi
    fi
    running="$("$container_runtime" container inspect \
        --format '{{.State.Running}}' "$service_name" 2>/dev/null || true)"
    if [[ "$running" != "true" ]]; then
        searxng_running_status="no"
    fi
    printf 'SearXNG: container exists but its JSON search health check failed at %s.\n' "$endpoint" >&2
    printf 'Inspect the LowP-owned container logs with: %s logs %s\n' "$container_runtime" "$service_name" >&2
}

training_status="not changed by $mode"
qlora_status="not changed by $mode"
model_config_status="deferred"
if [[ "$mode" == "install" ]]; then
    training_status="not installed (user declined optional dependencies)"
    qlora_status="not installed"
    install_optional_training
    select_model
    setup_searxng
else
    model_config_status="preserved by $mode"
fi

printf '\nLowP %s report\n' "$mode"
printf '  LowP application: installed and launcher verified\n'
printf '  Optional training packages: %s\n' "$training_status"
printf '  Optional QLoRA package: %s\n' "$qlora_status"
if command -v ollama >/dev/null 2>&1 && ollama list >/dev/null 2>&1; then
    selected_model="$("$venv_dir/bin/python" -c \
        'from lowparllm.core.config import load_config; print(load_config().base_model)')"
    if ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq -- "$selected_model"; then
        model_config_status="configured"
        model_installed_status="installed"
    else
        model_config_status="saved setting is not installed"
        model_installed_status="not installed"
    fi
    printf '  Model configured: %s\n' "$model_config_status"
    printf '  Selected Ollama model: %s (%s)\n' "$selected_model" "$model_installed_status"
else
    printf '  Model configured: %s\n' "$model_config_status"
    printf '  Ollama/model configuration: unavailable or deferred\n'
fi
if check_searxng_json "http://127.0.0.1:8080" >/dev/null 2>&1; then
    searxng_running_status="yes"
    searxng_health_status="verified"
    if [[ "$searxng_installed_status" == "not installed" ]]; then
        searxng_installed_status="existing service"
    fi
else
    searxng_health_status="not verified"
    if [[ "$searxng_running_status" == "yes" ]]; then
        searxng_running_status="unknown (health check failed)"
    fi
fi
printf '  SearXNG installed: %s\n' "$searxng_installed_status"
printf '  SearXNG running: %s\n' "$searxng_running_status"
printf '  SearXNG JSON search health: %s\n' "$searxng_health_status"
if [[ "$searxng_health_status" != "verified" ]]; then
    printf '  SearXNG is NOT ready; resolve the message above and rerun ./install.sh\n'
fi
printf '  User data, Ollama models, and adapters were not removed.\n'
