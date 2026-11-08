# Windows setup and support status

## Validation status

LowP has a native Python terminal application, PowerShell installers, Windows
per-user paths, Windows process cancellation, and a restricted set of native
PowerShell workspace commands. Linux remains the only platform verified by the
current automated install lifecycle. No Windows host, PowerShell runtime, or
Windows Terminal is available in the development environment used for this
release work, so Windows is not yet release-verified.

The application does not require WSL, Docker, a browser, administrator rights,
or an embedded terminal. The Windows scripts do not install Ollama, Git,
Docker/Podman, or language models.

## Install

Use Windows PowerShell 5.1 or PowerShell 7+ from a normal user account. From a
project checkout, run one of:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

The `Bypass` setting applies only to this PowerShell process; the installer
does not change user or machine execution policy. The installer checks the
Windows version, Python 3.11+, pip, and available disk space; creates a
dedicated virtual environment; installs dependencies from `pyproject.toml`;
offers optional LoRA/QLoRA packages only after confirmation; initializes user
directories; and verifies `lowp --check`. Re-running it is
intended to repair/update the same marked installation without replacing user
data.

The installer adds `%LOCALAPPDATA%\Programs\LowP\bin` to the current user's
PATH. Start a new PowerShell or Windows Terminal session to receive the
updated PATH. If that did not take effect, add this exact directory using
**Settings → System → About → Advanced system settings → Environment
Variables → User variables → Path**:

```text
%LOCALAPPDATA%\Programs\LowP\bin
```

No PowerShell profile or shell startup file is edited.

## Direct developer setup

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m lowparllm
```

Activation is optional. If policy blocks the activation script, invoking the
environment's `python.exe` directly avoids changing PowerShell execution
policy.

## Paths, services, and terminal behavior

- Configuration: `%APPDATA%\lowp\config.toml`
- Persistent data: `%LOCALAPPDATA%\lowp`
- Cache: `%LOCALAPPDATA%\lowp\cache`
- Application runtime: `%LOCALAPPDATA%\Programs\LowP\venv`
- Launcher: `%LOCALAPPDATA%\Programs\LowP\bin\lowp.cmd`

LowP accepts drive-letter paths, backslashes, forward slashes, relative paths,
home expansion, and quoted paths with spaces. For example:

```text
/cd C:\Users\User\Projects
/cd "C:\Users\User\My Projects\App"
/cd ..
/pwd
/ls
```

The slash commands `/cd`, `/pwd`, and `/ls` are LowP commands, not shell
commands. The coding command runner does not invoke arbitrary shell scripts.
On Windows it allows a restricted, read-only PowerShell subset (`Get-Location`,
`Get-ChildItem`, `Get-Content` with a bounded line count, and `Get-Process`).
Filesystem operations are constrained to existing, non-sensitive paths inside
the active workspace. External development tools such as Python, pytest, and
Git are executed directly without a shell and still require explicit approval.

`/copy` uses `clip.exe`. Ollama remains optional for startup and diagnostics;
the installer only inspects installed tags with `ollama list` and never pulls,
replaces, or deletes a model. SearXNG is optional and must be configured
separately. The installer does not install or expose a network service.

## Update and repair

From the same source checkout used for installation:

```powershell
.\install.ps1 -Update
.\install.ps1 -Repair
```

The installed launcher also supports:

```powershell
lowp doctor
lowp update
lowp repair
```

Updating reinstalls declared Python dependencies and verifies the installed
entry point. Repair uses the same non-destructive installer path. Neither
operation removes user configuration, model files, adapters, training data,
RAG databases, sessions, or projects.

## Uninstall

From a checkout, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\uninstall.ps1
```

It reports the marked runtime and launchers, asks the user to type `REMOVE`,
stops only processes whose executable path is inside the verified LowP virtual
environment, removes its PATH entry, and preserves all user data. It does not
uninstall shared software or delete a project checkout. There is no implicit
user-data purge option.

## Tests still required on Windows

Before describing Windows as release-ready, run and record the full lifecycle
on actual Windows 10/11 with PowerShell 5.1 and PowerShell 7+:

1. Fresh install, repeated install, first launch, `lowp doctor`, update, repair,
   uninstall, and reinstall.
2. Verify PATH from new PowerShell and Windows Terminal sessions, including a
   home path containing spaces.
3. Check Ollama model listing/connectivity without downloading or altering
   models; check missing Ollama and missing-model diagnostics.
4. Verify PowerShell workspace commands, external Python/Git commands,
   cancellation, timeout, approval, and workspace-boundary rejection.
5. Test `C:\` paths, spaces, relative navigation, symlinks where permitted,
   file locks, config/data preservation, and uninstallation with an active
   training worker.
6. Run the entire Python test suite and inspect any Windows-specific skips.

Do not infer Windows compatibility from the Linux platform-simulation tests in
this repository.
