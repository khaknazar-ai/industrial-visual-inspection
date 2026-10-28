# Installation and launch

## Prepared workstation

From the project folder in PowerShell 7:

```powershell
.\run.ps1
```

Open http://127.0.0.1:8000. Stop with Ctrl+C before switching scenarios.

```powershell
.\run.ps1 -List                        # List available scenarios
.\run.ps1 -Scenario patchcore           # MVTec bottle / PatchCore
.\run.ps1 -Scenario superadd           # LOCO connectors / SuperADD
.\run.ps1 -Scenario ron88 -Example      # Inspect a local bottle example
.\run.ps1 -Scenario pcad -Image <path>  # Inspect a local assembly image
```

Use `-Port 8001` to choose a different port or `-ShowPlan` to see the command
without starting inference. `-Image <path>` performs a single CLI inspection.
The local launchers use existing model files and environments; they do not
train or download them. Ron88 and PCAD scenarios require CUDA in this setup.

## Fresh installation

Python 3.11 or 3.12 is required. From PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# For deep models, also install the optional profile:
.\.venv\Scripts\python.exe -m pip install -e ".[deep]"
```

The repository does not include datasets or model weights. A fresh installation
needs a compatible saved model and calibration before it can inspect real
images. The workstation launchers use local paths under `.tmp` and existing
Python environments; supply compatible assets and paths when using another
machine. Third-party datasets and weights have their own terms.

## API

When the service is running, open http://127.0.0.1:8000/docs for the API
reference. `GET /ready` reports model readiness; `POST /inspect` accepts image
bytes and returns the decision and saved evidence paths.
