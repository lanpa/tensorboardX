# TensorBoardX Foundation Mandates

This file contains foundational project knowledge, release procedures, and environment constraints for future development.

## 🚀 Release Process

The project uses automated GitHub Actions for publishing to PyPI and creating GitHub Releases.

1.  **Branching:** All releases **must** be initiated from a branch matching the pattern `releases/*` (e.g., `releases/v2.6.5`).
2.  **Workflow:** Trigger the `.github/workflows/publish-pypi.yml` action manually.
    *   **Input Version:** Use the format `X.Y.Z` (e.g., `2.6.5`). Do **not** include the leading `v`.
    *   **Automation:** The workflow automatically prepends `v` for the Git tag (`v2.6.5`) and strips it for the Python version check to ensure consistency.
3.  **Manual Gate:** The `publish_to_real_pypi` input must be set to `true` to deploy to production and create the GitHub Release.
4.  **Resiliency:** The workflow is configured with `skip-existing: true`. If a PyPI upload fails because the version exists, the workflow will continue and complete the GitHub Release creation.
5.  **Finalization:** Once the release is successful, merge the release branch back into `master` to keep the official history and tags synchronized.

## 🏷️ Tagging Conventions

*   **Prefix:** All official release tags must use the `v` prefix (e.g., `v2.6.4`, `v2.6.5`).
*   **History:** Tags must be part of the direct history of the `master` branch.
    *   *Correction Note:* `v2.6.4` was re-anchored to commit `ca5072b` to ensure correct diff calculation for subsequent releases.

## 🛠️ Environment & Dependencies

*   **Setuptools Compatibility:** TensorBoard (up to 2.20.0) depends on `pkg_resources`.
    *   **Breaking Change:** `setuptools` >= 82.0.0 removed `pkg_resources`.
    *   **Mandate:** For local development and testing, `setuptools` should be pinned to **`81.0.0`** or earlier to avoid `ModuleNotFoundError`.
*   **Version Management:** The project uses `setuptools_scm`. The version is derived dynamically from the most recent git tag.

## 🧪 Local Testing

Tests should be run locally before pushing any changes.

### **Quick Start**
The project provides a helper script to install dependencies and run tests:
```bash
./run_pytest.sh
```

### **Manual / UV Testing**
If using `uv` (preferred), ensure your environment is correctly pinned:
1.  **Pin Setuptools:** `uv pip install "setuptools==81.0.0"`
2.  **Run Pytest:** `uv run pytest`

### **Common Test Files**
*   `tests/test_writer.py`: Core `SummaryWriter` logic (including `write_to_disk` checks).
*   `tests/test_summary.py`: Protobuf summary generation.
*   `tests/test_summary_writer.py`: Integration tests for the writer.

## 🐛 Known Fixes

*   **SummaryWriter:** When `write_to_disk=False` is passed to the constructor, the `add_scalars` method correctly respects this flag and uses a `DummyFileWriter` for sub-directories (fixed in #765).
