---
name: readme-sync
description: Use when modifying files under metasurface_data_pipeline to ensure README documentation stays consistent with code, commands, outputs, and data schema.
---

# README Sync

When modifying anything under `metasurface_data_pipeline`, check whether `README.md` must be updated in the same change.

## Required Checks

Before finishing a change, inspect whether the change affects:

- command-line entrypoints in `cli/`;
- text config keys or example config files;
- package layout or import paths;
- ROI, unit, subcell, polar analyzer, or A_matrix assumptions;
- output filenames;
- `.npz` / `.json` schema fields;
- expected downstream usage by 3DGS, NeSpoF, or other models;
- required environment, dependencies, or run commands.

If any item changed, update `README.md` in the same commit.

## How To Update

Keep README updates concise and concrete:

- update the exact command the user should run;
- list new or removed parameters;
- state behavior changes visibly;
- update output file names and schema notes;
- update the script/module purpose if responsibility moved;
- keep examples copy-pasteable for PowerShell.

Do not add vague notes such as "improved pipeline" without saying what changed.

## Verification

After modifying code or README, run at least:

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m unittest discover -s metasurface_data_pipeline\tests -v
```

If CLI behavior changed, also run the relevant `--help` command.

## Reporting

In the final response or commit summary, mention whether README was updated or why it did not need changes.
