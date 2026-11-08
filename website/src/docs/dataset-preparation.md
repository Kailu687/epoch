---
title: Dataset preparation
order: 3
status: partial
statusnote: Applies to CSV tabular data only.
desc: How to prepare a CSV for training.
---
## Format
- One row per example, one header row, one column per field.
- Use a consistent missing-value marker (empty cell is best).
- Keep original column names; the engine preserves them in the export schema.

## Before you import
- Remove columns that would not exist when making a real prediction (these cause target leakage).
- Remove or mark identifiers such as customer IDs. The engine excludes columns that are unique per row, but check its suggestion.
- If rows belong to the same person, device or session, keep an identifier column. Group-aware splitting is planned; today a random split may overstate performance for such data.
- Classes with fewer than a handful of examples cannot be split reliably and will be reported.