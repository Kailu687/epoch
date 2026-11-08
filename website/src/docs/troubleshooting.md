---
title: Troubleshooting
order: 6
status: partial
statusnote: Early; will grow with real issues.
desc: Common problems and fixes.
---
## The application will not start
Check Python 3.12 is active and run `pip install -r requirements.txt` again.

## Import fails
The file may not be a valid CSV, may be too large for the configured limit, or may use an unsupported encoding. The error message names the cause.

## Training reports a class with too few samples
Stratified splitting needs at least a few examples per class. Merge or remove rare classes.

## Metrics look too good
Look for target leakage warnings, and check whether rows from the same entity appear in many rows.