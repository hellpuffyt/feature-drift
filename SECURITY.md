# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/feature-drift/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

feature-drift reads local CSV/JSON Lines files and prints a report; it makes no network calls and never executes input content. Relevant reports include: input that executes code, crashes with a traceback instead of exit code 1, or yields a silently wrong drift metric.
