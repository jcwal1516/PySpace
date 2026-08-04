# Security policy

Report vulnerabilities through [GitHub private vulnerability reporting](https://github.com/jcwal1516/PySpace/security/advisories/new).
Do not open a public issue containing credentials, protected data, or exploit
details.

Version 0.1.x receives security fixes while it is the current release. No older
release line exists.

PySpace does not load pickle files. Treat all tables, images, result bundles, and
optional native-code dependencies as untrusted input and validate their
provenance. See `security_best_practices_report.md` for the release audit.
