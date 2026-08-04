# Security

PySpace is a local scientific library, not a network service. Its main trust
boundaries are imported tables/images, result bundles, optional native-code
dependencies, and the live-R parity checkout.

- Pickle is not accepted.
- NPZ loads use `allow_pickle=False`.
- Result member paths must remain below the bundle root.
- External commands use argument arrays and no shell.
- The live-R source must be the exact pinned commit with a clean Git status.
- Dependency, secret, CodeQL, and artifact scans run in CI.

The full prioritized review is in `security_best_practices_report.md` at the
repository root. Report vulnerabilities through [GitHub private vulnerability
reporting](https://github.com/jcwal1516/PySpace/security/advisories/new), not a
public issue.
