# AI-slop and maintainability audit

This audit treats “AI slop” as observable engineering risk, not an authorship
guess. GitHub's review guidance calls for human oversight, tests, static
analysis, security checks, and maintainability review of generated code.
OWASP separately highlights hallucinated dependencies, unsafe generated code,
unreviewed broad changes, and fabricated or weakened tests. Empirical work also
reports that generated projects often fail because of dependency/runtime gaps
and code defects, and that low edit frequency is not evidence of maintainable
AI-authored code.

Sources:

- [GitHub: Reviewing AI-generated code](https://docs.github.com/en/enterprise-cloud%40latest/copilot/tutorials/review-ai-generated-code)
- [OWASP: Secure Coding Practices for Generative AI](https://cheatsheetseries.owasp.org/cheatsheets/Secure_Coding_with_AI_Cheat_Sheet.html)
- [AI-Generated Code Is Not Reproducible (Yet)](https://arxiv.org/abs/2512.22387)
- [The Long Tail of AI-Generated Software Maintenance](https://arxiv.org/abs/2605.06464)

The concrete repository checks are ours, derived from those risk themes and the
project's scientific context.

## Findings addressed

- A 3,988-line census module and other catch-all modules mixed unrelated work.
  Census, image/table measurement, sampling, spatial statistics, visualization,
  pattern learning/back-mapping, and safe I/O now have responsibility-based
  boundaries.
- Forwarding layers, fake aliases, duplicate definitions, dead biomolecule code,
  an undocumented tumor/stroma boundary module, Ray integration, placeholder
  plots, authorship markers, and unsupported speed claims were removed.
- Constant parameter suggestions, permissive extension guessing, unsafe pickle
  interchange, global RNG/logging/plot state, and swallowed backend failures now
  have behavior tests and explicit paths.
- Ruff owns formatting/lint/complexity, mypy owns typing, and pytest/coverage
  owns behavior. Direct ports of the pinned R MI algorithms have narrow,
  documented complexity exemptions.
- Runtime and optional dependencies are declared once. The public-tree,
  dependency, license, secret, CodeQL, build-artifact, and clean-wheel gates are
  automated.

## Remaining deliberate complexity

The two R MI ports remain near the structural review threshold because their
ordering is coupled directly to the pinned R algorithms. Their narrowly scoped
complexity exemptions are parity safeguards, not general exemptions for new
code. Pattern training, learning, and back-mapping were split once their
ownership boundaries were explicit.
