# Public data policy

Only source code, documentation, and small synthetic parity fixtures may be
tracked. The audit rejects clinical/study directories, raw microscopy formats,
office documents, archives, pickle/joblib/R serialized objects, caches, compiled
files, and unexplained files over 1 MB.

Every file under `src/pyspace/data` is listed in `DATA_MANIFEST.json` with its
SHA-256 digest, byte size, provenance, license, and an explicit identifier-free
review. The current fixtures are synthetic; no tutorial census or MI output from
the private research tree was reused.

Run:

```bash
python scripts/check_public_tree.py
python scripts/write_data_manifest.py --check
python scripts/write_artifact_manifest.py --check
```
