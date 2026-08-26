# Third-party licenses

The base dependency declarations and their project-level licenses are:

| Dependency | Declared range | Project license |
| --- | --- | --- |
| matplotlib | `>=3.8` | Matplotlib license (PSF-style) |
| networkx | `>=3.2` | BSD-3-Clause |
| numpy | `>=1.26` | BSD-3-Clause |
| openpyxl | `>=3.1` | MIT |
| pandas | `>=2.0` | BSD-3-Clause |
| pillow | `>=10` | HPND |
| plotly | `>=5.20` | MIT |
| psutil | `>=5.9` | BSD-3-Clause |
| pyarrow | `>=15` | Apache-2.0 |
| scikit-learn | `>=1.3` | BSD-3-Clause |
| scipy | `>=1.11` | BSD-3-Clause |
| tifffile | `>=2023.0` | BSD-3-Clause |

These are dependency project licenses, not legal advice. Wheels can bundle
additional notices; inspect the exact resolved artifacts for a release. The
generated `DEPENDENCY_INVENTORY.json` records the installed release-candidate
versions and metadata.

## Optional community extra

The optional dependencies are not bundled into PySpace and are not installed by
default:

| Dependency | Declared range | Published open-source license |
| --- | --- | --- |
| igraph | `>=0.11` | GPL-2.0-or-later |
| leidenalg | `>=0.10.2` | GPL-3.0-or-later |
| Infomap | `>=2.9` | GPL-3.0-or-later (commercial licensing is also offered) |

Installing or redistributing this extra can add copyleft obligations. Review the
exact versions and their license texts for the intended distribution model.
