# Result bundles

A `.pyspace` result is a directory with:

- `manifest.json`: schema name, integer schema version, and tagged JSON payload;
- `tables/*.csv`: DataFrame values;
- `arrays.npz`: optional non-object NumPy arrays.

Version 1 loaders validate the schema and version, prevent bundle-member path
traversal, and call NumPy with `allow_pickle=False`. Writers reject object-dtype
arrays and existing destinations. Dataclass type names are descriptive metadata,
not import or execution instructions.

CSV does not preserve every pandas extension dtype. Consumers that require
categorical or nullable dtype identity should record and validate it separately;
scientific values and column order are preserved by the current schema.
