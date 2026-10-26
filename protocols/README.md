# Protocol files

The repository includes the public NightReID training-identity exclusion list
and the schema for a local animal split manifest. It does not include benchmark
images or the sealed LeopardID2022/WhaleSharkID test identities and filenames.

For an authorized sealed evaluation, keep the manifest outside the repository,
validate it against [`animal_split_manifest.schema.json`](animal_split_manifest.schema.json),
and materialize it with:

```bash
python tools/materialize_animal_split.py \
  --manifest /private/LeopardID2022_manifest.json \
  --source-root /data/LeopardID2022 \
  --output data/animals/leopardid2022
```

The tool writes `manifest_used.json` and `audit.json` beside the prepared
folders. The manifest itself remains local so that a sealed test split is not
published accidentally.
