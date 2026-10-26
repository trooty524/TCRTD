# Protocol files

This directory contains the public NightReID training-identity exclusion list and
the schema for local animal split manifests. Benchmark images and the sealed
LeopardID2022/WhaleSharkID test identities and filenames stay outside the release.

Keep a sealed manifest outside the repository, validate it against
[`animal_split_manifest.schema.json`](animal_split_manifest.schema.json), and
materialize it with:

```bash
python tools/materialize_animal_split.py \
  --manifest /private/LeopardID2022_manifest.json \
  --source-root /data/LeopardID2022 \
  --output data/animals/leopardid2022
```

The tool writes `manifest_used.json` and `audit.json` beside the prepared folders.
The source manifest remains local.
