# TCRTD: Re-Ranking at Training Time

**Topology-Consistent Relation Distillation for plain-cosine re-identification**

[![Status](https://img.shields.io/badge/status-under%20review-blue)](https://github.com/trooty524/TCRTD)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)

This repository contains the research implementation of **TCRTD**, a training-time
relation-distillation method for image re-identification (ReID).

> **Re-Ranking at Training Time: Topology-Consistent Relation Distillation for Plain-Cosine Re-Identification**
> Yu Ding and Xiuhui Wang, *IEEE Transactions on Multimedia*, under review.

## What problem does TCRTD solve?

k-reciprocal re-ranking improves retrieval by using query-gallery neighborhood
structure. Its deployment cost is substantial: a graph must be built for every
query, and a score depends on the current gallery.

TCRTD transfers this relation signal into the encoder during training:

1. an EMA teacher encodes the full training set;
2. a reciprocal-neighborhood graph produces a topology-aware target;
3. the student matches the target with a topology KL loss;
4. an AP-swap loss emphasizes pair swaps that change list-level AP;
5. the teacher and graph are discarded after training.

At deployment, the model uses its original image-wise descriptor and plain cosine
matching. No gallery graph and no query-time re-ranking are required.

![TCRTD training pipeline](assets/pipeline.png)

The implementation includes two relation semantics:

- **junk-aware**: removes same-identity, same-camera images ignored by the standard
  person-ReID evaluator;
- **legacy**: keeps all same-identity candidates and is intended for datasets
  without stable camera labels, including the animal benchmarks.

The fixed-support projection used in the paper is implemented as
<code>--relation fixed_support</code>.

## Repository map

| Path | Role |
| --- | --- |
| <code>tcrtd/relation.py</code> | EMA teacher memory, reciprocal graph, relation targets, topology KL and AP-swap |
| <code>tcrtd/model.py</code> | DINOv2-S/14, DINOv2-B/14, ResNet-50 and ResNet-101 |
| <code>tcrtd/losses.py</code> | CE companions, soft/hard triplet and Smooth-AP |
| <code>tcrtd/data.py</code> | Image-folder datasets, camera parsing, PK sampler and transforms |
| <code>tcrtd/metrics.py</code> | mAP, Rank-1, junk handling, fixed k-reciprocal re-ranking and bootstrap |
| <code>train.py</code> | M0, TCRTD, Smooth-AP and Smooth-AP + TCRTD |
| <code>evaluate.py</code> | Plain-cosine evaluation and per-query outputs |
| <code>tools/prepare_data.py</code> | Person data preparation and content-overlap audit |
| <code>tools/paired_bootstrap.py</code> | Paired seed comparison with identity-block intervals |
| <code>scripts/run_dinov2.sh</code> | DINOv2 person-ReID matrix and relation ablations |
| <code>scripts/run_resnet.sh</code> | ResNet-50/101 person-ReID matrix |
| <code>tests/test_tcrtd.py</code> | AP-swap and theorem/metric tests |

## Installation

~~~bash
git clone https://github.com/trooty524/TCRTD.git
cd TCRTD
conda create -n tcrtd python=3.10 -y
conda activate tcrtd
pip install -r requirements.txt
~~~

The implementation has been exercised with Python 3.10, PyTorch 2.11, torchvision
0.26 and CUDA 12.8. A CUDA GPU is required. DINOv2-B/14 uses approximately 25
GB with batch 128 at 448 x 224; the reported DINOv2 runs used one RTX 4080 SUPER
with 32 GB memory.

Pretrained weights are downloaded automatically unless <code>--weights</code> points
to a local file:

- DINOv2-S/14 and B/14: official DINOv2 checkpoints;
- ResNet-50 and ResNet-101: torchvision ImageNet-1K V1 checkpoints.

## Data layout

All commands use:

~~~text
data/<dataset>/
  train/<identity>/<image>
  test/
    query/<identity>/<image>
    gallery/<identity>/<image>
~~~

Images are not redistributed by this repository. Download them from the official
source, check the dataset license, and prepare a local copy.

### Person-ReID datasets

| Dataset | Official source | Paper | Protocol and prepared counts |
| --- | --- | --- | --- |
| [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html) | Project page | [Benchmark paper](https://doi.org/10.1109/ICCV.2015.133) | Camera-aware junk rule; 751 train IDs / 12,936 train images; 3,368 query / 19,732 gallery |
| [MSMT17](https://www.pkuvmc.com/dataset.html) | PKU VMC dataset page | [Benchmark paper](https://doi.org/10.1109/TPAMI.2018.2867508) | Complete official test with content-audited train; 1,041 train IDs / 32,391 train images; 11,659 query / 82,161 gallery |
| [NightReID](https://github.com/msm8976/NightReID) | Official repository | [AAAI paper](https://doi.org/10.1609/AAAI.V39I10.33142) | Camera-aware protocol; 491 train IDs / 15,154 train images; 1,394 query / 26,111 gallery |

For Market-1501 and NightReID, same-identity gallery images from the query
camera are ignored. Market identity <code>-1</code> junk images are ignored by
default. MSMT17 uses the complete official gallery and removes 230 training files
that duplicate test content at byte/pixel level.

Prepare the person datasets with:

~~~bash
python tools/prepare_data.py --dataset market1501 \
  --source /path/to/Market-1501-v15.09.15 --output data/market1501

python tools/prepare_data.py --dataset msmt17 \
  --source /path/to/MSMT17_V1 --output data/msmt17

python tools/prepare_data.py --dataset nightreid \
  --source /path/to/NightReID --output data/nightreid \
  --exclude-identities protocols/nightreid_excluded_identities.txt
~~~

Each command writes <code>audit.json</code> with counts, exclusions and removed
duplicates.

### Animal and wildlife datasets

The paper evaluates four datasets without stable camera labels. They use
identity-only retrieval, legacy relation targets and random PK sampling. Original
images and the sealed project test sets are not included in this repository.

| Dataset | Species / setting | Paper protocol and counts | Dataset source | Reference |
| --- | --- | --- | --- | --- |
| [ATRW](https://lila.science/datasets/atrw) | Amur tiger, terrestrial wildlife | 59 train IDs / 700 train images; 1,764-image all-versus-all test | [CVWC challenge page](https://cvwc2019.github.io/challenge.html) | [ACM MM 2020](https://doi.org/10.1145/3394171.3413569) |
| [SeaTurtleID2022](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022) | Mediterranean loggerhead sea turtles | 438 train IDs / 4,678 train images after one duplicate removal; temporal test, 2,629 query / 2,632 gallery | [WildlifeDatasets entry](https://wildlifedatasets.github.io/wildlife-datasets/datasets/#seaturtleid2022) | [WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Adam_SeaTurtleID2022_A_Long-Span_Dataset_for_Reliable_Sea_Turtle_Re-Identification_WACV_2024_paper.html) |
| [LeopardID2022](https://lila.science/datasets/leopard-id-2022/) | Camera-trap leopards | 291 train IDs / 4,865 train images; sealed test, 60 query / 1,180 gallery | [WildlifeDatasets entry](https://wildlifedatasets.github.io/wildlife-datasets/datasets/#leopardid2022) | [OpenAnimals](https://doi.org/10.1109/ICCV51701.2025.01333) |
| [WhaleSharkID](https://lila.science/datasets/whale-shark-id) | Whale sharks, underwater imagery | 372 train IDs / 5,564 train images; sealed test, 90 query / 1,067 gallery | [WildlifeDatasets entry](https://wildlifedatasets.github.io/wildlife-datasets/datasets/#whalesharkid) | [Dataset paper](https://www.int-res.com/abstracts/esr/v7/n1/p39-53/) |

The [WildlifeDatasets toolkit](https://github.com/WildlifeDatasets/wildlife-datasets)
is a common entry point for the animal datasets. Dataset-specific download terms
and licenses remain in force. Use the paper's manifests and split files to
reproduce the sealed LeopardID2022 and WhaleSharkID evaluations.

For an animal dataset, create:

~~~text
data/animals/<name>/
  train/<identity>/<image>
  test/<identity>/<image>       # all-versus-all test
~~~

A TCRTD animal run uses legacy relations and random sampling:

~~~bash
python train.py --train-root data/animals/atrw/train \
  --output runs/animals/atrw_b14_tcrtd_s13 --backbone dinov2_b14 \
  --objective tcrtd --relation legacy --sampler random --epochs 25 --seed 13

python evaluate.py --checkpoint runs/animals/atrw_b14_tcrtd_s13/model.pth \
  --query-root data/animals/atrw/test --all-vs-all --identity-only \
  --output runs/animals/atrw_b14_tcrtd_s13/test
~~~

For a separate query/gallery split, omit <code>--all-vs-all</code> and provide
<code>--gallery-root</code>; always pass <code>--identity-only</code>.

## Training objectives and options

<code>--objective</code> selects:

| Objective | Loss terms | Purpose |
| --- | --- | --- |
| <code>m0</code> | CE + backbone-family triplet | Matched baseline |
| <code>tcrtd</code> | M0 + topology KL + AP-swap | Main method |
| <code>smooth_ap</code> | M0 + Smooth-AP | AP-style comparator |
| <code>tcrtd_smooth_ap</code> | M0 + TCRTD + Smooth-AP | Combination |

Important TCRTD options:

| Option | Default | Meaning |
| --- | ---: | --- |
| <code>--candidates</code> | 64 | Candidate count K |
| <code>--reciprocal-k</code> | 10 | Reciprocal-neighborhood k |
| <code>--raw-weight</code> / <code>--jaccard-weight</code> | 0.5 / 0.5 | Teacher distance fusion |
| <code>--teacher-temperature</code> / <code>--student-temperature</code> | 0.1 / 0.1 | Relation temperatures |
| <code>--ema</code> | 0.995 | Teacher momentum |
| <code>--topology-weight</code> / <code>--swap-weight</code> | 0.5 / 0.3 | DINOv2 TCRTD weights |
| <code>--relation</code> | <code>junk_aware</code> | <code>junk_aware</code>, <code>legacy</code> or <code>fixed_support</code> |
| <code>--rank-policy</code> | <code>student_score</code> | Student-score or candidate-order AP-swap ranking |

DINOv2 uses 448 x 224 inputs and a 32 x 16 patch grid. ResNet-50/101 use
256 x 128 inputs and a 16 x 8 final feature map. The deployment descriptor is
the L2-normalized mean token/GAP feature.

## Minimal person-ReID example

~~~bash
python train.py --train-root data/market1501/train \
  --output runs/market1501/dinov2_b14/tcrtd_s13 \
  --backbone dinov2_b14 --objective tcrtd --epochs 6 --seed 13

python evaluate.py \
  --checkpoint runs/market1501/dinov2_b14/tcrtd_s13/model.pth \
  --query-root data/market1501/test/query \
  --gallery-root data/market1501/test/gallery \
  --output runs/market1501/dinov2_b14/tcrtd_s13/test --save-features
~~~

The evaluator writes <code>metrics.json</code>, <code>per_query.npz</code> and
optionally <code>features.npz</code>.

## Reproducing experiment families

~~~bash
bash scripts/run_dinov2.sh
bash scripts/run_resnet.sh
~~~

<code>run_dinov2.sh</code> covers the three person datasets, DINOv2-S/14 and
B/14, M0/TCRTD/Smooth-AP/Smooth-AP+TCRTD, and NightReID relation-policy checks.
<code>run_resnet.sh</code> covers the six fixed ResNet person cells with three
seeds.

For paired seed statistics:

~~~bash
python tools/paired_bootstrap.py \
  --baseline runs/market1501/dinov2_b14/m0_s13/test \
             runs/market1501/dinov2_b14/m0_s42/test \
             runs/market1501/dinov2_b14/m0_s2026/test \
  --method runs/market1501/dinov2_b14/tcrtd_s13/test \
          runs/market1501/dinov2_b14/tcrtd_s42/test \
          runs/market1501/dinov2_b14/tcrtd_s2026/test
~~~

## Reported results

Protocol labels are part of every number. Reduced/select, full-official person
tests and dataset-specific animal tests are separate estimands.

### DINOv2-B/14 stress and animal matrix

| Dataset | Protocol | M0 mAP | TCRTD mAP | Delta mAP |
| --- | --- | ---: | ---: | ---: |
| Market-1501 | reduced/select | 85.09 | 86.51 | +1.41 |
| MSMT17 | reduced/select | 63.97 | 66.58 | +2.62 |
| NightReID | legacy identity-only | 35.41 | 33.89 | -1.51 |
| SeaTurtleID2022 | temporal/open-set | 55.60 | 58.15 | +2.55 |
| WhaleSharkID | identity-only | 27.21 | 30.01 | +2.80 |
| LeopardID2022 | identity-only | 44.15 | 45.36 | +1.21 |
| ATRW | identity-only | 80.80 | 80.38 | -0.42 |

### DINOv2-B/14 full-official person matrix

| Dataset | Train IDs | Query / gallery | M0 mAP | TCRTD mAP | Delta mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market-1501 | 751 | 3,368 / 19,732 | 81.42 | 82.26 | +0.84 |
| MSMT17 | 1,041 | 11,659 / 82,161 | 69.89 | 72.51 | +2.62 |

These are paper snapshots. Exact reproduction requires the matching manifest,
split, budget and evaluator.

A separate fixed-budget ResNet verification gives:

| Dataset | Backbone | Epochs | M0 mAP | TCRTD mAP | Delta mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| Market-1501 | ResNet-101 | 70 | 79.68 | 79.96 | +0.28 |
| Market-1501 | ResNet-50 | 80 | 79.51 | 79.75 | +0.25 |
| MSMT17 | ResNet-101 | 80 | 54.08 | 54.08 | -0.00 |
| MSMT17 | ResNet-50 | 80 | 51.17 | 51.35 | +0.17 |
| NightReID | ResNet-101 | 55 | 30.60 | 30.55 | -0.04 |
| NightReID | ResNet-50 | 75 | 27.96 | 28.15 | +0.20 |

Mixed outcomes are retained; TCRTD is not expected to improve every
dataset/backbone combination.

## Tests

~~~bash
python -m pytest tests
~~~

The tests cover AP-swap weights, Theorem 1 and Corollary 1 identities,
junk-aware versus identity-only rules, stable ranking and paired bootstrap.

## Reproducibility and data policy

This repository contains source code, scripts, tests, a pipeline illustration and
the NightReID exclusion list. It does not contain benchmark images, pretrained or
fine-tuned weights, private/sealed test splits, experiment outputs, credentials
or environment files.

For exact reproduction, keep the dataset license/citation, prepared image-folder
manifest, content-overlap audit, split manifest, seed, epoch budget and command
line with the downloaded data.

## Citation

~~~bibtex
@article{ding2026tcrtd,
  title   = {Re-Ranking at Training Time: Topology-Consistent Relation Distillation for Plain-Cosine Re-Identification},
  author  = {Ding, Yu and Wang, Xiuhui},
  journal = {IEEE Transactions on Multimedia},
  note    = {under review},
  year    = {2026}
}
~~~

Dataset references: [Market-1501](https://doi.org/10.1109/ICCV.2015.133),
[MSMT17](https://doi.org/10.1109/TPAMI.2018.2867508),
[NightReID](https://doi.org/10.1609/AAAI.V39I10.33142),
[ATRW](https://doi.org/10.1145/3394171.3413569),
[SeaTurtleID2022](https://doi.org/10.1109/WACV57701.2024.00699),
[WildlifeDatasets](https://doi.org/10.1109/WACV57701.2024.00585), and
[OpenAnimals](https://doi.org/10.1109/ICCV51701.2025.01333).

## License

The code is released under the MIT License; see [LICENSE](LICENSE).
Benchmark images and annotations remain subject to their dataset owners' terms.
