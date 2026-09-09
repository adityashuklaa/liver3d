# Data card

One card per dataset actually used. The supplied project documents mention LiTS
and CHAOS as candidates but do not name the dataset that produced any result, so
this card starts empty on purpose.

## Source and authorization

| Field | Value |
|---|---|
| Dataset name and version | TO_CONFIRM |
| Provider / URL | TO_CONFIRM |
| License and permitted use | TO_CONFIRM |
| Access approval (who, when) | TO_CONFIRM |
| Ethics / institutional approval reference | TO_CONFIRM (required for hospital data) |
| De-identification method | TO_CONFIRM |
| Storage location and access group | TO_CONFIRM |
| Retention and deletion date | TO_CONFIRM |

## Content

| Field | Value |
|---|---|
| Patients / studies / volumes | TO_CONFIRM |
| Modality and contrast phase(s) | CT, TO_CONFIRM |
| Scanners / sites represented | TO_CONFIRM |
| Voxel spacing distribution (min / median / max) | TO_CONFIRM |
| Slice thickness range | TO_CONFIRM |
| Volume dimensions range | TO_CONFIRM |
| Known exclusions | TO_CONFIRM |

## Labels

| Field | Value |
|---|---|
| Label source (expert, consensus, semi-automatic) | TO_CONFIRM |
| Class definition used here | liver parenchyma incl. intrahepatic lesions; background otherwise |
| Raw label values and mapping | e.g. LiTS 0 background, 1 liver, 2 tumour -> merged to 1 (`data.merge_labels_above_zero: true`) |
| Inter-observer variability information | TO_CONFIRM |

## Split

| Field | Value |
|---|---|
| Manifest | `data_manifests/split_v1.csv` |
| Manifest SHA-256 | TO_CONFIRM (`data_manifests/split_v1_meta.json`) |
| Ratios and seed | TO_CONFIRM |
| Grouping | patient level; one case per patient unless `--patient-pattern` was used |
| Author and date | TO_CONFIRM |
| Test set opened on | TO_CONFIRM (after the pipeline was frozen) |

## Quality gates run

Enforced by `src/data/validate.py` when the cache is built:

- image and label share shape and affine (hard fail),
- label non-empty and values map to documented classes,
- liver volume inside the plausible range (warning otherwise),
- spacing inside the validated range (warning otherwise),
- volume finite and 3D, minimum size respected (hard fail).

Record here: cases that failed, cases excluded, and the rule that excluded them.

| Case | Gate | Outcome | Action |
|---|---|---|---|
| TO_CONFIRM | | | |

## Privacy check

- [ ] No direct identifiers in filenames, headers, logs, screenshots or exports
- [ ] Case ids are anonymous and stable
- [ ] Cache, outputs and API work directories are inside the approved storage
- [ ] Retention and deletion procedure documented and scheduled
