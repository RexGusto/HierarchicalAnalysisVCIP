# Submission history

This repo backs five submissions:

- **CVGIP25, ACMMMW25 and ICME26** study how the backbone, its pretraining and the
  optimisation hyperparameters affect hierarchical fine-grained image recognition (FGIR).
  They use aircraft, CUB and cars.
- **ICME26** adds pseudo-hierarchies built without annotation (BPH).
- **CVPRW26** makes HPH the main method.
- **VCIP26** applies them to ultra-fine-grained leaf datasets as Hungarian Pseudo-Hierarchies
  (HPH).

Manuscripts are in `drafts/`, reviews in `reviews/`.

| # | Venue | Title | Outcome | Reviews | Manuscript |
|---|---|---|---|---|---|
| 1 | CVGIP 2025, extended abstract | On the Importance of Backbone, Pretraining and Hyperparameter Selection for Hierarchical Fine-Grained Image Recognition | not recorded | none | `drafts/CVGIP25/` |
| 2 | ACM MM 2025 workshop AOCV, submission 3 (submitted 12 Jul 2025) | Rethinking Hierarchical Fine-Grained Image Recognition: Backbone, Pretraining, and Hyperparameters Importance | rejected, 2 Aug 2025 | `reviews/review_hierarchical_acmmmw25.md` | `drafts/ACMMMW25/` |
| 3 | ICME 2026 | Pseudo-Hierarchies for Robust Backbone Selection in Fine-Grained Image Recognition | rejected: 2 borderline, 3 weak reject | `reviews/review_hierarchical_icme26.md` | `drafts/ICME26/` |
| 4 | CVPR 2026 workshop on efficiency, paper 10 | Pseudo-Hierarchies Improve Optimization Hyperparameter Robustness in Fine-Grained Image Recognition | desk rejected for workshop fit, no reviews | none | `drafts/CVPRW26/` |
| 5 | VCIP 2026 | Hungarian Pseudo-Hierarchies for Coarse-to-Fine Ultra Fine-Grained Image Recognition | accepted | `reviews/review_hierarchical_vcip26.md` | `drafts/VCIP26/` |

## 1 CVGIP25

The paper benchmarks 21 pretrained ResNet-50 and ViT-B backbones on aircraft, CUB and cars,
each with a real hierarchy. Each backbone is fine-tuned or frozen over 10 learning rates
(serials 23 and 24). The paper introduces the accuracy-deviation (ADA) ratio: 100 × standard
deviation / mean accuracy over the learning-rate sweep. The committed LaTeX source is a
later revision titled "Revisiting the Backbone, Pretraining and Transferability for
Hierarchical Fine-Grained Image Recognition". It adds feature-transferability metrics
(serials 25–28).

## 2 ACMMMW25

This is the same study in the ACM format. Both reviewers rejected it: strong reject with
confidence 3, and clear reject with confidence 4. Their points:

- There is no new method. The ADA ratio is a simple robustness metric (standard deviation
  over mean).
- The paper reads as a survey-style benchmark. Its conclusions are descriptive and not
  analysed in depth.
- The workshop wants methods for ambiguous-object analysis, and the paper is outside that
  scope.
- The contribution is small for the paper's length.

## 3 ICME26

The paper adds Bipartite Pseudo-Hierarchy (BPH) to the backbone study. BPH is evaluated on
ResNet IG-1B on aircraft and CUB (serial 39):
- the pseudo-hierarchy is built from IG-1B or SigLIP-v2 features;
- six cluster ratios are tested, from 10 to 70.

Of five reviewers, two scored borderline and three weak reject.

The reviewers credited these strengths:
- The main findings are useful and credible: ResNet is robust, ViT is sensitive to
  hyperparameters, a hierarchy lowers the ADA ratio, and pseudo-hierarchies give the same
  benefit without annotation.
- The ADA ratio is useful.

They raised these weaknesses:
- The ADA ratio is the coefficient of variation. Three reviewers raised this.
- The bipartite clustering is under-specified and of limited novelty.
- Only the ADA ratio is shown, and results are truncated.
- The training regimes are narrow: one batch size, generic augmentation, and only the
  optimiser and learning rate swept.
- The paper is missing references, standard hierarchical metrics, modern hierarchical
  baselines and public code.

## 4 CVPRW26

The paper names the method Hungarian Pseudo-Hierarchy (HPH). It reports robustness as the
coefficient of variation, which replaces the ADA ratio.

It adds ablations on:
- cluster ratio (serial 40);
- clustering algorithm: k-means and agglomerative (serials 52 and 53);
- SoyLocal (serials 39 and 32);
- DeiT-3 backbone size (serial 70);
- 21 feature extractors (serial 51).

The efficiency workshop desk-rejected it for fit, without reviews.

## 5 VCIP26

The paper narrows to ultra-fine-grained recognition on leaves:
- SoyLocal with 21 backbones and 7 cluster ratios (serials 54 and 55).
- Cotton and SoyAgeing-R1 with DeiT-B and ViT-B (serials 83 and 86).

It was accepted. The reviewers asked for these changes. The camera-ready version made some
of them, listed in the review file's slide 2:
- **Low-resource conclusions.** They hold only for the tested datasets.
- **Loss and levels.** The paper should explain how the per-level losses are balanced, and
  how poor pretrained features and deeper hierarchies are handled.
- **Missing comparisons.** There are no baselines against centroid, hierarchical or random
  grouping, and no ablation of Hungarian matching.
- **Construction procedure.** The padding and reassignment steps need pseudocode.
- **Statistics.** Results should report the mean and standard deviation over seeds.
- **Cost.** The paper should report construction time, FLOPs and memory overhead.
- **Other material.** Reviewers asked for a limitations section and for datasets beyond
  agriculture.

The camera-ready changes are listed in `drafts/README.md`.
