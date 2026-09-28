# ICME 2026 reviews: Pseudo-Hierarchies for Robust Backbone Selection in Fine-Grained Image Recognition

Converted from `reviews/originals/ICME26 Paper Reviews - 23 Feb 2026.pptx`.

## Slide 1: ICME26 Paper Reviews

- 23 Feb 2026

## Slide 2: Review results

- Limited but convincing or lacking in some aspects for validation and reproducibility
- Clear enough presentation
- Some missing references
- 2 borderlines and 3 weak rejects

## Slide 3: Overall Strengths

- Main findings proves useful and credible
  - Resnet more robust, ViT sensitive to hyperparameter, hierarchy reduces ADA, pseudo-hierarchies can confer similar robustness benefits without annotation
- ADA ratio metric is useful to assess both performance and robustness
- Addresses an important and practical question how backbone choice and hierarchical supervision interact to affect robustness in FGIR

## Slide 4: Overall Weaknesses

- ADA ratio itself isn’t novel (identical to Coefficient of Variation)
- The Bipartite Clustering (BPH) is not explained in detail and limited in novelty
- Truncated results, only ADA is shown
- Limited training regimes (only 1 batch size, and generic augmentation; Only 2 hyperparams were adjusted extensively, optimizer and LR)

## Slide 5: Reviewer 1

- Less Confident
- Strength:
- Weakness:
  - CNN only use Resnet50 variants, ViT only use ViT-B variants. Insufficient coverage of backbone networks
  - Only two types of hyperparameters (optimizer and learning rate) were adjusted, while other hyperparameters that significantly affect the performance of FGIR—such as batch size, weight decay coefficient, data augmentation strategy, and number of training epochs—were not included.

## Slide 6: Reviewer 2

- Less Confident
- Strength:
  - Provides a comprehensive analysis of architectures under hierarchical constraints
- Weakness:
  - 'ADA ratio' is mathematically identical to the standard Coefficient of Variation
  - Presentation issues: Figure 3 are illegible due to overlap, Reference [24] is incomplete

## Slide 7: Reviewer 3

- Less Confident
- Strength:
  - Addresses a practical and underexplored problem in hierarchical FGIR, namely the interaction between backbone selection, hyperparameter robustness, and hierarchical supervision.
  - ADA ratio provides a useful and intuitive metric to jointly assess performance and robustness, which is valuable for practical deployment.
- Weakness:
  - The proposed pseudo-hierarchy method is relatively straightforward, and its novelty compared to existing clustering-based hierarchy construction approaches is incremental.
  - Some conclusions (e.g., backbone–dataset alignment and ViT behavior) rely mainly on post-hoc empirical observations, without deeper mechanistic analysis.

## Slide 8: Reviewer 4

- Confident
- Strength:
  - Main findings proves useful and credible
  - Addresses an important and practical question how backbone choice and hierarchical supervision interact to affect robustness in FGIR
- Weakness:
  - Missing references
  - Methodological novelty is minor (ADA same as Coefficient of Variation; BPH is a straightforward, under-specified clustering approach)
  - Truncated results (only show ADA)
  - Hierarchical loss design is not fully described
  - Standard hierarchical metrics are missing
  - Comparisons to modern hierarchical baselines are absen

## Slide 9: Reviewer 5

- Less Confident
- Strength:
  - Offers actionable guidance for practitioners: discriminative ResNet variants are robust baselines; hierarchical labels reduce hyperparameter sensitivity; pseudo-hierarchies can confer robustness without manual labels.
- Weakness:
  - ADA ratio is Coefficient of Variation
  - Only a single batch size and generic augmentations are used across diverse backbones; this may disproportionately disadvantage some models (e.g., ViTs)
  - Some figures/tables suffer from partial captions or truncation; the pseudo-hierarchy algorithm details are too brief for replication.
  - Methodology lacks rigor in certain parts and the novelty is limited
  - Absence of publicly available code
