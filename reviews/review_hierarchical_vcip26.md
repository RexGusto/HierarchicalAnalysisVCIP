# VCIP 2026 reviews: Hungarian Pseudo-Hierarchies for Coarse-to-Fine Ultra Fine-Grained Image Recognition

Converted from `reviews/originals/VCIP paper review.pptx`.

## Slide 1: Hierarchical Paper Review

- 28 Apr 2026

## Slide 2: VCIP Reviews (Summarized)

- Small additions:
  - Other clustering algorithm (No space for figure)
  - Pseudo-hierarchy construction procedure
    - Explain padding + reassignment
    - Pseudo-code (No space for figure)
  - Mean/std across seeds (Across LRs, not seeds)
  - Parameters/FLOPs/time/memory/scalability (Partially, only added the parameters)
  - Limitation section for HPH (No Space for extra section)
  - Pseudo-hierarchy construction time
- Clarification:
  - Poor initialization of pretrained features
  - Loss clarification
  - Hyperparameter selection
  - Symbols, abbreviations, and module names are defined at their first occurrence
- Future Works:
  - Apply to dataset(s) other than leaves, such as Dogs or Pets
  - Scalability to larger numbers of fine-grained classes

## Slide 3: Review 1

- 3. [Justification of rating]
  - This paper presents a solution that groups semantically-related fine-grained classes together avoiding human hierarchical annotation. The conclusions on the low-resource classes are limited and are only valid for the tested datasets.
- 4. [Additional comments] Minor suggestions, questions, corrections, etc. that can help the authors improve the paper, if any.
  - Avoid sing bold in the text
- Nothing can be done for revision since the comment above needs to add new experiment/data

## Slide 4: Review 2

- 3. [Justification of rating]
  - 1) How does the proposed method handle poor initialization of pretrained features
  - 2) In Section 3B the author mentioned that a two-level hierarchy has been formed, what about larger dataset that requires larger number of levels has not been fully addressed.
  - 3) It's not fully explained how the classification losses are balanced between different levels.
  - 4) Might be worth discussing if the proposed method could be applied to other dataset not just agricultural.

## Slide 5: Review 2

- 1) How does the proposed method handle poor initialization of pretrained features
  - HPH doesn’t assume perfect representation
  - Even though, if representation captures class relationship poorly, the hierarchy is less meaningful
  - The hierarchy is used as a learning method to enable curriculum learning rather than making a “correct” hierarchy

## Slide 6: Review 2

- 2) In Section 3B the author mentioned that a two-level hierarchy has been formed, what about larger dataset that requires larger number of levels has not been fully addressed.
  - Requires new data/experiment, not applicable in this case?

## Slide 7: Review 2

- 3) It's not fully explained how the classification losses are balanced between different levels.
  - Add the L_hierarchy = L_lvl1 + L_lvl2 + L_final formula from Chang et al. paper

## Slide 8: Review 2

- 4) Might be worth discussing if the proposed method could be applied to other dataset not just agricultural.
  - Similar to 2nd comment, needs new data/experiment

## Slide 9: Review 3

- 3. [Justification of rating]
  - 1.The methodological novelty is moderate because the approach mainly combines established components: pretrained features, class centroids, distance-based grouping, Hungarian matching, and hierarchical multi-task classification.
  - 2.The paper does not directly compare HPH with conventional alternatives for automatic pseudo-hierarchy construction, such as centroid-based clustering or hierarchical clustering.
  - 3.The pseudo-hierarchy construction procedure is not fully specified, particularly the padding strategy for the distance matrix and the reassignment rule for classes matched to padded columns.
  - 4.The experiments report best Top-1 accuracy but do not provide repeated-run statistics, such as mean and standard deviation, which is especially relevant in extremely low-resource settings.
  - 5.The hyperparameter-selection procedure requires clarification. The paper tests seven cluster ratios on SoyLocal but evaluates only 70% and 90% on Cotton and SoyAgeing-R1, and the protocol for selecting optimizer and learning-rate settings is not fully detailed.
  - 6.No ablation is provided to separate the contribution of Hungarian matching from the contribution of adding generic pseudo-coarse hierarchical supervision.
  - 7.The paper does not report parameter overhead, FLOPs, pseudo-hierarchy construction time, training/inference time, memory usage, or scalability to larger class sets.

## Slide 10: Review 3

- 4. [Additional comments] Minor suggestions, questions, corrections, etc. that can help the authors improve the paper, if any.
  - 1.Please provide a concise pseudocode block for the complete HPH construction procedure, including class-centroid computation, distance-matrix construction, padding, Hungarian matching, and reassignment of classes matched to padded columns.
  - 2.Please clarify the rationale for using Hungarian matching rather than standard clustering or hierarchical clustering for pseudo-hierarchy generation, and discuss the expected effect of the matching constraint on group balance and semantic consistency.
  - 3.Please explicitly state whether the cluster-ratio selection and all training hyperparameters are determined using a validation set. This is particularly important because the paper reports the best-performing ratios on SoyLocal.
  - 4.Consider reporting mean and standard deviation over multiple random seeds for representative backbone models and cluster ratios, given the extremely limited number of training samples per class.
  - 5.Consider adding a compact ablation comparing HPH with random grouping and at least one conventional centroid-based clustering baseline. This would help isolate the contribution of the Hungarian matching design.
  - 6.Please report the computational overhead of HPH, including pseudo-hierarchy construction time and, where possible, training/inference cost relative to the non-hierarchical baseline.
  - 7.Please ensure that all symbols, abbreviations, and module names are defined at their first occurrence, particularly those related to the hierarchical classification framework and the pseudo-label construction process.
  - 8.The paper could benefit from a brief discussion of limitations, including sensitivity to pretrained representations, cluster-ratio choices, potential instability in very small-sample regimes, and scalability to larger numbers of fine-grained classes.
