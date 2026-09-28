# ACMMM 2025 AOCV workshop reviews: On the Importance of Backbone, Pretraining, and Hyperparameter Selection for Hierarchical Fine-Grained Image Recognition

OpenReview page (submission 3), pasted verbatim. Decision: reject.

```text
On the Importance of Backbone, Pretraining, and Hyperparameter Selection for Hierarchical Fine-Grained Image Recognition
Download PDF
Augusto Christian Surya, Edwin Arkel Rios, Bo-Cheng Lai, Min-Chun Hu 
12 Jul 2025 (modified: 02 Aug 2025)
Submitted to ACMMM-AOCV 2025
AOCV, Area Chairs, Reviewers, Authors
Revisions
BibTeX
CC BY 4.0
Keywords: Fine-Grained Object Categorization, Hierarchical Classification, Backbone Architecture, Transfer Learning, Model Robustness
TL;DR: This work benchmarks 21 pretrained CNN and ViT models for hierarchical FGIR and shows that certain backbone consistently perform best, offering practical guidance for future model selection in this task.
Abstract:
Fine-grained image recognition (FGIR) involves distinguishing visually similar subcategories often organized in hierarchical structures. While prior work has explored hierarchical FGIR, the impact of backbone architecture and pretraining remains underexplored. In this work, we benchmark 21 pretrained convolutional and transformer models under both fine-tuned and frozen settings. Our study highlights backbone selection as a key factor in hierarchical FGIR. Specifically, certain backbones such as MoCov3 and SwAV on ResNet consistently yield better results in terms of mean accuracy and standard deviation. To support hyperparameter robustness evaluation, we introduce the ADA ratio, a simple yet effective metric that jointly considers accuracy and standard deviation across runs, enabling a more reliable evaluation of model performance. Overall, our analysis provides a practical guidance for future work in hierarchical FGIR.

Submission Number: 3
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
3 / 3 replies shown
Paper Decision
Decisionby Program Chairs02 Aug 2025, 14:15 (modified: 03 Aug 2025, 00:55)Program Chairs, Area Chairs, Reviewers, AuthorsRevisions
Decision: Reject
Comment:
The paper presents an empirical evaluation of pretrained CNN and transformer backbones for hierarchical fine-grained image recognition (FGIR). It examines the impact of backbone architecture, pretraining strategies, and hyperparameter choices on classification outcomes. Although the analysis is comprehensive, the study reads more like a survey-style benchmark and shows limited alignment with the goals of the AOCV Workshop.

Backbone Benchmarking for Hierarchical FGIR Lacks Methodological Novelty and AOCV Fit
Official Reviewby Reviewer sycp01 Aug 2025, 18:09 (modified: 03 Aug 2025, 00:47)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer sycp, AuthorsRevisions
Review:
Summary:
This paper benchmarks the influence of backbone choice, pretraining strategy, and hyperparameters on hierarchical fine-grained image recognition (FGIR). While the evaluation is comprehensive, the paper lacks methodological novelty and is primarily empirical. It resembles a survey-style benchmark and is misaligned with the core innovation focus of the AOCV Workshop.

Weaknesses
1. Lack of Technical Novelty

No new model, loss function, or training algorithm is proposed.
ADA ratio is a simple robustness metric (std/mean), not sufficient to count as a meaningful contribution.
The analysis confirms known results (e.g., MoCo/SwAV perform well on FGIR) rather than offering fresh insight or innovation.
2. Benchmarking Overload, But Little Insight

The paper benchmarks 21 models across multiple hyperparameters and datasets, but the conclusions are mostly descriptive (e.g., "ResNet works better for machines").
Claims like "SigLIP outperforms other ViTs" are interesting but not deeply analyzed or theoretically supported.
3. Weak Fit to AOCV Workshop

The workshop theme emphasizes ambiguous object analysis and novel algorithms for real-world recognition challenges.
This submission lacks any method for resolving ambiguity, learning hierarchical structure, or improving robustness beyond exhaustive empirical tuning.
4. Poor Contribution-to-Length Ratio

A long paper (with 5 pages of references and plots) that could be summarized in a blog post.
Repetition of baseline statistics without strong interpretability or actionable conclusions for model design.
Suggestions for Future Work
Introduce a new method for backbone-aware dynamic hierarchy learning or task-specific adaptation.
Propose a learned metric for transferability tailored to hierarchical FGIR, rather than relying on ADA or silhouette coefficients.
Include qualitative or real-world analysis (e.g., medical or industrial FGIR cases where ambiguity and hierarchy interact).
Despite being a useful benchmarking study, this paper lacks technical depth, originality, and alignment with AOCV 2025 goals. It may be more suitable for a dataset/model report track or as an arXiv resource, but does not meet the bar for acceptance at a top workshop.

Rating: 2: Strong rejection
Confidence: 3: The reviewer is fairly confident that the evaluation is correct
Empirical Benchmarking of FGIR Backbones with Limited Novelty
Official Reviewby Reviewer r4n901 Aug 2025, 15:44 (modified: 03 Aug 2025, 00:47)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer r4n9, AuthorsRevisions
Review:
Overview: This paper presents a large-scale empirical evaluation of 21 pretrained CNN and transformer backbones for hierarchical fine-grained image recognition (FGIR). The authors explore how backbone architecture, pretraining strategy, and hyperparameters affect classification performance across three benchmarks (FGVC Aircraft, CUB-200-2011, and Stanford Cars), and propose a metric called ADA ratio to assess accuracy and robustness.
Strengths:
The paper provides a wide and diverse set of evaluation to differnt architectures, covering fully-supervised and self-supervised models.
The ADA ratio, despite being simple, is a reasonable metric to balance the accuracy and robustness.
Weaknesses:
The paper does not provide a novel method, the main contribution is the evaluation of the effect of the backbone choosing, hyperparameters, pretraining strategies to wAP.
This look like a "survey" of the effect of backbone, pretraining, and hyperparameters on FGIR, instead of a paper which proposes an innovative method.
Because the main goal of the AOCV workshop is to advance multimedia research in ambiguous object analysis, I think this does not match with the main contribution of the paper.
Rating: 3: Clear rejection
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct

```
