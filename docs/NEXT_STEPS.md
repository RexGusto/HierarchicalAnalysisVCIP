# Next steps

Open items only. Items marked "decision" need the author.

## VCIP26 camera-ready corrections (decision)

`python -m codebase.verify.verify_paper_numbers` fails on the numbers below. Items are in
reading order through `drafts/VCIP26/latex/camera_ready.tex`, with source line numbers. Each
item gives the location, the change and its reason, then the text before and after.

Values follow one rule:
- a printed value is the stored value rounded once, half-up, to the decimals the paper uses
  (one);
- a derived value (a gain) is computed from the printed operands when the paper prints them,
  and otherwise from the stored values, rounded once.

These are proposals; the source is unchanged.

1. **Author block, l.24.** Change: the camera-ready still reads "Anonymous Authors". Kevin
   Tandian is the third author, and Augusto and Edwin share first authorship. Affiliations
   follow the CVGIP25 camera-ready.
   - Before: `\author{Anonymous Authors}`
   - After:
     ```latex
     \author{\IEEEauthorblockN{Augusto Christian Surya\textsuperscript{1,*}, Edwin Arkel Rios\textsuperscript{2,*}, Kevin Tandian\textsuperscript{1}, Bo-Cheng Lai\textsuperscript{2}, Min-Chun Hu\textsuperscript{1}}
     \IEEEauthorblockA{\textsuperscript{1}Department of Computer Science, National Tsing Hua University, Hsinchu, Taiwan\\
     \textsuperscript{2}Department of Electronics Engineering, National Yang Ming Chiao Tung University, Hsinchu, Taiwan\\
     \textsuperscript{*}Equal contribution}}
     ```
   - Kevin Tandian's affiliation is given as the NTHU Department of Computer Science; confirm
     the department. Add e-mails only if VCIP asks for them.
2. **Abstract, l.29.** Change 22.3 to 23.7 and 18.16 to 18.1. Reason: the largest gain is
   SparK, 56.83 − 33.17 = 23.67. The SoyLocal gain over CSDNet from the printed table values
   is 64.3 − 46.2 = 18.1.
   - Before: "achieving absolute gains of up to **22.3%** across backbone. Furthermore, HPH
     consistently improves performance, achieving improvements of up to **18.16%** on
     datasets containing only 3–5 training images per class"
   - After: "achieving absolute gains of up to **23.7%** across backbones. Furthermore, HPH
     consistently improves performance, achieving improvements of up to **18.1%** on datasets
     containing only 3–5 training images per class"
3. **Contributions, l.48.** Change: the same two numbers as item 2, for the same reason.
   - Before: "achieving gains of up to **22.3%** over non-hierarchical baselines and up to
     **18.16%**"
   - After: "achieving gains of up to **23.7%** over non-hierarchical baselines and up to
     **18.1%**"
4. **Parameter overhead, l.170.** Change: name the backbone. Reason: 86.8027M is the DeiT-3
   count; DeiT, the model the results headline, has 86.7851M. The two steps (0.012M per 10%,
   0.096M from 10% to 90%) are correct for either.
   - Before: "the SoyLocal model contains 86.8027M trainable parameters at a cluster ratio of
     10%."
   - After: "the DeiT-3 model contains 86.8027M trainable parameters at a cluster ratio of
     10%."
5. **Results, l.177.** Change three statements. Reasons:
   - three ResNets lose maximum accuracy with HPH: Gluon −2.00, IN21k-MIIL −1.83, MoCo v3
     −1.17; three are unchanged;
   - the SparK gain is 23.7;
   - DeiT-3 IN-1k (+13.3) and ViT MoCo v3 (+12.8) gain more than DeiT (+12.5).
   - Before: "Despite the performance differences between architectures, HPH consistently
     improves the maximum accuracy of CNN and transformer backbones. Among the ResNet models,
     ResNet SparK achieves the largest improvement, increasing the maximum accuracy by
     **22.3%** over the non-hierarchical baseline. For transformer models, DeiT obtains the
     largest gain of **12.5%** to max accuracy over the non-hierarchical baseline."
   - After: "Despite the performance differences between architectures, HPH improves the
     maximum accuracy of 15 of the 21 backbones; three ResNets lose at most 2.0 points. Among
     the ResNet models, ResNet SparK achieves the largest improvement, increasing the maximum
     accuracy by **23.7%** over the non-hierarchical baseline. Among transformers, DeiT-3
     (IN-1k) gains the most, **13.3%**, and DeiT gains **12.5%**."
6. **Table `tab_compare_othersoy`, l.193 and l.196.** Change the ViT-B SoyLocal cell, and
   both HPH SoyAgeing-R1 cells. Reasons:
   - ViT-B's SoyLocal maximum is 58.83;
   - the SoyAgeing-R1 cells come from serial-86 runs trained on the full 4,944-image
     training set. At the 5-per-class split the text describes (984 images), DeiT reaches
     86.16 and ViT-B 84.24. DeiT stays best in every column.
   - Before: "HPH (Ours) & 68.3 & 86.3 & 58.7" and "HPH (Ours) & 72.5 & 89.6 & 64.3"
   - After: "HPH (Ours) & 68.3 & 84.2 & 58.8" and "HPH (Ours) & 72.5 & 86.2 & 64.3"
7. **Dataset sizes, l.206.** Decision: 990 or 984. Reason: the serial-86 low-resource runs
   logged 984 training images; 990 is 198 classes × 5. Keep 990 if it is the dataset's
   official split size.
   - Before: "SoyAgeing-R1 contains 198 classes with 990 training images"
   - After, if the runs' count is used: "SoyAgeing-R1 contains 198 classes with 984 training
     images"
8. **Comparison with CSDNet, l.208.** Change: 18.16 to 18.1, 14.60 to 14.6, and 14.45 to 11.0.
   Reason: one decimal like the table; SoyAgeing-R1 follows item 6 (86.2 − 75.2).
   - Before: "DeiT+HPH improves accuracy by 18.16% on SoyLocal, 14.60% on Cotton, and 14.45%
     on SoyAgeing-R1."
   - After: "DeiT+HPH improves accuracy by 18.1% on SoyLocal, 14.6% on Cotton, and 11.0% on
     SoyAgeing-R1."
9. **Conclusion, l.215.** Change: the same three numbers as item 8, for the same reason.
   - Before: "improving accuracy over the ViT+CSDNet SoTA by **14.60%** on Cotton,
     **14.45%** on SoyAgeing-R1, and **18.16%** on SoyLocal."
   - After: "improving accuracy over the ViT+CSDNet SoTA by **14.6%** on Cotton, **11.0%** on
     SoyAgeing-R1, and **18.1%** on SoyLocal."
10. **Figure `compare_soylocal.png`.** Decision: regenerate from serials 54 and 55. Reason:
    the figure was plotted from the serial 39 (HPH) and 32 (baseline) SoyLocal rows, while the
    text and the other two figures use 54 and 55. The two sets differ slightly: the ratio-10
    lower whisker is 32.2 in the figure and about 33.7 from 54/55. The baseline box is the
    same in both.
    - Before: `plotter.sh` plots `--keep_serials 39 32 --input_file data/hierarchical_pseudo.csv`
    - After: `--keep_serials 55 54 --input_file data/hierarchical_soylocal.csv`, then copy to
      `drafts/VCIP26/latex/figures/`

Counts: 10 items. 7 changes are ready to apply (1–6 and 8–9). 3 need a decision: item 7,
item 10, and Kevin Tandian's department in item 1.

## Data and checkpoints

- The checkpoint store's README names serial 84 as the baseline and 85 as the
  pseudo-hierarchy. The checkpoint configs and `data/cotton_seeds.csv` say the opposite.
- The CVPRW26 submission date is not recorded in the repo.
- `results_all/` folders with no recorded command: `diff_tables/`, `soylocal_acc/`,
  `soylocal_hyperparam/`, `val_acc_level1/`, `extracted_text/` and `lr_scripts/`.
- `codebase/analyses/summarize_cost.py` defaults to `--host server-3090`. No serial 23 or 24
  run was logged from that host, so it summarises nothing unless `--host` is given. It is not
  in `plotter.sh`.
- The checkpoint release (repository names, which groups) waits on the FGIR-Backbones Model
  Zoo session's proposal, `/mnt/hdd10tb/edwin/hier_zoo/PROPOSAL.md`.
