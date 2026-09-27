# Full Run Report (Final)

## Status: SUBMISSION-READY

### Test Inference Metrics
- **S1 Entities Processed:** 1,732,544
- **Candidate Pairs Scored:** 78,345,365 (Avg 45.2 candidates per S1)
- **Total Matches Predicted:** 4,941,689
- **Empty match sets (singletons):** 200,335
- **Runtime:** 4361 seconds (~72 minutes)
- **Peak Memory:** 3.80 GiB

### Validation Results
The official validation script student_resource/utils/validate_submission.py was run with --check-ids enabled against the output files. 

**Result: PASS — no blocking issues found. Safe to submit.**

### Model Metrics (from 30k training sample)
- **Macro F0.5:** 0.8175
- **Macro Precision:** 0.8877
- **Macro Recall:** 0.7076
- **Blocking Recall:** 0.7605
- **Threshold:** 0.525
- **Total True Pairs (Train Set):** 10,403

### Pipeline Features Implemented
1. **Bounded Disk Index (disk_index.py)**: Uses char-trigrams (for typo OCR variants), token bigrams (for subset matches), phonetic blocks, and address strings. Implements rigid memory bounds while ensuring true matches aren't displaced by overly common tokens.
2. **Batched Feature Matrix (atch_features.py)**: Uses string-jaccard, tf-idf, jaro-winkler, and levenshtein distances processed efficiently in chunked Arrow arrays.
3. **Calibrated Threshold (calibrator.py)**: Automatic F0.5 threshold calibration via internal validation set.
4. **Checkpoint Resume (production.py)**: Complete scale-tested architecture that buffers JSON state chunks and yields valid TSVs.

The outputs matching_results.tsv and candidate_pairs.tsv in the output/ directory are generated and fully verified.
