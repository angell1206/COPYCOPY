# AI-Based Help Line Triage — Dataset v2 notes

## What is in this pack
- `calls_master_240_v2.csv`: original 240 fictional examples with reviewed language-style metadata and one corrected contradictory-current-threat label.
- `calls_train_v2.csv`: 270 development examples (the 240-row development corpus plus 30 curated contrastive training examples).
- `calls_fresh_eval_36.csv`: 36 newly authored examples, 12 per label, reserved for one evaluation after model selection. Do not merge into training or use to tune parameters.
- `label_audit_log.csv`: the one confirmed label correction plus notes on the nine earlier model errors. The error notes do not automatically relabel those examples.
- `train_model_v2.ipynb`: training/cross-validation and one-time fresh evaluation notebook.

## Label rubric
- `PRIORITY_REVIEW`: transcript reports possible immediate/current danger, active serious violence, acute medical danger, imminent self-harm, a current forced-entry or weapon threat, or another situation that warrants prompt human assessment. An explicit current dangerous cue takes precedence over a vague or contradictory reassurance; the operator still verifies context.
- `SUPPORT_REVIEW`: a clear ongoing or past concern requiring guidance, support or reporting, without transcript evidence that establishes immediate physical danger (e.g. recurring harassment, online blackmail, housing dispute, past threat reporting, stalking while currently in a safer place).
- `UNCLEAR_REVIEW`: not enough context to classify the incident, or the mention is hypothetical, quoted, fictional, academic, news-related, historical/resolved, explicitly negated, or general information only. This label never means the caller is safe. Human follow-up is required when information is incomplete or contradictory.

These are prototype workflow labels, not validated clinical, policing, or emergency-risk categories. Borderline cases should be reviewed consistently by more than one teammate.

## Preprocessing and metadata
- `transcript` preserves original text. Do not drop negation words such as `nahi`, `not`, or `never`.
- `label_reason`, `incident_type`, `language_style`, IDs and group fields are metadata for auditing only. The first text classifier should use `transcript` only to avoid label leakage.
- `language_style` was reassigned using a transparent script/word-cue heuristic, not a human language-identification model. Review this metadata before making any language-stratified claim.
- `scenario_family` consolidates numbered variations sharing the same scenario-group stem. It is useful for grouped cross-validation, but it cannot detect every semantically related story across differently named families.

## Evaluation protocol
- Previous random and grouped test results have already been seen during model development. They are development diagnostics, not an untouched final benchmark.
- `calls_fresh_eval_36.csv` was created after reviewing those errors. Keep it out of training and model selection, and evaluate it once after the model is frozen. It is still synthetic and authored by the same project workflow, so it is not evidence of real-world performance.
- Report accuracy, macro F1, class precision/recall and the confusion matrix. Inspect mistakes, especially priority misses and false escalations.
- Do not manipulate labels just to raise accuracy. Correct labels only when they conflict with the written rubric.

## COMI-LINGUA
COMI-LINGUA TN, NER and LID remain separate auxiliary language resources. Their annotations are not urgency labels and are not included as target labels here. If a component is trained on COMI-LINGUA, evaluate it as a separate task and provide appropriate attribution under the CC BY 4.0 licence.
