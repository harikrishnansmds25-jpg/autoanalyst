# Failure cases

Problems found while building and evaluating AutoAnalyst, how each was detected, and how it is handled now.
The evaluation section is generated from `evaluation/results/` and summarised at the end.

## Found during development

| # | Phase | What went wrong | How it was found | How it is handled |
|---|---|---|---|---|
| 1 | 1 | The distribution-shift validator failed every capped column: capping a 500-year-old "age" changes the standard deviation by design. | Smoke run on the messy fixture | The check now measures imputation only; both sides are capped. Capping itself is bounded by its factor (1-10). |
| 2 | 2 | The profiler crashed on Titanic `Name`: a unique text column is classed as an identifier, and identifiers got numeric statistics. | First run on real data | Numeric statistics only for numeric dtypes. Regression test added. |
| 3 | 2 | Comparing the model with the dummy on macro F1 is unfair: a majority-class dummy has an artificially low F1, so noise "wins". | Noise-target experiment | Baseline comparison uses balanced accuracy (dummy = 1/k) for classification and RMSE for regression. |
| 4 | 2 | With 3-fold CV a model trained on pure noise was once judged to beat the baseline (1 of 12 runs; 0 of 12 with 5 folds). | Same experiment | Kept as a documented heuristic (margin of one CV std plus a test-set win); tests use the default 5 folds over several seeds. The report says "does not clearly beat". |
| 5 | 3 | Rendering single Jinja blocks skipped the template's top-level `set` statements, so the narrative template failed. | Test run | Shortcuts are passed in from Python. |
| 6 | 3 | The default python-docx Title style has a coloured border and blue theme headings, against the black-and-white requirement. | Rendering the PDF and looking at it | Styles are reset to black; the border is removed; a test checks heading colours. |
| 7 | 3 | The number checker cannot tell whether a number is attributed correctly ("accuracy 0.810" when 0.810 is the F1) or check qualitative claims. | Design review, then seen in Phase 4 ("Age had the weakest association") | Documented limitation; metric names are fixed in the template narrative; the writer prompt forbids computing new numbers. |
| 8 | 4 | qwen2.5:7b failed every Planner call: under constrained decoding it left out the optional `column` field on every step. | First live run against Ollama | `column` is required but nullable in the agent schemas; a test checks the JSON schema; the prompt version was bumped to invalidate cached plans. |
| 9 | 4 | The mock provider could not rebuild a profile from the prompt, because fields hidden from agents (hash, encoding) were missing. | Smoke run (the system fell back to the rule-based plan, as designed) | The mock fills the hidden fields. |
| 10 | 4 | The local Critic approved a plan that still had a validator error (Titanic `Embarked` not imputed). | Live debate | Validators overrule the approval, the overrule is logged, and the Planner fixes the plan in the next round. |
| 11 | 4 | The local Critic raised a wrong objection ("target column is not encoded"). | Live debate | The Planner kept the target untouched; validator checks forbid any step on the target, so a wrong objection cannot change it. |
| 12 | 5 | An invalid edited plan row showed the pydantic documentation link instead of the problem. | Unit test of the plan table | The first validation message is shown with its row number. |
| 13 | 6 | The rule-based plan keeps categorical columns that leak the target (`species_code`, `outcome_label`): the validator only warns when a category maps one-to-one onto the target. | Evaluation datasets with planted leaks | Measured as a silent mistake. The numeric leak (`value_index`, r = 1.000) is a validator error and stops the run. See the evaluation section for how the LLM variants compare. |
| 14 | 6 | With an 8k context, qwen2.5:7b needs 5.4 GB and about 21% of it runs on the CPU of the 6 GB GPU laptop. | `ollama ps` during evaluation | Accepted; reflected in the measured time per run. `AUTOANALYST_OLLAMA_NUM_CTX` can be lowered for shorter prompts. |
| 15 | ops | A session ended while the model download and the SDK install were still running; nothing recorded how far they had got. | Restart | State was checked directly (installed packages, `ollama list`, files). The evaluation now writes a progress log, and every job has `job.log`. |
