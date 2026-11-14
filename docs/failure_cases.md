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

## Found by the evaluation (local qwen2.5:7b, 6 datasets, unattended runs)

Full numbers: `evaluation/results/results.md`. The hosted model was not evaluated (no API key).

| # | What went wrong | Where | How it was handled |
|---|---|---|---|
| 16 | The first LLM plan failed a validator in 5 of 6 runs per variant: transforming the target (Iris), forgetting to impute `Embarked` (Titanic), a step whose column was literally named `drop_column` (California), dropping and transforming the same column, leaving 100 duplicates. | Plan and data validators | The Planner-Critic and fix loops repaired 5 of 6 runs in both variants. |
| 17 | Mode imputation of numeric columns with missing values (`petal length`, `petal width`, `Age`). Validators pass it: the result has no nulls and the distribution shift stays under the threshold. | iris_corrupt (A, B), titanic_corrupt (B) | Counted as a silent mistake. Candidate fix: a plan check that warns on `impute_mode` for numeric columns. |
| 18 | Categorical target leak kept (`species_code`) by the rule-based plan and by both LLM variants. The leakage validator only warns when a category maps one-to-one onto the target. | iris_corrupt | Silent mistake. Candidate fix: make an exact one-to-one mapping an error when the column has few levels. The numeric leak (`value_index`, r = 1.000) was always blocked. |
| 19 | The Critic approved plans that still had validator errors 6 times, and raised objections that made plans worse (Titanic: B 0.58 first-proposal accuracy vs 0.75 for A). | B runs | Validators overruled every such approval. With this local model, B was not better than A: plan accuracy 0.87 vs 0.90, silent-mistake runs 2/5 vs 1/5, at about twice the LLM calls and tokens. |
| 20 | Runs that could not be repaired in 3 rounds: titanic_corrupt A (`outcome_label` never encoded or dropped), california_corrupt B (duplicates and the numeric leak). | Fix loop | Stopped with an explanation, as designed; in the app the job returns to checkpoint 1 for the user. The rule-based plan also stopped on california_corrupt because it cannot see the numeric leak. |
| 21 | Invalid JSON or schema replies: 1.0 (A) and 2.3 (B) per run. | All LLM runs | Rejected and retried; no run had to fall back because of invalid output. |
| 22 | Redundant steps: parsing already-numeric columns and imputing columns with nothing missing (up to 8 per run). | LLM runs | Harmless; reported separately rather than as mistakes. The first scoring counted them as errors and was corrected before the results were written. |
| 23 | Report writer: no invented numbers in any first draft (0 of 6). The number checker cannot judge qualitative claims (case 7). | Writer runs | No regeneration or stripping was needed. |

## Summary

On clean data every variant reaches the known-good plan once the validators and loops have done their work.
On corrupted data the local 7B Planner is less accurate than the rule-based plan, and its typical silent
mistake is mode-imputing numeric columns. The deterministic validators and the bounded loops are what make
the LLM variants usable; the local Critic adds cost without improving plans. The human checkpoint remains
necessary for categorical leaks, which neither the rules, the agents nor the validators caught.
