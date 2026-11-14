# Evaluation results

Hosted model (Gemini): **not run** (no `GEMINI_API_KEY` configured).

## Summary by variant

Plan accuracy: share of columns (plus the duplicate-row rule) whose effective actions in the executed plan match the known-good plan. Steps that cannot change the data (parsing an already numeric column, imputing a column with nothing missing) are not mistakes; they are counted as redundant steps. Recovered: runs whose first proposal failed a validator and that ended with a validated plan. Silent mistakes: validated runs that still keep an identifier or leak column, impute a mostly-missing column, or mode-impute a numeric column that has missing values. Runs are unattended (no user edits) and single-shot at temperature 0.1 with a fixed seed.

| Variant | Model | Runs | Plan acc. | Exact plans | Recovered | Validated | Runs with silent mistakes | Redundant steps/run | LLM calls/run | Rejected replies/run | Tokens/run (in / out) | Time/run (s) | Cost/run (USD) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rule-based (no LLM) | - | 6 | 0.95 | 3/6 | 0/1 | 5/6 | 2/5 | 0.0 | 0.0 | 0.0 | 0 / 0 | 0.1 | 0.0000 |
| A: Planner + validators | ollama (qwen2.5:7b) | 6 | 0.90 | 4/6 | 5/6 | 5/6 | 1/5 | 1.7 | 4.3 | 1.0 | 18,883 / 2,901 | 178.1 | 0.0000 |
| B: Planner + Critic + validators | ollama (qwen2.5:7b) | 6 | 0.87 | 3/6 | 5/6 | 5/6 | 2/5 | 3.3 | 9.2 | 2.3 | 39,520 / 3,532 | 236.7 | 0.0000 |

## Per dataset

| Dataset | Variant | Debate | Rounds | First proposal acc. (strict) | Plan acc. | Execution | Fix rounds | Silent | Redundant | Critic objections | Critic overruled | LLM calls | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris | fallback | fallback | 0 | 1.00 | 1.00 | passed | 1 | 0 | 0 | 0 | 0 | 0 | 0.04 |
| iris | A | escalated | 3 | 0.80 | 1.00 | passed | 1 | 0 | 0 | 0 | 0 | 4 | 109.55 |
| iris | B | approved | 3 | 0.80 | 1.00 | passed | 1 | 0 | 0 | 4 | 1 | 10 | 186.37 |
| titanic | fallback | fallback | 0 | 1.00 | 1.00 | passed | 1 | 0 | 0 | 0 | 0 | 0 | 0.08 |
| titanic | A | approved | 2 | 0.75 | 1.00 | passed | 1 | 0 | 3 | 0 | 0 | 2 | 71.89 |
| titanic | B | approved | 2 | 0.58 | 1.00 | passed | 1 | 0 | 5 | 0 | 0 | 7 | 121.07 |
| california | fallback | fallback | 0 | 1.00 | 1.00 | passed | 1 | 0 | 0 | 0 | 0 | 0 | 0.14 |
| california | A | approved | 3 | 1.00 | 1.00 | passed | 1 | 0 | 0 | 0 | 0 | 3 | 67.09 |
| california | B | approved | 2 | 1.00 | 1.00 | passed | 1 | 0 | 8 | 3 | 0 | 5 | 99.74 |
| iris_corrupt | fallback | fallback | 0 | 0.88 | 0.88 | passed | 1 | 1 | 0 | 0 | 0 | 0 | 0.07 |
| iris_corrupt | A | escalated | 3 | 0.12 | 0.62 | passed | 2 | 3 | 4 | 0 | 0 | 5 | 233.67 |
| iris_corrupt | B | approved | 3 | 0.12 | 0.50 | passed | 1 | 3 | 3 | 2 | 2 | 9 | 235.4 |
| titanic_corrupt | fallback | fallback | 0 | 0.93 | 0.93 | passed | 1 | 1 | 0 | 0 | 0 | 0 | 0.11 |
| titanic_corrupt | A | escalated | 3 | 0.71 | 0.79 | stopped | 3 | 0 | 0 | 0 | 0 | 6 | 299.03 |
| titanic_corrupt | B | escalated | 3 | 0.71 | 0.79 | passed | 3 | 1 | 1 | 5 | 3 | 15 | 570.54 |
| california_corrupt | fallback | fallback | 0 | 0.92 | 0.92 | stopped | 1 | 0 | 0 | 0 | 0 | 0 | 0.19 |
| california_corrupt | A | approved | 1 | 0.50 | 1.00 | passed | 3 | 0 | 3 | 0 | 0 | 6 | 287.45 |
| california_corrupt | B | approved | 1 | 0.50 | 0.92 | stopped | 3 | 0 | 3 | 0 | 0 | 9 | 207.37 |

## Report writer

| Dataset | Model | Source | Attempts | Invented (1st draft) | Invented per attempt | Sentences removed | Invented in final report | Seconds |
|---|---|---|---|---|---|---|---|---|
| iris | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 26.65 |
| titanic | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 21.81 |
| california | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 19.26 |
| iris_corrupt | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 29.59 |
| titanic_corrupt | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 25.73 |
| california_corrupt | ollama (qwen2.5:7b) | writer | 1 | 0 | [] | 0 | 0 | 27.22 |

## Failure cases found

- **Recovered**, iris / A (ollama): first proposal failed (Target 'species' must not be transformed by the plan); fixed by the loop.
- **Escalated**, iris / A (ollama): no agreement after 3 rounds.
- **Invalid LLM output**, iris / A (ollama): 1 reply(ies) rejected by the schema and retried.
- **Recovered**, iris / B (ollama): first proposal failed (Target 'species' must not be transformed by the plan); fixed by the loop.
- **Critic overruled**, iris / B (ollama): Critic approved 1 plan(s) with validator errors.
- **Invalid LLM output**, iris / B (ollama): 3 reply(ies) rejected by the schema and retried.
- **Recovered**, titanic / A (ollama): first proposal failed ('Embarked' has missing values but no imputation step); fixed by the loop.
- **Recovered**, titanic / B (ollama): first proposal failed ('Embarked' has missing values but no imputation step); fixed by the loop.
- **Invalid LLM output**, titanic / B (ollama): 3 reply(ies) rejected by the schema and retried.
- **Recovered**, california / A (ollama): first proposal failed (Column 'drop_column' does not exist); fixed by the loop.
- **Recovered**, california / B (ollama): first proposal failed (Column 'drop_column' does not exist); fixed by the loop.
- **Silent mistake**, iris_corrupt / fallback (none): leak column 'species_code' kept. Validators passed the plan.
- **Silent mistake**, iris_corrupt / A (ollama): numeric column 'petal length (cm)' with missing values imputed with the mode. Validators passed the plan.
- **Silent mistake**, iris_corrupt / A (ollama): numeric column 'petal width (cm)' with missing values imputed with the mode. Validators passed the plan.
- **Silent mistake**, iris_corrupt / A (ollama): leak column 'species_code' kept. Validators passed the plan.
- **Recovered**, iris_corrupt / A (ollama): first proposal failed ('measured_on' is dropped but also has other steps; drop_column(measured_on) failed: Column 'measured_on' not found); fixed by the loop.
- **Escalated**, iris_corrupt / A (ollama): no agreement after 3 rounds.
- **Invalid LLM output**, iris_corrupt / A (ollama): 1 reply(ies) rejected by the schema and retried.
- **Plan differs from known-good**, iris_corrupt / A (ollama): petal length (cm), petal width (cm), species_code
- **Silent mistake**, iris_corrupt / B (ollama): numeric column 'petal length (cm)' with missing values imputed with the mode. Validators passed the plan.
- **Silent mistake**, iris_corrupt / B (ollama): numeric column 'petal width (cm)' with missing values imputed with the mode. Validators passed the plan.
- **Silent mistake**, iris_corrupt / B (ollama): leak column 'species_code' kept. Validators passed the plan.
- **Recovered**, iris_corrupt / B (ollama): first proposal failed ('measured_on' is dropped but also has other steps); fixed by the loop.
- **Critic overruled**, iris_corrupt / B (ollama): Critic approved 2 plan(s) with validator errors.
- **Invalid LLM output**, iris_corrupt / B (ollama): 2 reply(ies) rejected by the schema and retried.
- **Plan differs from known-good**, iris_corrupt / B (ollama): petal length (cm), petal width (cm), measured_on, species_code
- **Silent mistake**, titanic_corrupt / fallback (none): leak column 'outcome_label' kept. Validators passed the plan.
- **Stopped**, titanic_corrupt / A (ollama): validation still failing after 3 round(s). First errors: non-numeric model inputs: outcome_label
- **Escalated**, titanic_corrupt / A (ollama): no agreement after 3 rounds.
- **Invalid LLM output**, titanic_corrupt / A (ollama): 1 reply(ies) rejected by the schema and retried.
- **Plan differs from known-good**, titanic_corrupt / A (ollama): Sex, Embarked, outcome_label
- **Silent mistake**, titanic_corrupt / B (ollama): numeric column 'Age' with missing values imputed with the mode. Validators passed the plan.
- **Recovered**, titanic_corrupt / B (ollama): first proposal failed ('Sex' has missing values but no imputation step; 'Embarked' has missing values but no imputation step; 'outcome_label' (categorical) is kept but never encoded; non-numeric model inputs: outcome_label); fixed by the loop.
- **Critic overruled**, titanic_corrupt / B (ollama): Critic approved 3 plan(s) with validator errors.
- **Escalated**, titanic_corrupt / B (ollama): no agreement after 3 rounds.
- **Invalid LLM output**, titanic_corrupt / B (ollama): 4 reply(ies) rejected by the schema and retried.
- **Plan differs from known-good**, titanic_corrupt / B (ollama): Sex, Age, Embarked
- **Stopped**, california_corrupt / fallback (none): validation still failing after 1 round(s). First errors: 'value_index' correlates 1.000 with the target
- **Recovered**, california_corrupt / A (ollama): first proposal failed (100 duplicate rows remain; 'value_index' correlates 1.000 with the target); fixed by the loop.
- **Invalid LLM output**, california_corrupt / A (ollama): 3 reply(ies) rejected by the schema and retried.
- **Stopped**, california_corrupt / B (ollama): validation still failing after 3 round(s). First errors: 100 duplicate rows remain; 'value_index' correlates 1.000 with the target
- **Invalid LLM output**, california_corrupt / B (ollama): 2 reply(ies) rejected by the schema and retried.
- **Plan differs from known-good**, california_corrupt / B (ollama): (rows)
