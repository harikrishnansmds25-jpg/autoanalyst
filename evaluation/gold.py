"""Known-good plans and plan scoring.

For each column, the gold standard lists the acceptable sets of core actions. 'keep', 'cap_outliers'
and 'scale_standard' are neutral (a matter of taste here), and one-hot vs ordinal encoding count as
the same choice. Columns have a role; mistakes on 'id', 'leak' and 'mostly_missing' columns, or
mode-imputing a numeric column, are the silent mistakes validators may not catch.
"""

from dataclasses import dataclass, field

from autoanalyst.schemas import ColumnProfile, Plan, Profile

NEUTRAL = {"keep", "cap_outliers", "scale_standard"}
NONE: list[frozenset] = [frozenset()]
DROP = [frozenset({"drop_column"})]
DATE_NOISE = [frozenset({"parse_dates"}), frozenset({"parse_dates", "impute_median"}), frozenset({"drop_column"})]


def _sets(*alternatives) -> list[frozenset]:
    return [frozenset(a) for a in alternatives]


@dataclass(frozen=True)
class Gold:
    columns: dict[str, list[frozenset]]
    roles: dict[str, str] = field(default_factory=dict)
    needs_dedup: bool = False
    numeric: frozenset = frozenset()


_TITANIC_BASE = {
    "PassengerId": DROP,
    "Name": DROP,
    "Pclass": _sets(set(), {"encode"}),
    "Sex": _sets({"encode"}),
    "Age": _sets({"impute_median"}),
    "SibSp": NONE,
    "Parch": NONE,
    "Ticket": _sets({"drop_column"}, {"encode"}),
    "Fare": NONE,
    "Cabin": DROP,
    "Embarked": _sets({"impute_mode", "encode"}),
}
_CALIFORNIA = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
_IRIS = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

GOLD: dict[str, Gold] = {
    "iris": Gold({c: NONE for c in _IRIS}, needs_dedup=True, numeric=frozenset(_IRIS)),
    "iris_corrupt": Gold(
        {
            "record_id": DROP,
            "sepal length (cm)": _sets({"impute_median"}),
            "sepal width (cm)": NONE,
            "petal length (cm)": _sets({"coerce_numeric", "impute_median"}),
            "petal width (cm)": _sets({"impute_median"}),
            "measured_on": DATE_NOISE,
            "species_code": DROP,
        },
        roles={"record_id": "id", "species_code": "leak"},
        needs_dedup=True, numeric=frozenset(_IRIS),
    ),
    "titanic": Gold(
        _TITANIC_BASE,
        roles={"PassengerId": "id", "Name": "id", "Cabin": "mostly_missing"},
        numeric=frozenset({"Age", "Fare", "SibSp", "Parch"}),
    ),
    "titanic_corrupt": Gold(
        {
            **_TITANIC_BASE,
            "Sex": _sets({"impute_mode", "encode"}),
            "Age": _sets({"coerce_numeric", "impute_median"}),
            "Fare": _sets({"coerce_numeric"}, {"coerce_numeric", "impute_median"}),
            "booking_date": DATE_NOISE,
            "outcome_label": DROP,
        },
        roles={"PassengerId": "id", "Name": "id", "Cabin": "mostly_missing", "outcome_label": "leak"},
        needs_dedup=True, numeric=frozenset({"Age", "Fare", "SibSp", "Parch"}),
    ),
    "california": Gold({c: NONE for c in _CALIFORNIA}, numeric=frozenset(_CALIFORNIA)),
    "california_corrupt": Gold(
        {
            "record_id": DROP,
            **{c: NONE for c in _CALIFORNIA},
            "MedInc": _sets({"coerce_numeric"}, {"coerce_numeric", "impute_median"}),
            "HouseAge": _sets({"coerce_numeric", "impute_median"}),
            "AveRooms": _sets({"impute_median"}),
            "survey_date": DATE_NOISE,
            "value_index": DROP,
        },
        roles={"record_id": "id", "value_index": "leak"},
        needs_dedup=True, numeric=frozenset(_CALIFORNIA),
    ),
}


def core_actions(plan: Plan, column: str) -> frozenset:
    actions = {s.action for s in plan.steps_for(column)} - NEUTRAL
    return frozenset("encode" if a.startswith("encode_") else a for a in actions)


def effective(actions: frozenset, column: ColumnProfile | None) -> tuple[frozenset, list[str]]:
    """Remove steps that cannot change the data: parsing a column that is already numeric, or
    imputing a column with nothing to fill. Returns (effective actions, removed no-ops)."""
    if column is None:
        return actions, []
    noops = []
    if column.kind == "numeric" and "coerce_numeric" in actions:
        noops.append("coerce_numeric")
    if column.n_missing + column.n_unparseable == 0:
        noops += sorted(actions & {"impute_median", "impute_mode"})
    return actions - set(noops), noops


@dataclass
class PlanScore:
    columns: int
    correct: int
    dedup_ok: bool
    wrong: list[dict] = field(default_factory=list)
    silent: list[str] = field(default_factory=list)
    noops: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.correct + self.dedup_ok) / (self.columns + 1)

    @property
    def exact(self) -> bool:
        return self.correct == self.columns and self.dedup_ok


def _judge(column: str, got: frozenset, gold: Gold, profile: Profile | None, score: "PlanScore") -> None:
    got, noops = effective(got, profile.column(column) if profile else None)
    score.noops += [f"{a} on '{column}'" for a in noops]
    if got in gold.columns[column]:
        score.correct += 1
        return
    score.wrong.append({"column": column, "got": sorted(got),
                        "expected": [sorted(a) for a in gold.columns[column]]})
    role = gold.roles.get(column)
    if role in ("id", "leak") and "drop_column" not in got:
        score.silent.append(f"{role} column '{column}' kept")
    elif role == "mostly_missing" and "drop_column" not in got:
        score.silent.append(f"mostly-missing column '{column}' imputed instead of dropped")
    if column in gold.numeric and "impute_mode" in got:
        score.silent.append(f"numeric column '{column}' with missing values imputed with the mode")


def score_plan(plan: Plan, gold: Gold, profile: Profile | None = None) -> PlanScore:
    """Score a plan against the gold standard. With a profile, no-op steps are neutral."""
    score = PlanScore(columns=len(gold.columns), correct=0,
                      dedup_ok=any(s.action == "drop_duplicates" for s in plan.steps) or not gold.needs_dedup)
    if not score.dedup_ok:
        score.wrong.append({"column": "(rows)", "got": [], "expected": [["drop_duplicates"]]})
    for column in gold.columns:
        _judge(column, core_actions(plan, column), gold, profile, score)
    return score


def rescore(row: dict, gold: Gold, profile: Profile) -> PlanScore:
    """Re-judge a stored run from its wrong columns (correct columns stay correct)."""
    score = PlanScore(columns=len(gold.columns), correct=0, dedup_ok=True)
    wrong = {w["column"]: frozenset(w["got"]) for w in row["wrong_columns"]}
    if "(rows)" in wrong:
        score.dedup_ok = False
        score.wrong.append({"column": "(rows)", "got": [], "expected": [["drop_duplicates"]]})
    for column in gold.columns:
        if column in wrong:
            _judge(column, wrong[column], gold, profile, score)
        else:
            score.correct += 1
    return score
