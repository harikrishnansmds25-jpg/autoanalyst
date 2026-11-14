from autoanalyst.agents import fallback_plan


def _actions(plan):
    return {(s.column, s.action) for s in plan.steps}


def test_fallback_covers_each_problem(messy_profile, settings):
    plan = fallback_plan(messy_profile, settings)
    assert plan.source == "fallback" and plan.target == "churn"
    assert {
        (None, "drop_duplicates"),
        ("customer_id", "drop_column"),
        ("notes", "drop_column"),
        ("constant", "drop_column"),
        ("age", "impute_median"),
        ("age", "cap_outliers"),
        ("income", "coerce_numeric"),
        ("income", "impute_median"),
        ("signup_date", "parse_dates"),
        ("city", "impute_mode"),
        ("city", "encode_onehot"),
        ("member", "encode_ordinal"),
        ("churn", "keep"),
    } == _actions(plan)


def test_every_step_has_a_reason(messy_profile, settings):
    assert all(len(s.reason) > 5 for s in fallback_plan(messy_profile, settings).steps)


def test_fallback_is_deterministic(messy_profile, settings):
    first = fallback_plan(messy_profile, settings).model_dump_json()
    assert first == fallback_plan(messy_profile, settings).model_dump_json()


def test_high_cardinality_and_mostly_missing_are_dropped(messy_profile, settings):
    profile = messy_profile.model_copy(deep=True)
    city = profile.column("city")
    city.n_unique = 80
    age = profile.column("age")
    age.missing_frac = 0.7
    plan = fallback_plan(profile, settings)
    assert ("city", "drop_column") in _actions(plan)
    assert ("age", "drop_column") in _actions(plan)


def test_scaling_is_left_to_the_model_registry(messy_profile, settings):
    assert all(s.action != "scale_standard" for s in fallback_plan(messy_profile, settings).steps)
