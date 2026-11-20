"""The published numbers, their transcription, and the audit that recomputes derived quantities from them."""
import json

import pytest

from vrfraudnet import reported as R
from vrfraudnet.evaluation.audit import run_audit, to_markdown


@pytest.fixture(scope="module")
def findings():
    return {f.id: f for f in run_audit()}


def test_headline_numbers_are_transcribed_exactly():
    t5 = R.load("table5")["panels"]
    full = {k: R.row(t5[k]["rows"], "method", "VR-FraudNet")["AUPRC"] for k in ("5a", "5b", "5c")}
    assert full["5a"] == {"mean": 0.5247, "sd": 0.0058} and full["5b"] == {"mean": 0.5824, "sd": 0.0065} and full["5c"] == {"mean": 0.7456, "sd": 0.0049}
    tt = [R.mean(R.row(t5[k]["rows"], "method", "TabTransformer")["AUPRC"]) for k in ("5a", "5b", "5c")]
    assert tt == [0.5012, 0.5137, 0.7234]
    t8 = R.load("table8")["panels"]["8c"]["rows"]
    f = R.row(t8, "method", "Full VR-FraudNet")
    assert R.mean(f["train_auprc"]) == 0.6823 and R.mean(f["post_shock"]) == 0.6087 and f["delta_post_pp"] == -7.36
    t9 = R.load("table9")["panels"]
    assert [R.row(t9[k]["rows"], "method", "Full VR-FraudNet")["mean"] for k in ("9a", "9b", "9c")] == [0.0667, 0.1016, 0.1672]
    s3b = R.load("supplement_s3b")["rows"]
    assert R.mean(s3b[9]["p99 (ms)"]) == 10.23 and R.mean(s3b[10]["p99 (ms)"]) == 175.23


def test_every_main_table_row_count_matches_the_paper():
    t5 = R.load("table5")["panels"]
    assert all(len(t5[k]["rows"]) == 10 for k in t5)                         # nine baselines + VR-FraudNet
    assert len(R.load("table6")["rows"]) == 9 and len(R.load("table7a")["rows"]) == 10 and len(R.load("table8")["panels"]["8d"]["rows"]) == 5
    assert all(len(p["rows"]) == 5 for p in R.load("table9")["panels"].values())


def test_every_reference_file_names_its_source_location():
    for name in ("table1", "table5", "table6", "table7a", "table7b", "table8", "table9"):
        src = R.load(name)["source"]
        assert src["page"] and src["table"] and src["document"]
    for name in ("s1a", "s1b", "s1c", "s2a", "s2b", "s2c", "s3a", "s3b", "s4a", "s4b"):
        src = R.load(f"supplement_{name}")["source"]
        assert src["kind"] == "supplement_docx" and src["table"].startswith("Supplementary Table")


def test_supplement_cells_keep_units_and_sd():
    row = R.load("supplement_s1a")["rows"][3]
    assert row["Sum FN loss"] == {"value": 148500.0, "unit": "USD"} and row["Savings (%)"] == {"value": 78.1, "sd": 1.42}


def test_d3_recall_at_top_1pct_exceeds_its_arithmetic_ceiling(findings):
    f = findings["A-01D3"]
    assert f.status == "inconsistent" and f.severity == "blocking"
    assert f.data["positives"] == 4134 and f.data["n"] == 118108 and f.data["k"] == 1182
    assert f.data["ceiling"] == pytest.approx(0.2859, abs=1e-4)
    assert len(f.data["values_above"]) == 10                                  # every printed value, baselines included
    assert f.data["partition_prevalence"] == pytest.approx(0.035, abs=1e-4)   # the stated 3.50% DOES describe the holdout: it is not the explanation
    assert findings["A-01D1"].status == "consistent" and findings["A-01D2"].status == "consistent"


def test_derived_quantities_that_check_out(findings):
    for k in ("A-02", "A-03", "A-06", "A-08", "A-09", "A-10", "A-12", "A-13", "A-14", "A-15", "A-17", "A-18"):
        assert findings[k].status == "consistent", (k, findings[k].detail)


def test_defects_the_audit_is_expected_to_find(findings):
    assert findings["A-04"].status == "inconsistent"                          # printed p-values do not belong to the printed V values
    m = {d["baseline"]: d["V_implied_by_printed_p"] for d in findings["A-04"].data["mismatches"]}
    assert m["XGBoost"] == 464 and m["LightGBM"] == 462
    assert findings["A-05"].status == "unresolvable"                          # effect-size convention cannot be identified
    assert findings["A-07"].status == "note"                                  # Table 8a 'Train AUPRC' equals the Table 5a TEST value
    assert findings["A-19"].status == "inconsistent" and findings["A-19"].data["max_possible"] == 50000 and findings["A-19"].data["published"] == 100000
    assert findings["A-11"].data["drop_pp"] == pytest.approx(7.36, abs=0.005)


def test_s1_unit_costs_are_recovered(findings):
    d = findings["A-17"].detail
    assert "300.0, 5.0, 5.0" in d and "25000.0" in d and "150.0" in d


def test_audit_never_edits_published_values():
    before = json.dumps(R.load("table5"), sort_keys=True)
    run_audit()
    assert json.dumps(R.load("table5"), sort_keys=True) == before


def test_markdown_report_lists_every_finding(findings):
    md = to_markdown(list(findings.values()))
    assert md.count("\n## ") == len(findings) and "A-01D3" in md and "blocking" in md
