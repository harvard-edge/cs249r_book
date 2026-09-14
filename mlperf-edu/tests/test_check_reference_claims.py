from __future__ import annotations

from tools import check_reference_claims


def test_draft_claim_checker_uses_complete_provisional_index():
    # Historical draft evidence remains internally verifiable after benchmark
    # development moves on. It must not be mistaken for current-source evidence.
    source_sha, records = check_reference_claims.load_provisional_cases()

    assert source_sha == "163d42ee3df54ab122543469ccf2b6b3bd119455"
    assert len(records) == 12
    assert {payload["workload"] for _entry, payload in records.values()} == {
        "anomaly-detection",
        "causal-language-modeling",
        "graph-node-classification",
        "image-classification",
        "information-retrieval",
        "keyword-spotting",
        "text-classification",
        "time-series-forecasting",
        "visual-wake-words",
    }
    assert check_reference_claims.check_documents(source_sha, records) == []
