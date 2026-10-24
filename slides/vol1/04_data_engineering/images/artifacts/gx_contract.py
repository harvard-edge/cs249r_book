#!/usr/bin/env python3
"""A data contract for keyword-spotting clip metadata, run with Great Expectations.

Tested with great_expectations 1.24.0 (pip install great_expectations; Apache 2.0).
The batch is six clip records, two of them recorded at 8 kHz.
The slide "A data contract is a test suite run on every batch" quotes the
suite lines and the printed result exactly.

Run: python3 gx_contract.py
"""
import pandas as pd
import great_expectations as gx

batch_df = pd.DataFrame({
    "clip_id":     ["c01", "c02", "c03", "c04", "c05", "c06"],
    "sample_rate": [16000, 16000, 8000, 16000, 8000, 16000],
    "duration_s":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "label":       ["alexa", "none", "none", "alexa", "none", "none"],
})
context = gx.get_context(mode="ephemeral")
asset = context.data_sources.add_pandas("clips").add_dataframe_asset("clips")
batch = asset.add_batch_definition_whole_dataframe("all").get_batch(
    batch_parameters={"dataframe": batch_df})

# --- slide excerpt begins ---
E = gx.expectations
suite = gx.ExpectationSuite(name="kws_clips")
add = suite.add_expectation
add(E.ExpectColumnValuesToBeInSet(
    column="sample_rate", value_set=[16000]))
add(E.ExpectColumnValuesToBeBetween(
    column="duration_s",
    min_value=0.9, max_value=1.1))
add(E.ExpectColumnValuesToBeUnique(
    column="clip_id"))
add(E.ExpectColumnValuesToBeInSet(
    column="label",
    value_set=["alexa", "none"]))
result = batch.validate(suite)
# --- slide excerpt ends ---

print("success:", result.success)
for r in result.results:
    c = r.expectation_config
    print(c.kwargs["column"], r.success, r.result.get("unexpected_count"),
          r.result.get("partial_unexpected_list"))
