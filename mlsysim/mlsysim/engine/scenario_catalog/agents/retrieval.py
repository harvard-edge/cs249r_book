"""Named agents scenarios for retrieval.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ... import scenario_types as _st


# --- Retrieval from long-term memory (added 2026-10-20) ------------------
RetrievalRecordsInAllowance = _st.RetrievalPayloadScenario(
    name="Whole retrieval records staged in a 32,768-token call",
    description="2,048 tokens reserved for output and 24,576 for the system prompt, tool "
    "schemas, and history; a 128-token envelope and records of 512 body plus 64 "
    "metadata tokens; metadata doubled for comparison.",
    window=32_768,  # scenario assumption: hypothetical call limit
    output_reserve=2_048,  # scenario assumption
    other_input=24_576,  # scenario assumption: system prompt, tool schemas, history
    envelope=128,  # scenario assumption: response envelope
    body=512,  # scenario assumption: body tokens per record
    metadata=64,  # scenario assumption: provenance tokens per record
    metadata_scale=2,  # scenario assumption: per-record metadata doubled
)


Bm25ShortAgainstLongFile = _st.Bm25PairScenario(
    name="One rare term in a short file with one match and a long file with six",
    description="100,000 documents, 5 containing the term, mean length 400 analyzer terms; "
    "80 terms with one occurrence against 4,000 with six; k1 = 1.2, b = 0.75.",
    documents=100_000,  # scenario assumption: stipulated corpus statistics
    matching_documents=5,  # scenario assumption
    average_length=400,  # scenario assumption: analyzer terms
    short_length=80,  # scenario assumption
    short_frequency=1,  # scenario assumption
    long_length=4_000,  # scenario assumption
    long_frequency=6,  # scenario assumption
    k1=1.2,  # scenario assumption: a common default
    b=0.75,  # scenario assumption: a common default
)


ExactScanTenMillionVectors = _st.ExactScanScenario(
    name="Exact scan of 10 million 768-dimension FP16 vectors on the reference desktop CPU",
    description="20 searches in one trajectory; a scope filter that leaves 10,000 vectors; a 10 ms per-search budget.",
    host=_Hardware.Cloud.ReferenceCPU,
    vectors=10_000_000,  # scenario assumption: hypothetical corpus
    dimension=768,  # scenario assumption: a common embedding width
    bytes_per_value=_BYTES_FP16,
    searches=20,  # scenario assumption: searches in one trajectory
    filtered_vectors=10_000,  # scenario assumption: eligible set after a selective filter
    budget=Q_(10, "ms"),  # scenario assumption: per-search latency budget
)


AnnIndexesTenMillionVectors = _st.AnnFootprintScenario(
    name="HNSW links and IVF-PQ codes over the 10-million-vector exact-scan corpus",
    description="32 eight-byte links per node; 64 one-byte subvector codes with 256 "
    "centroids each and an FP32 lookup table.",
    vectors=ExactScanTenMillionVectors.vectors,
    dimension=ExactScanTenMillionVectors.dimension,
    bytes_per_value=ExactScanTenMillionVectors.bytes_per_value,
    links_per_node=32,  # scenario assumption: graph degree M
    link_bytes=Q_(8, "byte"),  # scenario assumption: one 64-bit neighbor id
    subvectors=64,  # scenario assumption: PQ subvectors m
    centroids=256,  # scenario assumption: one-byte codes
    code_bytes=Q_(1, "byte"),
    table_entry_bytes=_BYTES_FP32,
)


TypedTraversalDepthThree = _st.GraphNeighborhoodScenario(
    name="A depth-3 exact-tree neighborhood against the same call's retrieval allowance",
    description="3 children per node, 180 serialized tokens per node, 100 tokens of edges and envelope.",
    window=RetrievalRecordsInAllowance.window,
    output_reserve=RetrievalRecordsInAllowance.output_reserve,
    other_input=RetrievalRecordsInAllowance.other_input,
    children=3,  # scenario assumption: exact tree, no repeated nodes
    depth=3,  # scenario assumption: the seed is depth zero
    node_tokens=180,  # scenario assumption: per node, provenance included
    envelope_tokens=100,  # scenario assumption: edges and envelope
)


RepairEditIndexMaintenance = _st.IndexMaintenanceScenario(
    name="An eight-file edit: synchronous re-indexing against exclusion",
    description="Two chunks per file; stipulated per-stage costs for parsing, postings, a "
    "local embedding batch of 16, and a remote embedding request; in-memory "
    "exclusion entries and checks over 50 candidates.",
    files=8,  # scenario assumption
    chunks_per_file=2,  # scenario assumption
    parse_time=Q_(3.5, "ms"),  # scenario assumption: per file
    postings_time=Q_(2.0, "ms"),  # scenario assumption: per chunk
    local_batch_time=Q_(28.0, "ms"),  # scenario assumption: one local batch
    batch=16,  # scenario assumption
    remote_request_time=Q_(150, "ms"),  # scenario assumption: one remote request
    exclusion_write_time=Q_(50, "ns"),  # scenario assumption: per entry, in memory
    candidates=50,  # scenario assumption
    exclusion_check_time=Q_(30, "ns"),  # scenario assumption: per candidate, in memory
)


LexicalGraphAgainstDenseStore = _st.StorageAmplificationScenario(
    name="A lexical-plus-graph store against a lexical-plus-dense store over 50,000 files",
    description="200 tokens and 800 bytes per file; 0.8 bytes of postings per token; four "
    "32-byte edges per file; 100-token chunks of 768 FP32 values with the ANN "
    "graph's links; 15 edited files; stipulated throughputs.",
    files=50_000,  # scenario assumption
    tokens_per_file=200,  # scenario assumption
    bytes_per_file=Q_(800, "byte"),  # scenario assumption
    postings_per_token=Q_(0.8, "byte"),  # scenario assumption
    edges_per_file=4,  # scenario assumption
    edge_bytes=Q_(32, "byte"),  # scenario assumption
    passage_tokens=100,  # scenario assumption
    dimension=768,  # scenario assumption
    bytes_per_value=_BYTES_FP32,
    links_per_node=AnnIndexesTenMillionVectors.links_per_node,
    link_bytes=AnnIndexesTenMillionVectors.link_bytes,
    edited_files=15,  # scenario assumption
    lexical_rate=1_250,  # scenario assumption: files per second
    parse_rate=680,  # scenario assumption: files per second
    embed_rate=50,  # scenario assumption: chunks per second
)
