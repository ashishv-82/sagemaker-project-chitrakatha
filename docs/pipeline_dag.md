# Pipeline DAG

Two independent flows branch after `Preprocessing` and never block each other.

```mermaid
flowchart TD
    A([Preprocessing]) --> B[EmbedAndIndex\nFlow A — pgvector]
    A --> C[SynthesizePairs\nFlow B — RAFT pairs]
    C --> D[TrainQLoRARAFT]
    D --> E[EvaluateRAFT]
    E --> F{CheckEvalThresholds\nROUGE-L ≥ 0.35\nRobustness ≥ 0.70}
    F -->|pass| G[CreateChitrakathaModel]
    G --> H[RegisterChitrakathaModel\nPendingManualApproval]
    F -->|fail| I([End — no registration])

    style B fill:#d4edda,stroke:#28a745
    style D fill:#d4edda,stroke:#28a745
```

**Flow A** (`EmbedAndIndex`): completes independently → Lambda RAG inference works as soon as this step finishes.

**Flow B** (`SynthesizePairs → TrainQLoRARAFT → EvaluateRAFT → ...`): fine-tuning path, runs in parallel with Flow A and proceeds regardless of embedding outcome.
