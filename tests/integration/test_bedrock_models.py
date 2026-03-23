"""
One-off script to test which Bedrock models are invocable in ap-southeast-2.

Usage:
    python data/scripts/test_bedrock_models.py

Requirements: AWS credentials configured (same profile used for the project).
Uses the unified `converse` API so no model-specific request formatting needed.
"""

import boto3

REGION = "ap-southeast-2"

CANDIDATES = [
    ("Claude Haiku 4.5 (direct)",       "anthropic.claude-haiku-4-5-20251001:0"),
    ("Claude Haiku 4.5 (ap. profile)",  "ap.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("Claude 3.5 Sonnet v2 (known ok)", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ("Qwen3 Next 80B A3B",              "qwen.qwen3-next-80b-a3b"),
    ("Qwen3 32B dense",                 "qwen.qwen3-32b"),
]

client = boto3.client("bedrock-runtime", region_name=REGION)

print(f"\nTesting Bedrock model availability in {REGION}\n{'─' * 60}")

for name, model_id in CANDIDATES:
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with one word: OK"}]}],
            inferenceConfig={"maxTokens": 10},
        )
        reply = response["output"]["message"]["content"][0]["text"].strip()
        print(f"  ✅  {name}\n      {model_id}\n      Reply: {reply!r}\n")
    except Exception as exc:
        print(f"  ❌  {name}\n      {model_id}\n      {type(exc).__name__}: {exc}\n")
