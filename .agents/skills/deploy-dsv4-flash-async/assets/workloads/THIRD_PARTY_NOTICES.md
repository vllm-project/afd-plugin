# Third-party notices for the DeepSeek-V4-Flash workload

## WildChat-4.8M

This database contains transformed information from WildChat-4.8M by AllenAI,
revision `c827c6df8fcf008219ffaffa4d1dd77491099367`, made available under the Open
Data Commons Attribution License (ODC-By) 1.0. The transformation selects
natural conversation prefixes, removes prompt-text and user/request metadata
from the delivered artifacts, renders them for DeepSeek-V4-Flash, and stores
token IDs. Token IDs remain reversible with the matching tokenizer.

Source: <https://huggingface.co/datasets/allenai/WildChat-4.8M/tree/c827c6df8fcf008219ffaffa4d1dd77491099367>

## Mooncake

Arrival shapes and original target lengths come from
`FAST25-release/traces/conversation_trace.jsonl` in kvcache-ai/Mooncake,
commit `e94a0b86ba067455d8b0524eb2cbb5fbac2db024`, released under Apache-2.0.

Source: <https://github.com/kvcache-ai/Mooncake/tree/e94a0b86ba067455d8b0524eb2cbb5fbac2db024>

## DeepSeek-V4-Flash tokenizer and encoding

Prompt token IDs were generated with the tokenizer and official encoding from
deepseek-ai/DeepSeek-V4-Flash, revision
`60d8d70770c6776ff598c94bb586a859a38244f1`, released under the MIT License.

Source: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1>

Exact source and implementation hashes are recorded in
`source_manifest.json`.

## Bundled vLLM benchmark adapter

This skill ships only the concatenated formal_0/1/2 workload: 1,536 requests.
The custom JSONL adapter contains decoded prompt text and generated workload
metadata for vLLM bench serve, rather than the token-ID-only artifacts described
above. Compression preserves the existing adapter bytes and request order.
Its checksum and frozen source revisions are recorded in manifest.json.
