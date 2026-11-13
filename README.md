<p align="center">
  <img src="docs/assets/afd-plugin-logo.png" alt="AFD Plugin" width="500">
</p>

<p align="center">
| <a href="docs/README.md"><b>Documentation</b></a> | <a href="recipe/README.md"><b>Recipe</b></a> | <a href="https://docs.google.com/presentation/d/16DMlDLmwtgYCktXiayoM3W36YsoPURq15g50OJ5Iv-8/edit?usp=sharing"><b>Slides</b></a> | <a href="https://deepwiki.com/vllm-project/afd-plugin"><b>DeepWiki</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://vllm-dev.slack.com/archives/C0B4C1D84GG"><b>Developer Slack</b></a> | <a href="docs/assets/WeChat.png"><b>WeChat</b></a> |
</p>

## Overview

**afd-plugin** is a [vLLM](https://github.com/vllm-project/vllm)
external plugin for **Attention-FFN Disaggregation (AFD)**. It provides
plugin-owned worker classes, model runners, model wrappers, connectors,
configuration validation, compatibility shims, and hardware-gated integration
tests for GPU and Ascend NPU deployments.

> [!NOTE]
> This project is still experimental and needs more large-scale testing across
> different hardware backends.

The plugin extends vLLM through its general-plugin entry point without
modifying the vLLM source tree.

## Architecture

![afd-plugin architecture](docs/assets/vllm-afd-plugin-architecture.svg)

## Getting started

1. [Install AFD](docs/installation.md) for CUDA or Ascend NPU.
2. Choose a [connector and supported model](docs/support.md), then follow its
   [deployment recipe](recipe/README.md).
3. Pass role/topology settings through
   [`--additional-config`](docs/configuration.md). Send requests to Attention;
   FFN serves connector work.

## Support matrix

See the [support matrix](docs/support.md) for models, connectors, tested
configurations, and known accuracy/shutdown limits.

## Documentation

Browse the [documentation index](docs/README.md) for user guides, module design,
validation and troubleshooting.

## Development

For CPU-only or macOS development, follow [source setup](docs/installation.md#source-setup),
then run:

```bash
uv run pytest
uv run ruff check .
```

Read the [contributor rules](AGENTS.md) and [test maintenance guide](docs/UT_MAINTENANCE.md)
before changing code or tests. See [module design](docs/design/module/index.md)
for ownership, [native builds](csrc/README.md) for backend operators, and the
[E2E guide](tests/e2e/README.md) for hardware validation.

## License

afd-plugin is licensed under the [Apache License 2.0](LICENSE).

## Cite

If you find afd-plugin helpful in your research or projects, please
consider citing it:

```bibtex
@misc{afdplugin2026,
  title={afd-plugin: Attention-FFN Disaggregation for vLLM},
  author={AFD Plugin Contributors},
  year={2026},
  howpublished={\url{https://github.com/vllm-project/afd-plugin}},
}
```
