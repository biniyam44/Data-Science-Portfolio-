# Forecast Research Assistant

The assistant answers questions about the two saved forecasting studies. It can compare models at one horizon, identify the lowest measured MAE, inspect the largest errors, and quote the relevant methodology section. It does not forecast new observations, offer trading advice, search the web, or explain an error's cause without evidence.

## Try it without a model or an API key

From the repository root, after installing the project:

```bash
python scripts/ask_forecasts.py \
  "Which energy model has the lowest validation MAE at six hours, and where did it fail?"

python scripts/ask_forecasts.py \
  "Compare crypto models on test at 7 days."

python scripts/ask_forecasts.py \
  "How are prediction intervals calibrated?"

python scripts/evaluate_analyst.py
python -m pytest tests/test_analyst.py
```

The default is `deterministic-evidence`: a small phrase router selects a typed operation. It is a working evidence-retrieval demo, not an LLM, and is not labelled as model inference. Numeric questions must name the dataset, horizon and phase. For example, a validation winner and a test winner can differ. The assistant does not silently substitute one for the other.

## What an answer is based on

At startup, `EvidenceStore` reuses the portfolio export validator. It checks source fingerprints, forecast targets, complete model/origin coverage, intervals, recomputed metrics, selected models and implementation fingerprints. It requires the full six-model comparisons, not quick-run output. It then retains an in-memory snapshot. A changed artifact requires a process restart and fresh validation; answers include the snapshot's hashes rather than pretending to have re-read a changing file.

The tools have no caller-supplied paths:

- `best_model`: rank all models by MAE on the specified phase and include the winner's three largest absolute errors.
- `compare_models`: compute MAE, RMSE, counts and test interval coverage for all six models at the same horizon.
- `worst_errors`: read the three largest errors for a named model.
- `read_methodology`: retrieve one of six bounded sections from `docs/methodology.md`.
- `decline`: refuse an unsupported or underspecified request.

Numbers are calculated in Python from the verified prediction rows. Each result cites a repository-relative file, exact line range, SHA-256 and row selectors. Error records include the origin, target timestamp, observation, prediction and absolute error. Test rankings are explicitly descriptive: the experiment selected models on validation, and its historical test period was inspected during development.

## Optional model-selected tools

`CompatibleToolPlanner` supports the tool-calling Chat Completions interfaces of OpenRouter and OpenAI. There is no default model, automatic model fallback, automatic retry, or enabled paid call. Choose a model that actually supports this interface and strict function tools; models requiring another API are not supported by this adapter. The implementation follows the [OpenAI function-calling contract](https://developers.openai.com/api/docs/guides/function-calling) and [OpenRouter Chat Completions schema](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).

Set these server-side environment variables only when ready to make a metered request:

- `FORECAST_ENABLE_LLM=1`
- `FORECAST_LLM_PROVIDER`: `openrouter` or `openai`
- `FORECAST_LLM_MODEL`: the chosen model identifier
- `OPENROUTER_API_KEY` or `OPENAI_API_KEY`, matching the provider
- `FORECAST_INPUT_USD_PER_MILLION` and `FORECAST_OUTPUT_USD_PER_MILLION`: the current prices for that exact model and route
- `FORECAST_MAX_COST_USD`: a per-question budget covering the configured token ceiling

Then explicitly opt in for that invocation:

```bash
python scripts/ask_forecasts.py \
  "Which energy model is best on test at 6 hours?" \
  --llm --allow-paid
```

Do not put a key into the repository, browser JavaScript, a screenshot, or the command itself. The command reads the environment. No key or provider request was used to produce the committed offline evaluation.

In model mode, the only provider request contains the question and tool definitions. The model chooses one operation; the server validates its fields, supported values, explicit phase, horizon and dataset before execution. The model's free-text answer is discarded. A trusted renderer writes the final numerical answer from retrieved facts. There is no second completion after retrieval, so the model cannot rewrite a checked result or turn a retrieved passage into instructions.

The default limits are one request, one function call, 1,600 UTF-8 bytes per question, 512 completion tokens, a 20-second network timeout and a 64 KiB response bound. A conservative serialized-byte estimate plus a framing allowance bounds input to 8,192 tokens. Requests exceeding the configured token-cost ceiling are rejected before network access. The adapter will not follow redirects, use environment proxy settings, or send requests to an arbitrary host. Provider usage is retained when supplied; estimated cost uses the explicitly supplied prices. Missing usage remains `null`, not a fabricated zero. Estimates are not a substitute for provider-enforced account spending limits or a billing record.

## Evaluation and limits

`tests/fixtures/analyst_questions.json` fixes 26 questions covering rankings, comparisons, worst errors, methodology, missing scope, unsupported horizons, and several prompt-injection attempts. `scripts/evaluate_analyst.py` checks tool selection, independently aggregates errors using the standard library, verifies file hashes and citation row identities, and records observed local latency.

The offline report is `results/analyst/offline_evaluation.json`. It explicitly records `llm_evaluation: false` and `provider_requests: 0`. Startup validation time is separate from per-question latency, and answered and refused questions have separate medians. These are local measurements, not network-service benchmarks. Provider-adapter tests use mocked HTTP responses, including wrong scopes, invented text, extra tools, malformed arguments, timeouts and oversized bodies. They demonstrate application behavior, not that a particular live model selects the right tool reliably.

The refusal phrase filter is a convenience, not the main security boundary. Typed tools, fixed file access, bounded retrieval and trusted numerical rendering still apply if unfamiliar prompt wording passes that filter. The fixed questions are a regression set, not an independent held-out language benchmark. A live model evaluation, a separate unseen question set and actual provider cost measurements remain outstanding. Do not describe the offline pass rate as LLM accuracy or comprehensive prompt-injection resistance.

The public website can show saved examples without a key. A live endpoint must be hosted separately and require server-side authentication, request limits and monitoring. Publishing GitHub Pages does not deploy this Python service.
