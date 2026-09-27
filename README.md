# karpathy_rnn_lab

An executable, test-driven companion to Andrej Karpathy's
[*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
(2015) — read the post section by section, run the matching test file
alongside it, and see the claim demonstrated instead of taking it on faith.
No notebook: the tests *are* the exploration surface.

It started as the "minimal character-level RNN language model in
Python/numpy" the post links to (the 100-line gist) and now also covers the
post's two follow-on ideas, [`karpathy/char-rnn`](https://github.com/karpathy/char-rnn)'s
actual upgrades over that gist: an **LSTM cell** ("Getting fancy") and
**layer stacking** ("Going deep") — `char-rnn` itself is Lua/Torch, so
these are original ports of the *ideas*, not a line-by-line translation;
see "On char-rnn" below.

## Why this structure

- **Domain (`domain/`)** — pure functions and immutable value objects only.
  Every function takes a state and returns a *new* state; nothing here
  mutates an argument or does I/O. This is the "functional core":
  - `vocabulary.py` — 1-of-k encoding
  - `rnn_model.py` / `training.py` — the vanilla RNN cell + its BPTT
  - `lstm_model.py` / `lstm_training.py` — the LSTM cell + its BPTT
  - `stacked_rnn.py` / `stacked_training.py` — N-layer stacking of vanilla
    RNN cells + its BPTT
  - `stacked_lstm.py` / `stacked_lstm_training.py` — N-layer LSTM cells,
    inter-layer dropout, minibatch forward/BPTT, and nested Adagrad state
  - `optimization.py` — gradient clipping and the Adagrad update, written
    **once**, generically, for the flat vanilla-RNN and LSTM dataclasses
    (see "One optimizer, two cell types" below); both stacks reuse its
    lower-level `adagrad_step` directly, since their parameters are nested
- **Application (`application/`)** — `train_service.py`,
  `lstm_train_service.py`, `stacked_train_service.py`, and
  `stacked_lstm_train_service.py` train the vanilla RNN, LSTM, stacked
  vanilla RNN, and stacked LSTM respectively. The stacked-LSTM service
  supports minibatches and optional inter-layer dropout; its cross-entropy
  is averaged over target tokens. Each service returns final parameters
  with per-batch snapshots.
- **Infrastructure (`infrastructure/corpus.py`)** — where training text
  comes from (a file, or a small built-in default). The domain layer never
  knows or cares.
- **`quality/crap.py`** — composes `radon` (cyclomatic complexity) and
  `coverage.py` (line coverage) into the published CRAP formula, so
  "quality < 5" is something you can actually run, not just assert.

The domain layer's internal loops (`backpropagate_through_time`,
`lstm_backpropagate_through_time`, `stacked_backpropagate_through_time`,
the Adagrad `adagrad_step` closure) accumulate into local numpy arrays for
performance — the same way a hand-written BPTT loop would — but that's
invisible to callers, who only ever get back a fresh `*Gradients` /
`*Params` / `*Memory` value. Purity is a property of the function's
boundary, not a ban on local bookkeeping.

### One optimizer, two cell types

`clip_gradients`, `zero_memory`, and `adagrad_update` in
`domain/optimization.py` don't know or care whether a gradient came from
the vanilla RNN or a single LSTM cell — they work structurally, via
`dataclasses.fields()`, off one naming convention every cell type in this
project follows: a params field `Wxh` has a gradient field `dWxh` and a
memory field `mWxh`. `tests/test_optimization.py` exercises the same three
functions against both `Gradients`/`AdagradMemory` (RNN) and
`LSTMGradients`/`LSTMMemory` (LSTM) to prove that rather than just assert
it. Stacks have nested parameter structures, so their optimizers traverse
each layer and use the same lower-level `adagrad_step`. The chain rules
remain cell-specific because they genuinely differ.

## Running it

```bash
pip install -r requirements.txt
pytest                       # run everything (~1-2s)
pytest tests/test_lstm_gradient_check.py -v   # run just one section's tests

python -m scripts.run_qa     # run tests, refresh coverage, and enforce CRAP < 5

# Or run the quality steps separately:
coverage run --branch -m pytest
coverage json -o coverage.json
python quality/crap.py
```

> The CRAP check analyzes `infrastructure/`, `domain/`, `application/`,
> `scripts/`, and `tests/`; it excludes the quality tooling itself.

## Test file ↔ article section map

| Test file | Article section | What it demonstrates |
|---|---|---|
| `test_vocabulary.py` | *Character-Level Language Models* | 1-of-k ("one hot") encoding of a small vocabulary |
| `test_forward_pass.py` | *RNN computation* | `h = tanh(Whh·h + Wxh·x + bh)`; output is a real probability distribution; the hidden state actually matters — same input, different `h`, different output |
| `test_loss.py` | *"a more technical explanation ... Softmax classifier"* | cross-entropy loss behaves as expected: max at uniform, low when confident-and-correct, high when confident-and-wrong |
| `test_gradient_check.py` | *"we can run the backpropagation algorithm ..."* | a numerical gradient check — the same technique Karpathy's own reference gist ships — on all of `backpropagate_through_time`'s weight gradients, plus its `dh0` return value |
| `test_gradient_clipping.py` | (reference implementation, not named in the prose) | gradients get bounded so one bad batch can't blow up training |
| `test_adagrad_update.py` | *"per-parameter adaptive learning rate methods"* | Adagrad moves parameters against the gradient; a parameter that already saw big gradients gets a smaller effective step |
| `test_optimization.py` | (design property, not article prose) | `clip_gradients`/`zero_memory`/`adagrad_update` work identically for the RNN's and the LSTM's dataclasses — proof the "one optimizer" design holds |
| `test_sampling.py` | *"At test time ... we sample ... and feed it right back in"* + *Temperature* | RNN, LSTM, stacked-RNN, and stacked-LSTM sampling is reproducible under a fixed RNG seed; a synthetic peaked distribution sharpens toward the favorite at low temperature and flattens above 1 |
| `test_training_evolution_e2e.py` | *The evolution of samples while training* + the post's central claim | loss on a fixed window falls by >5x after training; a low-temperature sample from the trained model reproduces whole words it was never told about explicitly |
| `test_lstm_training_e2e.py` | *Getting fancy* | LSTM training lowers fixed-window loss, returns per-batch snapshots, and preserves the seeded initial parameters for a zero-epoch run |
| `test_stacked_rnn_training_e2e.py` | *Going deep* | stacked-RNN training lowers fixed-window loss, returns per-batch snapshots, and preserves the seeded initial parameters for a zero-epoch run |
| `test_lstm_forward_pass.py` | *Getting fancy* | the four gates squash into their expected ranges; the forget-gate-bias-1 trick keeps the cell "remembering" by default; the cell state genuinely blends forget/input contributions |
| `test_lstm_gradient_check.py` | *Getting fancy* | a numerical gradient check on all 14 of the LSTM's weight matrices, plus its `dh0`/`dc0` return values |
| `test_vanishing_gradient_comparison.py` | *"...owing to its more powerful update equation and some appealing backpropagation dynamics"* | the actual mechanism: seed a gradient at the *last* step of a 20-step sequence and measure how much survives back to the *first* — the vanilla RNN's has vanished (~1e-30); the LSTM's hasn't (~1e-7) |
| `test_stacked_rnn_forward_pass.py` | *Going deep* | `y1 = rnn1.step(x); y = rnn2.step(y1)` — layer shapes chain bottom-to-top correctly, and a deeper stack computes something genuinely different from a shallow one |
| `test_stacked_rnn_gradient_check.py` | *Going deep* | a numerical gradient check across every layer's weights, confirming gradients propagate correctly both through time *and* through depth |
| `test_stacked_rnn_optimizer.py` | (design property, not article prose) | clipping and Adagrad both walk every layer of the stack correctly |
| `test_stacked_lstm_forward_pass.py` | *Getting fancy* + *Going deep* | stacked gate/state shapes are correct, probability columns normalize, and batched columns match independent single-example forwards |
| `test_stacked_lstm_gradient_check.py` | *Getting fancy* + *Going deep* | central finite differences verify gradients through time and depth for every gate parameter in both layers and the output projection |
| `test_stacked_lstm_optimizer.py` | (design property, not article prose) | zeroed memory, clipping, and Adagrad updates traverse every stacked-LSTM layer and the output projection |
| `test_stacked_lstm_dropout.py` | *Going deep* | inter-layer masks are seeded and reproducible, inference disables dropout, and BPTT returns finite gradients with masks applied |
| `test_stacked_lstm_batching.py` | *Going deep* | per-column probabilities normalize; mean token loss agrees with separate examples, including batch size one |
| `test_stacked_lstm_batch_training.py` | *Going deep* | contiguous streams produce shifted targets, batched gradients pass finite differences, and seeded minibatch training lowers fixed-window loss |
| `test_delayed_copy.py` | *Long-range memory demo* | fixed delayed-copy examples, deterministic train/eval splits, and answer-only loss/accuracy that ignore the prompt and distractor span |
| `test_delayed_copy_e2e.py` | *Long-range memory demo* | the vanilla RNN and LSTM can be trained and evaluated on the same delayed-copy task with finite answer-only metrics |

## A note on the training corpus

`infrastructure/corpus.py` ships a short, repetitive, originally-written
string rather than a scraped/copyrighted text (Karpathy trained on Paul
Graham essays, Shakespeare, Wikipedia, Linux source...) — that keeps the
end-to-end test fast and license-clean. Point `load_corpus("/path/to/text")`
at any text file of your own to reproduce something closer to the post's
actual experiments; it'll just need far more than 8 training epochs to get
there.

## On char-rnn

[`karpathy/char-rnn`](https://github.com/karpathy/char-rnn) is Lua/Torch —
multi-layer LSTM/GRU, minibatched, GPU-oriented — and its own README
describes itself as "a slightly more fancy version of" the 100-line
Python/numpy gist this project was originally built from. So rather than
port Lua/Torch line-by-line, this project reimplements char-rnn's two real
upgrades over that gist (the LSTM cell, layer stacking) directly in the
same pure/DDD/gradient-checked style as everything else here.

## Long-range memory demo

The repository now includes a deterministic delayed-copy task that exercises
exactly the memory problem the LSTM is designed to handle: a short payload is
repeated only after a fixed delay filled with distractor symbols, and the
loss/accuracy are measured only on the answer positions at the end of the
sequence. Each example is bounded by dedicated start/end tokens. The helper in
`infrastructure/delayed_copy.py` builds those examples,
`application/delayed_copy_eval.py` computes answer-only metrics, and the
runner `scripts/run_delayed_copy_demo.py` trains the vanilla RNN and LSTM on
the same split for a quick baseline comparison. Run it with
`python -m scripts.run_delayed_copy_demo` from the repository root.

`test_vanishing_gradient_comparison.py` remains the mechanism-level proof for
vanishing vs. preserved gradients; the delayed-copy demo is the end-to-end
behavioral check on a task that is intentionally small and deterministic.
