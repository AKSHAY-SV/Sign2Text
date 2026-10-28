# Integrated word and alphabet demo

The integrated demo has one webcam owner: `tools/run_sign2text_integrated.py`
runs from `.venv-repair`, captures and mirrors the full BGR camera frame, and
shows the preview without landmark drawings. It starts both inference workers:
word mode uses `.venv-repair/bin/python`; alphabet mode uses
`.venv-alphabet/bin/python`. `1` selects WORD as the only text-accepting input;
`2` selects SPELL. Both predictions remain visible while the inactive worker's
commits are disabled.

The TensorFlow word runtime and ONNX Runtime alphabet environment are separate.
The UI passes full, uncompressed BGR frames through a bounded, length-prefixed
binary stdin protocol to both workers. The protocol includes a per-frame input
acceptance flag. Each worker performs its own MediaPipe processing and existing
model-specific preprocessing; neither worker opens a camera. A one-frame
latest-value queue drops stale frames under load to keep the preview responsive.
This can affect temporal continuity and motion-sensitive J/Z recognition if
inference is slower than capture. Both inference pipelines run simultaneously;
only the selected mode can commit text. No model inputs, normalization,
sequence lengths, label maps, weights, or environment packages are changed.
The UI displays per-worker dropped-frame counts; a bounded result backlog
overflow or worker/protocol failure is reported in the window and terminal.

The camera uses the backend's default capture resolution, matching the
standalone demos. The resizable window is initially sized to fit the reported
aspect ratio; display sizing does not resize frames delivered to the recognizer.

Run from the repository root:

```bash
./run_sign2text_integrated.sh
```

The integrated launcher also sends changes to the accepted text buffer to an
ESP32 over `/dev/ttyUSB0` at 115200 baud. The existing ESP32 firmware must be
running and accept UTF-8 lines terminated with `\\n`. Serial writes happen on a
background thread; an unavailable port is reported and retried while the
recognition preview continues. The sender suppresses unchanged buffer values
and bounds each line to 128 UTF-8 bytes. Disable TFT output with
`./run_sign2text_integrated.sh --no-tft`; use `--tft-port PORT` or
`--tft-baud BAUD` to override the defaults.

Only confirmed/accepted words and letters are added to the displayed text
buffer and sent. Pending live candidates are not transmitted. WORD mode appends
the model's existing word labels; SPELL mode appends the model's existing
letters. This launcher does not add sentence translation or change model
inference.

Controls: `1` WORD, `2` SPELL, `C` clear text, Backspace remove one character,
Space add a separator, `Q` quit. Accepted words append as uppercase tokens;
accepted letters concatenate. Switching from WORD to SPELL inserts one
separator when needed, so accepted HELLO followed by W/O/R/L/D predictions
produces `HELLO WORLD`. The buffer persists across mode switches. A word sign
is re-armed after no hand is detected; each word label can be accepted once
until that release. An alphabet letter is re-armed after hand release. Repeated
spelling letters therefore require a brief hand release between identical
letters. Backspace removes one character, including within an accepted word
token. Confidence scores are the
models' existing outputs and do not establish real-world accuracy.

The original standalone commands remain available for diagnostics:

```bash
./run_sign2text_words.sh
./run_sign2text_alphabet.sh
```

Both standalone previews hide landmark drawings by default. Add
`--show-landmarks` to either command for a diagnostic overlay; this only changes
drawing and leaves the inference inputs untouched.
