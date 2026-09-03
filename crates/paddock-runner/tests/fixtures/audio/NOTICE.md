# Audio fixtures - sources and licenses

## `ls-long-concat.wav`

- **Source:** LibriSpeech ASR corpus, `test-clean` split - utterances
  **6930-75918-0000 ... -0003** concatenated in order (one speaker, one chapter,
  contiguous speech), via `openslr/librispeech_asr` on the HuggingFace
  datasets-server. Built by our ASR oracle tool, which calls it
  `ls-long-concat`.
- **License:** CC BY 4.0 - LibriSpeech, Panayotov, Chen, Povey and Khudanpur,
  ICASSP 2015. <https://www.openslr.org/12>
- **Encoding:** 16 kHz mono PCM16, 46.07 s, 1.4 MB - bit-identical to the
  source (already 16 kHz/16-bit, round-tripped as int16 with no float pass).
- **Reference text:** `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS THE
  ENGLISH FORWARDED TO THE FRENCH BASKETS OF FLOWERS ...` (the four utterances
  joined; the manifest in the battery carries it in full).

Why this clip and not one of the synthetic tones in
`crates/paddock-engine/tests/data/asr-mel/`: the gate asserts things a
transcript of a sine wave cannot have - segments that exist, times that
advance and do not overlap, words that rejoin to their segment's text. Silence
makes every one of those vacuously true, which is a gate that cannot fail.

Why 46 s and not a 4 s utterance: whisper encodes in fixed 30 s windows, so a
clip this long spans TWO of them and the second window's times are the first's
plus an offset. That seam is where segment timing actually breaks, and a clip
inside one window never touches it. It also leaves the last window
zero-padded, which is what the end-of-clip clamp exists for.

Nothing here is scored. The gate checks WIRE SHAPE, never accuracy; WER lives
in our ASR oracle tool, run against the full audio batteries separately.
