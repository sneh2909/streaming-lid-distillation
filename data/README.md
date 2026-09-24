# Data — Applied ASR take-home (v2)

**No data ships with this assignment.** Language ID is trivial to get audio for, and part of the
exercise is that you set up a small, honest evaluation of your own. Bring or synthesize a handful
of clips — you do **not** need a large corpus; this is about the distillation machinery and the
pipeline design, not scale.

### What you need

- **A few monolingual clips per language** — enough to run the teacher, form distillation targets,
  and show one training step. A dozen short clips is plenty.
- **At least one self-concatenated multi-language clip** — stitch, say, ~4 s of Hindi followed by
  ~4 s of English into a single file. This is a cheap, controllable stand-in for a **mid-call
  language switch**, and the switch boundary is exactly where a streaming LID is hard (how fast it
  detects the change, whether it flip-flops). Use these to eyeball your student's / teacher's
  streaming behaviour.

### Where to get it

Any of these work — pick whatever is fastest for you:

- **VoxLingua107** (the training data behind the common ECAPA LID teacher), **FLEURS** (has several
  Indian languages), or **Common Voice** — all have Hindi and English.
- A few clips you record or download yourself.
- Teacher-generated / TTS audio if you just need something to push through the pipeline.

**Languages:** English + at least one Indian language; **Hindi↔English** is the case we care about
most. State what you used and why.

### Notes

- There are **no held-out labels** here — you own the tiny eval. Say how you'd scale it honestly.
- Keep whatever you use **out of your submission** unless it's small; a script or a one-line
  pointer to the dataset is better than committing audio.
