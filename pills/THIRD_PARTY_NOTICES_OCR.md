# Third-party components

The PARSeq implementation in `src/vendor_parseq` is Copyright 2022 Darwin Bautista and distributed under Apache License 2.0. Its original license, notice and upstream provenance are included in that directory.

Upstream: https://github.com/baudm/parseq

The pill adapter expands the character set/length, supports a 320×320 square input and a wider local model. The portable distribution additionally changes path/configuration/device handling; its original v12 training source is retained under `provenance/original_training_source`.

PyTorch, torchvision, timm, NumPy, Pillow, Matplotlib and filelock are installed separately by pip under their respective licenses. Their binaries are not bundled in this ZIP.

The included checkpoints derive from the PARSeq OCR initialization and the user's local pill training data. Dataset photographs as a whole are external; seven original photographs are included solely as inference/training smoke examples, with their original GT and split retained.
