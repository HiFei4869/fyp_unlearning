This repository is the code for IERG4999 final year project.


## Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):
```bash
uv sync
```

The `requirements.txt` file is also present for pip setup.

## Running the project

All available arguments may be listed using:
```bash
uv run src/unlearn.py --help
```

Important arguments:
```bash
  --model "Llama-2-7b-chat"       Model to use.
  --logdir LOGDIR       Logdir.

  --batch_size BATCH_SIZE
                        Batch size.
  --epochs EPOCHS       Number of unlearning epochs.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --beta BETA           Beta for NPO loss.
  --npo_mult NPO_MULT   NPO forget loss multiplier.
  --rt_mult_a RT_MULT_A Parameter a for retain loss control
  --rt_mult_b RT_MULT_B Parameter b for retain loss control
  --rt_mult_c RT_MULT_C Parameter c for retain loss control
  --kl_mult KL_MULT     KL divergence retain loss multiplier.
  --lora_rank LORA_RANK
                        Rank of the LoRAs.
  --lora_alpha LORA_ALPHA
                        The LoRA alpha parameter. None means alpha=rank.
  --save_every SAVE_EVERY
                        Save checkpoint every n epochs. `-1` means never.
  --save_model, --no-save_model
                        Save model after training.
```
