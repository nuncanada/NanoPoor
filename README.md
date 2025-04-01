# NanoPoor
NanoGPT-speedrunning for the poor T4 enjoyers

[Colab Notebook](https://colab.research.google.com/drive/1x87U-mCZCt7Kwc5-HGPOR1NVCOYAN1dr?usp=sharing) 

Inspired by [Modded NanoGPT](https://github.com/KellerJordan/modded-nanogpt) and my goat [Jonas Geiping (Cramming)](https://arxiv.org/pdf/2212.14034), I trained a custom GPT I've been working on over at [Dagonet](https://github.com/BambooML/Dagonet), got to the 3.28 val loss on a single T4.

some caveats:
 - This is an MoE, with 120M params, but ~75M active, @leloykun from the actual NanoGPT speedrunning has said that its 120M active params, so fitting more is free game if you can
 - used GPT-4 tokenizer (think GPT-2 would be better anyway because the models 124M, but CE loss is not comparable), I'm keeping that because I'm using this code in other models that benefit from this size
 - was just a 1B subset of finewebedu10b, not filtered or anything I just processed that much at this time, will probably fix this later

## Runs

| Ranking  | Time    | Data | Person | Description | log |
| -------- | ------- | ---- | ------ | ----------- | --- |
| 1st      | 37.17m  | ~14M tok (1024*6*4*570)| Vatsa  | Added PSGD | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/GPT4-tok-run.txt) |
| 2nd      | 70.61m  | ~6.14M tok (1024*5*4*300) | Vatsa  | First Run, has DS-MoE, MLA+NSA hybrid, Rope, etc | [log](https://github.com/VatsaDev/NanoPoor/blob/main/logs/PSGD_run.txt) |

