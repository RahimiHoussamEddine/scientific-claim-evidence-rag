# Data and model notices

Project source code is MIT licensed.

SciFact source data are provided by AllenAI and licensed CC BY-NC 2.0. Data are downloaded by the user into ignored local paths; this repository does not redistribute the dataset. Source: https://huggingface.co/datasets/allenai/scifact. Paper: Wadden et al. (2020), https://aclanthology.org/2020.emnlp-main.609/.

The default text generator is google/flan-t5-small, licensed Apache-2.0 by its upstream model card: https://huggingface.co/google/flan-t5-small. It is downloaded separately and is not redistributed here.

The dense text encoder is sentence-transformers/all-MiniLM-L6-v2, licensed Apache-2.0 by its upstream model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Its revision is recorded in data/source_manifest.json; weights are downloaded separately and are not redistributed here.
