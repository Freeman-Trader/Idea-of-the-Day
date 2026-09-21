**Check list**
* A pipeline to create a random but grammatically accurate sentence and store it
    * Needs to run daily or whenever a new commit is pushed to main
* A pipeline to compile and publish a website
* A place to store all thoughts as well as the most recent

## Run locally

The generator loads a Transformers model directly into the Python process. Install
the dependencies and place a downloaded model at
`models/SmolLM2-360M-Instruct`:

```sh
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-360M-Instruct', local_dir='models/SmolLM2-360M-Instruct')"
python idea_generator.py
```

Set `LOCAL_MODEL_PATH` to use a different local model directory. The script uses
`local_files_only=True`, so it never contacts a model host at runtime.
