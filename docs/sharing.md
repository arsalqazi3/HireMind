# Sharing large files

Code lives on GitHub. Everything git ignores (data, splits, the index, model weights) is shared through one Google Drive folder, so nobody downloads or rebuilds the same thing twice.

Shared folder: [HireMind-shared](https://drive.google.com/drive/folders/162kfIrXeEE7Ku_BMO7dpCmVxHOor4Jnb). It is owned by Arslan, and Hassaan is an editor.

## What needs downloading, and how often

| File | Size | Download once per laptop? | Share instead of rebuilding? |
|---|---|---|---|
| Kaggle datasets | ~650 MB | Yes. Arslan has them. | Hassaan downloads once with his own Kaggle token, or copies Arslan's |
| Processed data and splits | ~55 MB | No need | Yes. Both of you must use the exact same split |
| Embedding model | ~90 MB | Yes, it stays in `~/.cache/huggingface` | Not worth sharing |
| FAISS index | ~15 MB | No need | Yes, so you both search the same index |
| Fine-tuned model (later) | Small adapter, larger if merged | No need | Yes. Training happens once on Colab |

## Folder layout

The Drive folder mirrors the repo, so files go in the same place in both:

```
HireMind-shared/
  data/processed/   resumes.csv, jobs.csv
  data/splits/      train.txt, val.txt, test.txt
  models/index/     jobs.faiss, job_ids.npy
  models/slm/       the fine-tuned model, later
  labels/           relevance_arslan.xlsx, relevance_hassaan.xlsx (see docs/relevance_labelling.md)
```

`labels/` has no matching repo folder. Copy the finished workbooks into `data/labels/` before importing them.

## How we work

- **Code stays on GitHub**, as now.
- **Build once, share the result.** One person builds or trains, uploads to the matching Drive folder, and the other copies it into the same folder in their repo.
- **Colab reads from Drive.** Colab wipes its machine every session, so mount Drive instead of downloading again. Only the base model is re-downloaded each session (a few hundred MB, fast on Colab).
- **Only one person rebuilds shared files.** The scripts are seeded, so a rebuild should match, but uploading one copy guarantees it. If a rule in `docs/category_mapping.md` or `hiremind/skills.csv` changes, that person rebuilds, re-uploads and tells the other.

## Don't

- **Put data or weights in git.** The repo is not built for large files, and a public repo would republish the Kaggle datasets.
- **Use Git LFS.** GitHub's free LFS allowance is small and would run out quickly.
