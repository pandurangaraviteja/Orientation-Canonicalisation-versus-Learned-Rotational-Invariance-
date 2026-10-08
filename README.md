[Uploading README.md…]()
# Orientation canonicalisation: complete Python implementation

This package implements the methods and all 15 figure outputs used in the attached manuscript: one architecture figure and 14 result figures. It trains HOG + MLP, HOG + Random Forest, and a compact CNN, each with raw and canonicalized inputs. It evaluates the same frozen models under controlled rotations, records predictions, and generates the figures in PNG, PDF, and SVG formats.

**This is a runnable reimplementation, not a reconstruction of the historical experimental run.** The manuscript package contained plots and aggregate results but no image archive, split manifest, training scripts, or checkpoints. The code contains no hardcoded published accuracy, AUC, F1, or confusion-matrix values. Those numbers must be computed from actual images and predictions.

## 1. Use the right dataset

Default compatible source:

**Wani, Insha Majeed; Arora, Sakshi (2021). Knee X-ray Osteoporosis Database. Mendeley Data, version 2. DOI: [10.17632/fxjm8fb6mw.2](https://doi.org/10.17632/fxjm8fb6mw.2).**

Official download page: <https://data.mendeley.com/datasets/fxjm8fb6mw/2>. The source page specifies CC BY 4.0 and describes knee X-rays, clinical information, and QUS-derived T-scores. Preserve its clinical label provenance. These are dataset labels, not new diagnostic measurements made by this code.

The original attached manuscript still cites the **Knee Osteoarthritis Dataset with Severity Grading** on Kaggle. That citation must not be used to silently reinterpret osteoarthritis grades as Normal/Osteopenia/Osteoporosis. The source DOI above identifies a relevant public osteoporosis dataset; it does not establish the origin of the paper's exact 1,966 + 220 working split.

Download the **original images**, preserving patient metadata. Do not increase the cohort using augmented images, mirrored copies, crops, or rotations before partitioning. The code reports the counts actually present. It never expands this release to force the manuscript totals or the reported 81/53/86 test distribution.

The original data could not be downloaded in this execution environment: the download endpoint returned HTTP 403. No real medical images are bundled. Software validation was performed with explicitly labelled synthetic geometric phantoms. This is not real-data validation of the paper's conclusions.

If the download is a ZIP:

```bash
python dataset_setup.py --archive /path/to/official_dataset.zip --destination data/mendeley_original
```

For RAR or another format, extract it with your archive application. `dataset_setup.py` also supports `--download-url` when you copy an exact, current HTTPS ZIP link from the official download page. The program does not invent a download API or require account credentials.

For Google Colab, upload `Colab_Run_All_Plots.ipynb` from this package into Colab. It guides you through uploading the code ZIP, mounting your own Drive folder, preparing the patient manifest, training, displaying all figures, and downloading results.

## 2. Install Python packages

Python 3.10 or newer is required. Run these commands inside the extracted code folder:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux, macOS, or the Chromebook Linux terminal:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The CPU installation avoids installing the CUDA runtime on a machine without an NVIDIA GPU. For a supported GPU, choose the matching PyTorch installation command from <https://pytorch.org/get-started/locally/> and then install the remaining requirements. The code uses `--device auto`, `--device cpu`, or `--device cuda`.

The validation environment and exact installed package versions are recorded in `VALIDATION.json`. Those versions are an execution record, not a requirement to install a large CUDA stack. `requirements.txt` specifies compatible version ranges; every experiment records the versions actually used.

## 3. Prepare a trustworthy manifest

The class order is fixed:

| Label | Dataset category |
|---|---|
| 0 | Normal |
| 1 | Osteopenia |
| 2 | Osteoporosis |

There are two supported routes.

### Route A: existing authoritative train/validation/test split

Create a CSV containing at least these columns:

```csv
path,label,patient_id,split
/absolute/path/Normal/patient001_view1.jpg,0,patient001,train
/absolute/path/Osteopenia/patient002_view1.jpg,1,patient002,val
/absolute/path/Osteoporosis/patient003_view1.jpg,2,patient003,test
```

These three rows illustrate the format; a real manifest must contain all three classes in each split. Obtain labels and patient IDs from source metadata. Relative image paths are resolved against the manifest's parent directory. All images/views of the same patient must use the same split. Reusing a supplied authoritative manifest is the correct route for historical reproduction.

The validator checks file existence, labels, patient overlap, repeated file paths, file hashes, and exact decoded-pixel duplicates. If a frozen manifest already contains hashes, content changes are rejected. Near duplicates and undocumented augmentation derivatives require a separate provenance review; a hash check does not prove patient independence.

### Route B: construct a new patient-group split

Arrange original images into folders named `Normal`, `Osteopenia`, and `Osteoporosis`. This step organizes existing labels; it does not create or relabel data.

If you have confirmed that source filenames begin with the actual patient code, provide a regular expression with a capturing group for that code. Example **only after confirming the source convention**:

```bash
python run.py prepare --data data/mendeley_original --manifest data/manifest.csv --patient-regex "^((?:N|OP|OS)[-_ ]?[0-9]+)"
```

The expression must preserve patient identity across both-knee/one-knee images and repeated views. A filename prefix is not automatically an authoritative patient ID. If the source uses a different naming convention, use its spreadsheet to build Route A rather than guessing. The scanner deliberately refuses files whose names do not match the supplied expression.

Default new partitions are 70%/15%/15% of patient groups, stratified by label. Image fractions can differ because group sizes differ. No original folder split is silently treated as the paper's split: Route B explicitly creates a new split. Change fractions using `--test-fraction` and `--val-fraction`.

An optional `--allow-image-split` flag permits exploratory image-level splitting when identities are unavailable. Its audit records that patient independence is unverified; do not describe such output as patient-independent validation. This flag is never enabled by default.

DICOM is not silently converted. DICOM-based cohorts need a separate image-to-patient join, diagnostic labels, and a specified intensity/windowing protocol before using this pipeline.

## 4. Train and generate every plot

```bash
python run.py run --manifest data/manifest.csv --out results/mendeley_seed42 --device auto
```

This command trains six configurations, selects neural checkpoints exclusively by validation loss, evaluates the held-out images and seven controlled angles, calculates uncertainty, and generates every figure. The training log prints each model and rotation stage. Use an empty output directory so previous runs remain intact.

For a quick first run:

```bash
python run.py run --manifest data/manifest.csv --out results/quick_check --epochs 5 --device cpu
```

A short run checks execution; it is not sufficient evidence for final research results. The default maximum is 60 epochs with patience 10. Edit `config.json` before research runs and retain the resulting `run_metadata.json`. CPU execution works; a GPU reduces CNN training time. Runtime depends on image count and hardware.

To regenerate figures from completed predictions without training again:

```bash
python run.py plots --run results/mendeley_seed42 --device cpu
```

## 5. All figures produced

The exact original filenames are preserved, including the historical `real` prefix. In synthetic smoke runs, that prefix does **not** mean the input is real: every figure carries a synthetic-test footer.

| Manuscript filename | What the code computes | Model/input |
|---|---|---|
| `fig_arch.png` | Deterministic block diagram of the evaluated protocol | Entire pipeline |
| `fig_real_cm_both.png` | Counts and row-normalized confusion matrices | MLP raw/canonicalized |
| `fig_real_perclass.png` | Precision, recall, F1, and support | MLP canonicalized |
| `fig_real_roc.png` | One-vs-rest ROC curves and AUC | MLP canonicalized |
| `fig_real_modelcmp.png` | Accuracy, macro-F1, and QWK bars | MLP/RF, both input variants |
| `fig_real_compare.png` | Upright accuracy, macro-F1, and QWK comparison | MLP raw/canonicalized |
| `fig_real_robust.png` | Accuracy versus controlled rotation | MLP raw/canonicalized |
| `fig_real_robust_perclass.png` | Per-class recall versus rotation | MLP raw/canonicalized |
| `fig_cnn_gradcam.png` | Actual last-convolution Grad-CAM overlays | CNN canonicalized |
| `fig_cnn_robust.png` | Accuracy versus controlled rotation | CNN raw/canonicalized |
| `fig_real_calib.png` | Top-label reliability diagram and ECE | MLP raw/canonicalized |
| `fig_real_uncert.png` | Error rate across entropy quintiles | MLP canonicalized |
| `fig_real_entropy.png` | Entropy distributions by correctness and Spearman correlation | MLP canonicalized |
| `fig_real_pca.png` | Two-dimensional PCA of train-fitted standardized HOG | Canonicalized training subset |
| `fig_real_pr.png` | One-vs-rest precision–recall curves and AP | MLP canonicalized |

The `architecture/` directory also includes the exact editable TikZ architecture from the revised manuscript, its standalone LaTeX wrapper, and its vector PDF. The Python diagram provides a dependency-free renderer for the same protocol. To compile the original TikZ version, run `pdflatex architecture_standalone.tex` inside `architecture/`.

## 6. Saved outputs and how to use them

| Output | Purpose |
|---|---|
| `figures/` | 15 figures × PNG/PDF/SVG; PNGs use 300 dpi |
| `plot_data/` | Confusion counts, curve points, reliability bins, quintiles, PCA coordinates, Grad-CAM arrays and example-selection records |
| `predictions/` | Per-image class probabilities, labels, patient IDs, hashes, angles, predictions, severity scores, MC mean probabilities, entropy and mutual information |
| `models/` | Selected neural weights, RF estimators, train-fitted scalers and PCA |
| `metrics_summary.csv` | Actual accuracy, macro-F1, QWK, ECE, ordinal MAE/RMSE and severe-error rate |
| `metrics_*.json` | Full per-class metrics and entropy/error association |
| `rotation_metrics.csv` | All six configurations at all seven angles, including per-class recall |
| `rotation_estimator_diagnostics.csv` | Estimated angles, detection failures, and estimator-relative residuals |
| `paired_mlp_bootstrap.csv` | Paired patient-cluster bootstrap intervals for upright MLP differences |
| `paired_mlp_statistics.json` | Discordant prediction counts; exact McNemar p-value only if one image per patient |
| `run_metadata.json` | Parameters, software, split counts, seed, device, source and protocol assumptions |
| `dataset_audit.json` / `manifest.csv` | Frozen dataset membership and integrity checks |
| `training_summary.json` | Epochs, validation loss, parameter counts and measured training time |
| `SHA256SUMS.txt` | Integrity checksums for the run artifacts |

To replace manuscript results, copy the **newly generated** PNGs into a separate manuscript version's `figures/` folder, retaining the exact filenames. Revise the results tables, abstract, discussion, captions and counts using the corresponding newly computed CSV/JSON files. Copying new figures while retaining old numeric claims would create an inconsistent paper.

## 7. Method and implementation choices

The paper specifies 128 × 128 grayscale images, CLAHE, Canny/Hough orientation estimation, bilinear rotation with zero padding, HOG with 9 bins / 16 × 16 cells / 2 × 2 blocks / L2-Hys, MLP hidden sizes 256 and 128, 300 RF trees, and three CNN convolutional blocks with batch normalization, ReLU, pooling, global average pooling, dropout and Softmax probabilities. These are implemented directly.

Several original settings were not supplied. **The following are documented reimplementation defaults, not recovered historical settings:** CLAHE clip 2 and 8 × 8 tiles; Canny 50/150; Hough threshold 35, 0.5-degree angular resolution, search within 40 degrees of upright and 4-pixel boundary exclusion; dropout 0.3; CNN channels 16/32/64; AdamW at 0.001, weight decay 0.0001, batch size 32; ±15-degree training rotations; horizontal flip probability 0.5 and no vertical flips; 60 maximum epochs; patience 10; seed 42; 30 MC passes; 10 ECE bins. Parameters are explicit in `config.json`.

The Hough detector picks the strongest eligible line, not a clinically verified anatomical landmark. If no eligible line is found, it applies zero correction and records the failure. A line identifies an axis modulo 180 degrees. This is an upright-orientation correction, not a solution for arbitrary directed pose, homography, or three-dimensional anatomy. Its thresholds and upright prior make it parameterized; do not describe this implementation as parameter-free.

Rotation signs are tested using known geometric phantoms. In OpenCV's x-right/y-down image coordinates, a counterclockwise image rotation produces a negative Hough normal angle near a vertical axis. The code estimates the CCW image angle as the negative normal angle and rotates by its negative to correct it. Controlled test rotations are applied after common preprocessing and before canonicalization, isolating the geometric perturbation. Canonicalized CNN training images receive the same augmentation distribution as raw CNN training images.

Feature standardization and PCA are fitted on training images only. Raw and canonicalized models are trained separately with matched seeds and splits. CNN augmentation occurs only during training. The best neural checkpoint is selected by weighted validation cross entropy. No test labels are used to tune thresholds, augmentation, calibration, or training duration.

Classification and robustness plots use deterministic predictions. Entropy plots use averaged MC-dropout probabilities for the MLP and associate their entropy with the deterministic prediction's error. Only dropout is activated for MC sampling; BatchNorm is frozen and inference states are restored. RF entropy is descriptive; the code does not pretend that deterministic RF predictions provide MC-dropout epistemic uncertainty.

The PCA plot is a training-only visualization. Grad-CAM selects the highest-confidence correctly classified example per class and records each image hash. If a class has no correct example, the panel explicitly says so; no image or heatmap is fabricated. A zero positive Grad-CAM response is also labelled.

Equal-sized entropy quintiles use a stable ordering, so tied scores can occur in adjacent groups. Their empirical error rates are not forced to rise. Spearman correlation is undefined when entropy or error is constant and is stored as null in JSON. QWK and other quantities may also be undefined for a degenerate bootstrap draw; those draws are excluded and their valid count is reported.

The shaded gap in the rotation plot is a numerical difference, **not a confidence interval**. Estimator-relative residuals compare the current estimate with the upright estimate for the same image; they are not expert-labelled anatomical errors. Patient-cluster bootstrap intervals preserve views and raw/canonicalized pairing. They describe the test cohort conditional on a fitted model, not model-initialization uncertainty.

## 8. Optional repeated runs

```bash
python run.py run --manifest data/manifest.csv --out results/seed42 --seed 42
python run.py run --manifest data/manifest.csv --out results/seed43 --seed 43
python run.py run --manifest data/manifest.csv --out results/seed44 --seed 44
python run.py run --manifest data/manifest.csv --out results/seed45 --seed 45
python run.py run --manifest data/manifest.csv --out results/seed46 --seed 46
```

Keep the manifest fixed for initialization comparisons. Changing the split seed at the same time would confound the two sources of variation. This package exports each run; it does not claim an uncomputed multi-seed result or multiple-testing correction.

## 9. Run the software checks

```bash
python -m unittest discover -s tests -v
python run.py smoke --out smoke_check --epochs 3
```

The smoke command constructs geometric phantoms, trains the six configurations with shortened settings, and exercises every figure generator. All its images and figures are marked as synthetic software-test output. It does not estimate clinical performance and must not be substituted into the manuscript as experimental evidence.

## References

1. Wani IM, Arora S. Knee X-ray Osteoporosis Database. Mendeley Data v2, 2021. DOI 10.17632/fxjm8fb6mw.2. <https://data.mendeley.com/datasets/fxjm8fb6mw/2>
2. OpenCV geometric image transformations: <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html>
3. scikit-image HOG: <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog>
4. PyTorch Dropout: <https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html>
5. scikit-learn classification metrics: <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics>
6. Selvaraju RR et al. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017, pp. 618–626. <https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html>
