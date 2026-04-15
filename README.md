# Unified Histology Pipeline

This folder contains the new unified pipeline for histology processing.

It can run three stages:
- `clahe`
- `stain`
- `nuclei`

By default, `--all` runs all three.

## Files

- [run_slide.py](run_slide.py): run the pipeline for one slide
- [run_folder.py](run_folder.py): run the pipeline for a whole folder of slides
- [run_slide_job.sh](run_slide_job.sh): submit one slide to SLURM with `sbatch`
- [pipeline.py](pipeline.py): main implementation

## Output Layout

For one slide, output looks like:

```text
<OUTPUT>/<SLIDE_NAME>/
  meta.json
  clahe/
  stain/
  nuclei/
```

Raw extracted arrays stay in their original tile-grid resolution:
- `stain/*_map.npy`
- `nuclei/nuclei_density_map.npy`

Only the preview TIFFs are transformed to a fixed `1024 x 1024` canvas.

## Pipeline Schematic

```mermaid
flowchart TD
    A[Input WSI slide<br/>.svs .tif .ndpi .mrxs ...] --> B{Selected stages}
    B -->|clahe| C[OpenSlide thumbnail branch]
    B -->|stain and/or nuclei| D[large_image tile iterator<br/>magnification + tile_size]

    C --> C1[Load associated thumbnail]
    C1 --> C2[Convert RGB to grayscale]
    C2 --> C3[Tissue mask with histomicstk]
    C3 --> C4[Fit while preserving aspect ratio]
    C4 --> C5[Center-pad to 1024x1024]
    C5 --> C6[Gamma correction]
    C6 --> C7[CLAHE enhancement]
    C7 --> C8[Threshold cleanup]
    C8 --> C9[Write clahe/<stem>_0000.tif]

    D --> E[Tile as NumPy array]
    E --> F[Shared tile metadata<br/>coordinates level overlap]
    F --> G{Per-tile branches}

    G -->|stain| H[Positive pixel count]
    H --> H1[Compute stain metrics per tile]
    H1 --> H2[Collect tile metric table]
    H2 --> H3[Build tile maps]
    H3 --> H4[Resize preview to thumbnail space]
    H4 --> H5[Fit and pad preview to 1024x1024]
    H5 --> H6[Write stain/*_map.npy and stain/<stem>_0002.tif]

    G -->|nuclei| I[Preprocess tile]
    I --> I1[Drop alpha if present]
    I1 --> I2[Check tile overlap with global tissue mask]
    I2 --> I3[Reinhard normalization<br/>using reference tile]
    I3 --> I4[Color deconvolution]
    I4 --> I5[Take hematoxylin channel]
    I5 --> I6[Threshold foreground]
    I6 --> I7[Fill holes + remove small objects/holes]
    I7 --> I8[Connected components]
    I8 --> I9[Count nuclei per tile]
    I9 --> I10[Build nuclei density map]
    I10 --> I11[Resize preview to thumbnail space]
    I11 --> I12[Fit and pad preview to 1024x1024]
    I10 --> I13[Write nuclei_density_map.npy]
    I12 --> I14[Write nuclei/<stem>_0001.tif]
    C9 --> I15[Write <sample>/meta.json with reversible transforms]
    H6 --> I15
    I14 --> I15
```

## Step-By-Step

### 1. Input

Input to the pipeline is one WSI file, for example:

```text
../../images/00-1006-A-IBA1.svs
```

The pipeline can run one or more stages:
- `clahe`
- `stain`
- `nuclei`

### 2. CLAHE Branch

This is a thumbnail-based branch, not a full-tile branch.

Input:
- whole-slide thumbnail from `OpenSlide`

What happens:
- load `associated_images["thumbnail"]`
- convert RGB thumbnail to grayscale
- detect tissue mask with `histomicstk.saliency.tissue_detection.get_tissue_mask`
- keep only tissue pixels
- fit to the `1024 x 1024` canvas while preserving aspect ratio
- center-pad to `1024 x 1024`
- apply gamma correction
- apply CLAHE
- zero out very small values

Output:
- `clahe/<stem>_0000.tif`
- transform metadata in `<sample>/meta.json`

### 3. Shared Tile Iterator

This is the main full-slide branch for `stain` and `nuclei`.

Input:
- same WSI file
- analysis magnification
- tile size

What happens:
- `large_image` opens the slide
- `tileIterator(...)` walks across the slide
- each tile is returned as a NumPy array
- tile metadata is preserved:
  - tile coordinates
  - global coordinates
  - scale info
  - overlap info
  - iterator position info

This shared iterator is the core optimization:
- the slide is traversed once
- stain and nuclei are both computed from the same tile buffer
- tile inclusion is decided from one shared precomputed tile grid derived from a global thumbnail tissue mask

### 4. Stain Branch

Input:
- tile NumPy array
- tile metadata
- positive pixel count parameters

What is calculated:
- `histomicstk.segmentation.positive_pixel_count.count_image(...)`
- per-tile staining metrics including values like:
  - `IntensitySumPositive`
  - `NumberPositive`
  - `NumberTotalPixels`
  - derived `PercentagePositive`

What happens after all tiles:
- tile-level results are collected into a DataFrame
- metric grids are built by tile position
- raw stain maps stay at tile-grid resolution
- stain preview TIFF is resized into thumbnail space, then fit+pad to `1024 x 1024`

Output:
- `stain/IntensitySumPositive_map.npy`
- `stain/NumberPositive_map.npy`
- `stain/PercentagePositive_map.npy`
- `stain/<stem>_0002.tif`
- transform metadata in `<sample>/meta.json`

The stain TIFF preview is currently generated from `PercentagePositive` only, using a fixed display scale:
- min = `0.0`
- max = `0.02` (2%)

That fixed range is also stored in `<sample>/meta.json`.

### 5. Nuclei Branch

Input:
- tile NumPy array
- tile metadata
- reference tile `.npy` for normalization

What is calculated per tile:
- remove alpha channel if present
- check one shared tile keep-mask derived from the thumbnail tissue mask
- skip only tiles with zero tissue-mask support in that precomputed grid
- apply Reinhard color normalization
  - this normalizes tile color distribution to match a reference tile
  - goal: reduce stain/color variation across slides
- perform color deconvolution
  - separate stain channels using a stain matrix
- extract hematoxylin channel
  - this acts as the nuclei-focused channel
- threshold foreground
- fill holes
- remove small objects
- remove small holes
- run connected-component labeling
- count labeled nuclei objects

What happens after all tiles:
- nuclei counts are written into a 2D tile-grid density map
- raw nuclei map stays at tile-grid resolution
- nuclei preview TIFF is resized to thumbnail size
- nuclei preview TIFF is then fit+pad to `1024 x 1024`
- metadata is saved

Output:
- `nuclei/nuclei_density_map.npy`
- `<sample>/meta.json`
- `nuclei/<stem>_0001.tif`

`meta.json` includes reversible transform metadata for each preview TIFF:
- the embedded image size inside the `1024 x 1024` canvas
- padding offsets
- crop coordinates for extracting predictions back out of the padded image
- the raw target map size to resize predictions onto `stain_map.npy` or `nuclei_density_map.npy`

### 6. Parallelism

The tile branch currently uses:
- `ProcessPoolExecutor`
- one process per worker
- one shared tile iterator in the parent process
- stain/nuclei computation in child processes

This is meant to avoid GIL-limited CPU parallelism for the heavy per-tile work.

### 7. Logging and Performance Signals

The pipeline logs:
- stage start/finish
- tile count
- periodic progress
- ETA
- tile wall time
- child CPU time
- parallelism efficiency

These help answer:
- is the pipeline CPU-bound?
- is it mostly waiting on slide I/O?
- are multiple workers actually being used effectively?
- how many tiles were skipped by the tissue mask

## Environment

From inside this `unified_pipeline/` folder, use:

```bash
../../python-venvs/histomics-env/bin/python
```

## Run One Slide

Run everything:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_slide.py \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  --all
```

Run only `stain` and `nuclei`:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_slide.py \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  --stain --nuclei
```

Run only `clahe`:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_slide.py \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  --clahe
```

## Run One Slide Through SLURM

Run everything:

```bash
sbatch ./run_slide_job.sh \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  all
```

Run only `stain+nuclei`:

```bash
sbatch ./run_slide_job.sh \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  stain+nuclei
```

Available modes for `run_slide_job.sh`:
- `all`
- `clahe`
- `stain`
- `nuclei`
- `stain+nuclei`

## Run Whole Folder

Run the whole `images/` folder:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_folder.py \
  ../../images \
  ../../histoprocessor_outputs/unified_batch \
  --all
```

Example with explicit parameters:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_folder.py \
  ../../images \
  ../../histoprocessor_outputs/unified_batch \
  --stain --nuclei \
  --workers 16 \
  --tile-size 512 \
  --magnification 40
```

## Important Behavior

- `run_slide.py` overwrites output files in the target output folder
- `run_folder.py` skips a slide if its output folder already exists
- `run_folder.py --force` reruns slides even if output already exists

## Useful Options

- `--all`
- `--clahe`
- `--stain`
- `--nuclei`
- `--workers`
- `--tile-size`
- `--magnification`
- `--reference-image`
- `--force` for `run_folder.py`

## Logging

SLURM and pipeline logs now include:
- timestamps
- `[histoprocessor]` prefix
- CLAHE start/finish
- tile-pass start
- tile count
- progress updates with ETA
- tile wall time
- child CPU time
- parallelism efficiency

This helps tell whether the pipeline is CPU-bound or I/O-bound.

## Typical Commands

Run one slide locally:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_slide.py \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  --all
```

Run one slide with SLURM:

```bash
sbatch ./run_slide_job.sh \
  ../../images/00-1006-A-IBA1.svs \
  ../../histoprocessor_outputs/unified_test/00-1006-A-IBA1 \
  all
```

Run whole folder:

```bash
../../python-venvs/histomics-env/bin/python \
  ./run_folder.py \
  ../../images \
  ../../histoprocessor_outputs/unified_batch \
  --all
```

Run whole folder through SLURM:

```bash
sbatch --wrap="../../python-venvs/histomics-env/bin/python ./run_folder.py ../../images ../../histoprocessor_outputs/unified_batch --all"
```
