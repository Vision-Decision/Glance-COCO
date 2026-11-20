# Data license

The **code** in this repository is licensed under Apache-2.0 (see `LICENSE`).

The **data** is licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
and may be used for **non-commercial research only**. "Data" means:
- the Glance-COCO dataset on the Hugging Face Hub (`Vision-Decision/Glance-COCO`, formerly `Vision-Decision/Glance-Data`);
- every file this pipeline produces (`out/`, `aug_*.jsonl`, Parquet exports);
- the data-derived files in this repository: `docs/results/` and `hf/DATASET_CARD.md`.

## Why this license

The data is derived from the sources below. Their terms carry over to it, and gRefCOCO's ShareAlike clause requires
derived data to use the same license.

| Source | Terms | Used for |
|---|---|---|
| [gRefCOCO](https://henghuiding.github.io/GRES/) (Liu, Ding and Jiang, CVPR 2023) | CC BY-NC-SA 4.0; data released for non-commercial research purposes only | multi-target sentences (`gref_multi`); single-target sentences used as generation input |
| RefCOCO / RefCOCO+ (Yu et al., 2016), RefCOCOg (Mao et al., 2016) | research use; the official [refer](https://github.com/lichengunc/refer) toolkit is Apache-2.0 | human descriptions used as generation input |
| [COCO 2014](https://cocodataset.org/#termsofuse) annotations | CC BY 4.0 | object boxes and categories (ground truth) |
| COCO images | the license of each Flickr photo | **not redistributed**; rows reference images by `image_id` / `file` |
| DeepSeek (deepseek-flash) outputs | DeepSeek terms of use | wording of generated questions |

## Attribution

If you use the data, cite gRefCOCO (GRES), RefCOCO / RefCOCO+ / RefCOCOg and COCO, and link to this repository.

The README example image is COCO image 335624
([Flickr](http://farm7.staticflickr.com/6214/6266504428_369fe0f165_z.jpg), CC BY 2.0); the boxes and numbers are
our additions.
