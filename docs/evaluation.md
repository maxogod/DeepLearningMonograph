# Evaluation results

Class frequencies in the validation dataset (20% of the total data):
- [Background     NCR        ED         ET    ]
- [0.9516592  0.00787832 0.03012024 0.01034224]


## Best weighted annealing model with 133 epochs

```bash
2026-04-07 15:29:20,292 | INFO | src.utils.logger | Starting evaluation...
2026-04-07 15:29:20,292 | DEBUG | src.utils.logger | Found 63 volume files and 63 segmentation files.
2026-04-07 15:29:20,293 | INFO | src.utils.logger | Using single model for evaluation: trained_models/best_annealing_3.pth
2026-04-07 15:29:20,440 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_annealing_3.pth with epochs: 133
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [01:16<00:00,  1.21s/it]
2026-04-07 15:30:36,759 | INFO | src.utils.logger | Per-class IoU (NCR, ED, ET): [0.5290189981460571, 0.6142967343330383, 0.6094342470169067]
2026-04-07 15:30:36,759 | INFO | src.utils.logger | Per-class Dice (NCR, ED, ET): [0.6572405695915222, 0.7378100156784058, 0.7309862971305847]
2026-04-07 15:30:36,759 | INFO | src.utils.logger | Not-NaN batches: 58.0
2026-04-07 15:30:36,760 | INFO | src.utils.logger | Validation results - Mean IoU: 0.58, Mean Dice: 0.71
```

## Best balanced weights model with 168 epochs

```bash
2026-04-07 15:39:05,603 | INFO | src.utils.logger | Starting evaluation...
2026-04-07 15:39:05,604 | DEBUG | src.utils.logger | Found 63 volume files and 63 segmentation files.
2026-04-07 15:39:05,604 | INFO | src.utils.logger | Using single model for evaluation: trained_models/best_new_model_2.pth
2026-04-07 15:39:05,721 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_new_model_2.pth with epochs: 168
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [01:15<00:00,  1.20s/it]
2026-04-07 15:40:21,384 | INFO | src.utils.logger | Per-class IoU (NCR, ED, ET): [0.5487962961196899, 0.6419033408164978, 0.6469851732254028]
2026-04-07 15:40:21,384 | INFO | src.utils.logger | Per-class Dice (NCR, ED, ET): [0.6716948747634888, 0.762754499912262, 0.7613620162010193]
2026-04-07 15:40:21,384 | INFO | src.utils.logger | Not-NaN batches: 58.0
2026-04-07 15:40:21,384 | INFO | src.utils.logger | Validation results - Mean IoU: 0.61, Mean Dice: 0.73
```

## Best weighted model with 99 epochs

```bash
2026-04-07 15:42:09,797 | INFO | src.utils.logger | Starting evaluation...
2026-04-07 15:42:09,797 | DEBUG | src.utils.logger | Found 63 volume files and 63 segmentation files.
2026-04-07 15:42:09,798 | INFO | src.utils.logger | Using single model for evaluation: trained_models/best_model_weighted.pth
2026-04-07 15:42:09,969 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_model_weighted.pth with epochs: 99
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [01:14<00:00,  1.18s/it]
2026-04-07 15:43:24,709 | INFO | src.utils.logger | Per-class IoU (NCR, ED, ET): [0.5306794047355652, 0.6307308077812195, 0.6425608396530151]
2026-04-07 15:43:24,709 | INFO | src.utils.logger | Per-class Dice (NCR, ED, ET): [0.6539758443832397, 0.7537451982498169, 0.7562583684921265]
2026-04-07 15:43:24,709 | INFO | src.utils.logger | Not-NaN batches: 58.0
2026-04-07 15:43:24,710 | INFO | src.utils.logger | Validation results - Mean IoU: 0.60, Mean Dice: 0.72
```

## Ensemble of top 3 models

```bash
2026-04-07 15:19:05,685 | INFO | src.utils.logger | Starting evaluation...
2026-04-07 15:19:05,686 | DEBUG | src.utils.logger | Found 63 volume files and 63 segmentation files.
2026-04-07 15:19:05,686 | INFO | src.utils.logger | Using ensemble of models for evaluation: ['trained_models/best_annealing_3.pth', 'trained_models/best_new_model_2.pth', 'trained_models/best_model_weighted.pth']
2026-04-07 15:19:05,847 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_annealing_3.pth with epochs: 133
2026-04-07 15:19:05,997 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_new_model_2.pth with epochs: 168
2026-04-07 15:19:06,145 | DEBUG | src.utils.logger | Loaded checkpoint from trained_models/best_model_weighted.pth with epochs: 99
2026-04-07 15:19:06,207 | INFO | src.utils.logger | Loaded 3 models for ensemble
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 63/63 [03:40<00:00,  3.49s/it]
2026-04-07 15:22:46,376 | INFO | src.utils.logger | Per-class IoU (NCR, ED, ET): [0.546093225479126, 0.637911319732666, 0.6408887505531311]
2026-04-07 15:22:46,376 | INFO | src.utils.logger | Per-class Dice (NCR, ED, ET): [0.6690188646316528, 0.7590558528900146, 0.7551633715629578]
2026-04-07 15:22:46,376 | INFO | src.utils.logger | Not-NaN batches: 58.0
2026-04-07 15:22:46,377 | INFO | src.utils.logger | Validation results - Mean IoU: 0.61, Mean Dice: 0.73
```
