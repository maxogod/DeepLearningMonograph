# Evaluation results

## Best annealing model with 133 epochs

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
