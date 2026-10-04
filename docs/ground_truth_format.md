# Ground-truth file (our own simple format - NOT an official one)

```yaml
ceiling_height_m: 2.62
floor_area_m2: 14.3
wall_lengths_m: [4.51, 3.20, 4.50, 3.19]
opening_widths_m: [0.91, 0.80]
```
Only keys present are evaluated. Predicted and true values are matched one-to-one by optimal assignment (Hungarian) on absolute error;
unmatched ground-truth openings/walls count as misses. Run: `python -m applied_ai.cli evaluate --scene outputs/<scan>/scene.json --ground-truth gt.yaml [--repeat-scene other/scene.json] [--calibrate]`
