# vs_AQVisualizer
An adaptive quantization result visualization for Vapoursynth
First available draft, speed optimization is ongoing. Documentation is also preparing. 

# Very Basic Usage
```Python
import AQAnalyzer

AQAnalyzer.wrapper(clip: vs.VideoNode, aq_mode: int = 1, diff: int = 0, show_fin_delta_qp_map: int = 0, display_stats: int = 0, vmin: float = 0.0, vmax: float = 0.0, matrix = None)
```
### aq_mode
- 0 -> show AC energy map
- 1 -> show aq-mode = 1 map
- 2 -> show aq-mode = 2 map
- 3 -> show aq-mode = 3 map
### diff:
- 2 -> mode 2 - mode 1 (2*1)
- 3 -> mode 3 - mode 1 (3*1)
- 6 -> mode 3 - mode 2 (3*2)
- 0 -> Do not calculate difference between modes
### show_fin_delta_qp_map
- 0 -> display only Δqp_adj
- 1 ->display Δqp0 + Δqp_adj
- AQ final Δqp =  Δqp0 + Δqp
  - Δqp0 is based on AC energy only, not related to aq-strength.
  - Δqp_adj is adjustment on top of Δqp0, and proportional to aq-strength.
### display_stats
- Show stats of qp maps to help determine min and max values in colormaps.
- 1 -> yes
- 0 -> no
### vmin
- minimum value for the colormap, anything below will be treated by the same color.
- 0.0 -> automatic
### vmax
- maximum value for the colormap, anything above will be treated by the same color.
- 0.0 -> automatic
- By default, vmin and vmax are symmetric with respect to 0
### matrix
- matrix used to convert RGB to YUV, not too important. Refer to mvsfunc.ToYUV().
