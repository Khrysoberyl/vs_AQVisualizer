# vs_AQVisualizer
An adaptive quantization result visualization for Vapoursynth.<br>
First available draft, speed optimization is ongoing. Documentation is also preparing. <br>
Currently only YUV420P8 and YUV444P8 formats are supported. 10-bit support will be introduced in the future.

# Dependencies
- numpy
- matplotlib
- mvsfunc
- vapoursynth

# Very Basic Usage
```Python
import AQAnalyzer

AQAnalyzer.wrapper(clip: vs.VideoNode, aq_mode: int = 1, diff: int = 0, show_fin_delta_qp_map: int = 0, display_stats: int = 0, vmin: float = 0.0, vmax: float = 0.0, matrix = None)
```
### aq_mode
- 0 -> Show AC energy map
- 1 -> Show aq-mode = 1 map
- 2 -> Show aq-mode = 2 map
- 3 -> Show aq-mode = 3 map
### diff:
- 2 -> Show mode 2 - mode 1 Δqp_adj difference map (2*1)
- 3 -> Show mode 3 - mode 1  Δqp_adj difference map (3*1)
- 6 -> Show mode 3 - mode 2  Δqp_adj difference map (3*2)
- 0 -> Do not calculate difference between modes
### show_fin_delta_qp_map
- 0 -> Display only Δqp_adj
- 1 -> Display Δqp0 + Δqp_adj
- AQ final Δqp =  Δqp0 + Δqp_adj
  - Δqp0 is based on AC energy only, not related to aq-strength.
  - Δqp_adj is adjustment on top of Δqp0, and proportional to aq-strength.
### display_stats
- Show stats of qp maps to help determine min and max values in colormaps. The stats will be displayed in Vapoursynth Logs as INFO level entries (core.log_message).
- Based on how VSEdit works, normally 2 duplicate log entries can be seen for a single frame.
- 1 -> yes
- 0 -> no
### vmin
- minimum value for the colormap (blue), anything below will be treated by the same color.
- 0.0 -> automatic
- By default, vmin and vmax are symmetric with respect to 0
### vmax
- maximum value for the colormap (red), anything above will be treated by the same color.
- 0.0 -> automatic
- By default, vmin and vmax are symmetric with respect to 0
### matrix
- matrix used to convert RGB to YUV, not too important. Refer to mvsfunc.ToYUV().
### Display Priority
- `aq_mode = 0` gets highest priority as AC energy is calculated before all the Δqp maps.
- Non-zero `diff` overrides non-zero `aq_mode`.
- `diff = 0` displays corresponding aq_mode maps. Use `show_fin_delta_qp_map` to control displaying Δqp0 or Δqp_adj.

# Example1
## Source frame:
![Source](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/src.png)
## AQ-Mode = 1 Δqp_adj map
![aqmode1](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/aqmode1.png)
## Expected Stats if Enabled
![stats](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/stats.png)

# Example2
## Source frame:
![Source](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/src2.png)
## AQ-Mode = 1 Δqp_adj map
![aqmode1](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/aqmode1_2.png)
## Expected Stats if Enabled
![stats](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/stats2.png)

# Example3
## Source frame:
![Source](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/src3.png)
## AQ-Mode = 1 Δqp_adj map
![aqmode1](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/aqmode2_3.png)
## Expected Stats if Enabled
![stats](https://github.com/Khrysoberyl/vs_AQVisualizer/blob/main/docs/stats3.png)
