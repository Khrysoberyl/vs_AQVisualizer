import numpy as np
import mvsfunc as mvf
import vapoursynth as vs
core = vs.core

def build_bwr_LUT(n=256):
    steps = np.linspace(0.0,1.0,n, dtype=np.float32)
    lut = np.empty((n,3), dtype=np.float32)
    mask = steps < 0.5

    lut[:,0] = np.where(mask, 2*steps, 1.0)          # R
    lut[:,1] = np.where(mask, 2*steps, 2*(1-steps))  # G
    lut[:,2] = np.where(mask, 1.0, 2*(1-steps))      # B

    return (lut * (n-1)).astype(np.uint8)

BWR_LUT = build_bwr_LUT(256)

def padimage_plane(plane, blocksize = 16):
    #
    height, width = plane.shape
    height_padded = (-height)%blocksize
    width_padded = (-width)%blocksize
    if((height_padded==0) and (width_padded==0)):
        return plane
    else:
        return np.pad(plane, ((0, height_padded), (0, width_padded)), mode = "edge")

def plane_MB_generate(padded_plane, blocksize = 16):
    #Split image into a (y, x, blocksize, blocksize) shaped MB tensor
    height, width = padded_plane.shape
    return padded_plane.reshape(height//blocksize, blocksize, width//blocksize, blocksize).transpose(0,2,1,3)

def ac_energy_img(MB_list: list, planes: list = [0, 1, 2]):
    #Calculate AC energy given the MB tensor, planes used for calculation can be tuned
    ac_energy = 0
    for plane in planes:
        mb = MB_list[plane].astype(np.uint32, copy=False) #prevent overflow
        normal_sum = mb.sum(axis = (-1,-2))
        square_sum = np.square(mb).sum(axis = (-1,-2))
        MBsize = mb.shape[-1]*mb.shape[-2]
        ac_energy_temp = square_sum - normal_sum**2/MBsize
        ac_energy = ac_energy + ac_energy_temp #dtype of ac_energy is float64
    return ac_energy

def plot_MB_stats(MB_stat_img, blocksize: int = 16):
    #Plotting MB statistics map, e.g. AC energy, ΔQP, etc. into the original sized map from a downsized (small) MB image
    return np.repeat(np.repeat(MB_stat_img, blocksize, axis=0), blocksize, axis=1)

def get_ac_list(MB_AC_image):
    #Not used, only for outputing AC energy statistics
    return np.flatten(MB_AC_image)

def fast_bwr_map(input_map, vmin, vmax, lut=BWR_LUT):
    norm = (input_map-vmin)/(vmax-vmin)
    norm = np.clip(norm, 0.0, 1.0)
    height, width = norm.shape
    
    index = (norm*255).astype("int32")
    bwr_map = lut[index]
    #bwr_map = np.empty((height, width, 3), dtype=np.float64)
    #mask = norm < 0.5
    #
    #bwr_map[:, :,0] = np.where(mask, 2*norm, 1.0)          # R
    #bwr_map[:, :,1] = np.where(mask, 2*norm, 2*(1-norm))   # G
    #bwr_map[:, :,2] = np.where(mask, 1.0, 2*(1-norm))      # B
    
    #rgb_image = (bwr_map*255).astype("uint8")
    return plot_MB_stats(bwr_map, 16)

def plot_helper(input, mode: int = 1, vmin: float = 0.0, vmax: float = 0.0):
    #mode = 1 used for difference maps as they normally have very small ranges, especially on the positive side
    #mode = 2 used for normal Δqp maps, whether Δqp_adj or final Δqp.
    #         They normally have decent positive values and very negative values, so some negative values are clamped out of the range for clarity
    if(mode==1):
        v = np.max([(np.abs(input.min())+np.abs(input.max()))/2,np.abs(input).max()])
    elif(mode==2):
        v = np.min([np.abs(input.min()),np.abs(input.max())]) #np.abs is needed because sometimes max value is also negative, eg. for pure black images
    if(vmin!=0):
        return fast_bwr_map(input, vmin, vmax)
    else:
        return fast_bwr_map(input, -v, v)

def qp_maps_stats(MB_ac_map, BIT_DEPTH: int, mode: int = 1, aq_strength: float = 1.0):
    #Begins AQ algorithm
    bit_depth_correction = 1 / (1 << (2*(BIT_DEPTH-8)))
    bias_strength = aq_strength #aq-strength
    
    qp_adj_map = np.power(MB_ac_map * bit_depth_correction + 1, 0.125)
    avg_adj_raw = qp_adj_map.mean()
    avg_adj_pow2 = np.square(qp_adj_map).mean()
    avg_adj = avg_adj_raw - 0.5 * (avg_adj_pow2 - 14.) / avg_adj_raw
    #print(f"avg_adj_raw: {avg_adj_raw:.2f}, avg_adj:{avg_adj:.2f}\n")
    #AQ-mode = 1
    strength_1 = bias_strength * 1.0397
    #energy_helper_mat = np.ones(qp_adj_map.shape, dtype=qp_adj_map.dtype)
    qp_adj_map_mode1 = strength_1 * (np.log2( np.fmax(MB_ac_map, 1) ) - (14.427 + 2*(BIT_DEPTH-8)))
    fin_qp_offset_map_mode1 = qp_adj_map+qp_adj_map_mode1
    
    #AQ-mode = 2
    strength_2 = bias_strength * avg_adj
    qp_adj_map_mode2 = strength_2 * (qp_adj_map - avg_adj)
    fin_qp_offset_map_mode2 = qp_adj_map+qp_adj_map_mode2
    
    #AQ-mode = 3
    strength_3 = bias_strength * avg_adj # same as aq-mode = 2
    qp_adj_map_mode3 = strength_3 * (qp_adj_map - avg_adj) + bias_strength * (1 - 14 / np.square(qp_adj_map))
    fin_qp_offset_map_mode3 = qp_adj_map+qp_adj_map_mode3
    
    #print(f"QP offset map for Mode 1 min:{qp_adj_map_mode1.min():.2f}, max:{qp_adj_map_mode1.max():.2f}")
    #print(f"QP offset map for Mode 2 min:{qp_adj_map_mode2.min():.2f}, max:{qp_adj_map_mode2.max():.2f}")
    #print(f"QP offset map for Mode 3 min:{qp_adj_map_mode3.min():.2f}, max:{qp_adj_map_mode3.max():.2f}")
    
    #diff between 3 modes
    qp_map_diff_2to1 = fin_qp_offset_map_mode2-fin_qp_offset_map_mode1
    qp_map_diff_3to1 = fin_qp_offset_map_mode3-fin_qp_offset_map_mode1
    qp_map_diff_3to2 = fin_qp_offset_map_mode3-fin_qp_offset_map_mode2
    #print(f"QP offset difference map for Mode 2 - Mode 1 min:{qp_map_diff_2to1.min():.2f}, max:{qp_map_diff_2to1.max():.2f}")
    #print(f"QP offset difference map for Mode 3 - Mode 1 min:{qp_map_diff_3to1.min():.2f}, max:{qp_map_diff_3to1.max():.2f}")
    #print(f"QP offset difference map for Mode 3 - Mode 2 min:{qp_map_diff_3to2.min():.2f}, max:{qp_map_diff_3to2.max():.2f}")
    return [(avg_adj_raw, avg_adj), (qp_adj_map_mode1, fin_qp_offset_map_mode1), (qp_adj_map_mode2, fin_qp_offset_map_mode2), (qp_adj_map_mode3, fin_qp_offset_map_mode3),\
            (qp_map_diff_2to1, qp_map_diff_3to1, qp_map_diff_3to2)]

def fill_frame(n, f):
    source = f[0].copy()
    fout = f[1].copy()
    BIT_DEPTH = source.format.bits_per_sample
    aq_mode = f[1].props["_aqmode"]
    show_fin_delta_qp_map = f[1].props["_finqp"]
    diff = f[1].props["_diff"]
    display_stats = f[1].props["_display_stats"]
    vmin = f[1].props["_vmin"]
    vmax = f[1].props["_vmax"]
    y_MB = plane_MB_generate(padimage_plane(np.asarray(source[0])[:],16),16)
    if("YUV444" in source.format.name):
        u_MB = plane_MB_generate(padimage_plane(np.asarray(source[1])[:], blocksize = 16), 16)
        v_MB = plane_MB_generate(padimage_plane(np.asarray(source[2])[:], blocksize = 16), 16)
    elif("YUV420" in source.format.name):
        u_MB = plane_MB_generate(padimage_plane(np.asarray(source[1])[:], blocksize = 8), 8)
        v_MB = plane_MB_generate(padimage_plane(np.asarray(source[2])[:], blocksize = 8), 8)
    MB_ac_map = ac_energy_img([y_MB, u_MB, v_MB], planes = [0,1,2])
    (avg_adj_raw, avg_adj), (qp_adj_map_mode1, fin_qp_offset_map_mode1), (qp_adj_map_mode2, fin_qp_offset_map_mode2), (qp_adj_map_mode3, fin_qp_offset_map_mode3),\
    (qp_map_diff_2to1, qp_map_diff_3to1, qp_map_diff_3to2) = qp_maps_stats(MB_ac_map, aq_strength = 1.0, BIT_DEPTH = BIT_DEPTH)
    if(aq_mode == 0):
        rgb_image = get_mapped_map(MB_ac_map, 0, MB_ac_map.max()*0.5)
    elif(diff==2):
        rgb_image = plot_helper(qp_map_diff_2to1, mode = 1, vmin = vmin, vmax = vmax)
    elif(diff==3):
        rgb_image = plot_helper(qp_map_diff_3to1, mode = 1, vmin = vmin, vmax = vmax)
    elif(diff==6):
        rgb_image = plot_helper(qp_map_diff_3to2, mode = 1, vmin = vmin, vmax = vmax)
    elif(aq_mode == 1):
        if(show_fin_delta_qp_map == 1):
            rgb_image = plot_helper(fin_qp_offset_map_mode1, mode = 2, vmin = vmin, vmax = vmax)
        else:
            rgb_image = plot_helper(qp_adj_map_mode1, mode = 2, vmin = vmin, vmax = vmax)
    elif(aq_mode == 2):
        if(show_fin_delta_qp_map == 1):
            rgb_image = plot_helper(fin_qp_offset_map_mode2, mode = 2, vmin = vmin, vmax = vmax)
        else:
            rgb_image = plot_helper(qp_adj_map_mode2, mode = 2, vmin = vmin, vmax = vmax)
    elif(aq_mode == 3):
        if(show_fin_delta_qp_map == 1):
            rgb_image = plot_helper(fin_qp_offset_map_mode3, mode = 2, vmin = vmin, vmax = vmax)
        else:
            rgb_image = plot_helper(qp_adj_map_mode3, mode = 2, vmin = vmin, vmax = vmax)
    for p in range(fout.format.num_planes):
        np.asarray(fout[p])[:] = rgb_image[:fout.height,:fout.width,p]
    qamm1_min = qp_adj_map_mode1.min()
    qamm1_max = qp_adj_map_mode1.max()
    qamm2_min = qp_adj_map_mode2.min()
    qamm2_max = qp_adj_map_mode2.max()
    qamm3_min = qp_adj_map_mode3.min()
    qamm3_max = qp_adj_map_mode3.max()
    fqommm1_min = fin_qp_offset_map_mode1.min()
    fqommm1_max = fin_qp_offset_map_mode1.max()
    fqommm2_min = fin_qp_offset_map_mode2.min()
    fqommm2_max = fin_qp_offset_map_mode2.max()
    fqommm3_min = fin_qp_offset_map_mode3.min()
    fqommm3_max = fin_qp_offset_map_mode3.max()
    qmdf2to1_min = qp_map_diff_2to1.min()
    qmdf2to1_max = qp_map_diff_2to1.max()
    qmdf3to1_min = qp_map_diff_3to1.min()
    qmdf3to1_max = qp_map_diff_3to1.max()
    qmdf3to2_min = qp_map_diff_3to2.min()
    qmdf3to2_max = qp_map_diff_3to2.max()
    stats = f"""avg_adj_raw: {avg_adj_raw:.2f}, avg_adj after centering:{avg_adj:.2f}

Δqp map for mode 1, 2 and 3: [{qamm1_min:.2f}, {qamm1_max:.2f}], [{qamm2_min:.2f}, {qamm2_max:.2f}], [{qamm3_min:.2f}, {qamm3_max:.2f}]

fin Δqp map (Δqp0 + Δqp) for mode 1, 2 and 3: [{fqommm1_min:.2f}, {fqommm1_max:.2f}], [{fqommm2_min:.2f}, {fqommm2_max:.2f}], [{fqommm3_min:.2f}, {fqommm3_max:.2f}]

Δqp difference map of mode2 - mode 1:[{qmdf2to1_min:.2f}, {qmdf2to1_max:.2f}]
Δqp difference map of mode3 - mode 1:[{qmdf3to1_min:.2f}, {qmdf3to1_max:.2f}]
Δqp difference map of mode3 - mode 2:[{qmdf3to2_min:.2f}, {qmdf3to2_max:.2f}]
"""
    if(display_stats==1):
        core.log_message(vs.MESSAGE_TYPE_INFORMATION, stats)
    return fout

def wrapper(clip: vs.VideoNode, aq_mode: int = 1, diff: int = 0, show_fin_delta_qp_map: int = 0, display_stats: int = 0, vmin: float = 0.0, vmax: float = 0.0, matrix = None):
    """
    aq_mode: show AC energy map -> 0
          show aq-mode = 1 map -> 1
          show aq-mode = 2 map -> 2
          show aq-mode = 3 map -> 3
    diff: mode 2 - mode 1 -> 2*1 = 2
          mode 3 - mode 1 -> 3*1 = 3
          mode 3 - mode 2 -> 3*2 = 6
          Do not calculate difference between modes -> 0
    show_fin_delta_qp_map: display only Δqp_adj -> 0
                           display Δqp0 + Δqp_adj -> 1
    AQ final Δqp =  Δqp0 + Δqp_adj
            Δqp0 is based on AC energy only, not related to aq-strength
            Δqp_adj is adjustment on top of Δqp0, and proportional to aq-strength.
    display_stats: Show stats of qp maps to help determine min and max values in colormaps
                   yes -> 1
                   no -> 0
    vmin: minimum value for the colormap, anything below will be treated by the same color.
          automatic -> 0.0
    vmax: maximum value for the colormap, anything above will be treated by the same color.
          automatic -> 0.0
          By default, vmin and vmax are symmetric with respect to 0
    matrix: matrix used to convert RGB to YUV, not too important.
    """
    if("YUV444" in clip.format.name):
        css = "11"
    elif("YUV420" in clip.format.name):
        css = "22"
    else:
        raise TypeError('\"clip\" must be YUV420 or YUV444!')
    #clip2 = mvf.ToRGB(clip)
    clip2 = core.std.BlankClip(clip, format=vs.RGB24)
    clip2 = core.std.SetFrameProp(clip2, prop="_aqmode", intval = aq_mode)
    clip2 = core.std.SetFrameProp(clip2, prop="_diff", intval = diff)
    clip2 = core.std.SetFrameProp(clip2, prop="_finqp", intval = show_fin_delta_qp_map)
    clip2 = core.std.SetFrameProp(clip2, prop="_display_stats", intval = display_stats)
    clip2 = core.std.SetFrameProp(clip2, prop="_stats", data = "") #Used to deliver qp stats if needed
    clip2 = core.std.SetFrameProp(clip2, prop="_vmin", floatval = vmin)
    clip2 = core.std.SetFrameProp(clip2, prop="_vmax", floatval = vmax)
    rgb_clip = core.std.ModifyFrame(clip2, [clip, clip2], fill_frame)
    clip_out = mvf.ToYUV(rgb_clip, full=True, css=css, matrix=matrix, depth=clip.format.bits_per_sample)
    return clip_out