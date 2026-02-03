import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import mvsfunc as mvf
import vapoursynth as vs
core = vs.core

def padimage_plane(plane, chroma=0):
    #chroma = 0 -> 444 and for all Y planes
    #chroma = 1 -> 420 for Chroma
    height = plane.shape[0]
    width = plane.shape[1]
    if(chroma==0):
        if(height%16==0):
            height_padded = 0
        else:
            height_padded = int(np.ceil(height/16)*16-height)
        if(width%16==0):
            width_padded = 0
        else:
            width_padded = int(np.ceil(width/16)*16-width)
        padded = plane.copy()
        padded = np.pad(padded, ((0, height_padded), (0, width_padded)), mode = "edge")
    if(chroma==1):
        if(height%8==0):
            height_padded = 0
        else:
            height_padded = int(np.ceil(height/8)*8-height)
        if(width%8==0):
            width_padded = 0
        else:
            width_padded = int(np.ceil(width/8)*8-width)
        padded = plane.copy()
        padded = np.pad(padded, ((0, height_padded), (0, width_padded)), mode = "edge")
    
    return padded

def MB_list_generate(padded_planes, chroma=0):
    #padded_planes = [y, u, v]
    #chroma = 0 -> 444 and for all Y planes
    #chroma = 1 -> 420 for Chroma
    MB_y_num = int(padded_planes[0].shape[0]/16)
    MB_x_num = int(padded_planes[0].shape[1]/16)
    MB_index_mat = np.zeros((MB_y_num, MB_x_num), dtype="int64")
    MB_dict = {}
    if(chroma==0):
        for y_index in range(MB_y_num):
            for x_index in range(MB_x_num):
                total_index = int(y_index*1000+x_index)
                MB_index_mat[y_index, x_index] = total_index
                MB_dict[total_index] = [padded_planes[0][y_index*16:(y_index+1)*16, x_index*16:(x_index+1)*16].copy().astype("int64"),\
                                        padded_planes[1][y_index*16:(y_index+1)*16, x_index*16:(x_index+1)*16].copy().astype("int64"),\
                                        padded_planes[2][y_index*16:(y_index+1)*16, x_index*16:(x_index+1)*16].copy().astype("int64")]
    else:
        for y_index in range(MB_y_num):
            for x_index in range(MB_x_num):
                total_index = int(y_index*1000+x_index)
                MB_index_mat[y_index, x_index] = total_index
                MB_dict[total_index] = [padded_planes[0][y_index*16:(y_index+1)*16, x_index*16:(x_index+1)*16].copy().astype("int64"),\
                                        padded_planes[1][y_index*8:(y_index+1)*8, x_index*8:(x_index+1)*8].copy().astype("int64"),\
                                        padded_planes[2][y_index*8:(y_index+1)*8, x_index*8:(x_index+1)*8].copy().astype("int64")]
    return MB_index_mat, MB_dict

def ac_energy(MB, planes: list = [0, 1, 2]):
    ac_energy = 0
    for plane in planes:
        ac_energy_temp = np.square(MB[plane][:,:]).sum() - (MB[plane][:,:].sum())**2/MB[plane].size
        ac_energy = ac_energy + ac_energy_temp
    return ac_energy

def calculate_AC_energy_whole_img(MB_index_mat, MB_dict, planes: list = [0, 1, 2]):
    #AC Energy for all the MBs
    MB_AC_dict = {}
    for y_index in range(int(MB_index_mat.shape[0])):
        for x_index in range(int(MB_index_mat.shape[1])):
            MB_AC_dict[MB_index_mat[y_index, x_index]] = ac_energy(MB_dict[MB_index_mat[y_index, x_index]], planes = planes)
    return MB_AC_dict

def plot_MB_stats(MB_index_mat, MB_stat_dict):
    #Plotting MB statistics map, e.g. AC energy, ΔQP, etc
    stats_img = np.zeros((int(MB_index_mat.shape[0]*16), int(MB_index_mat.shape[1]*16)), dtype="float64")
    for y_index in range(int(MB_index_mat.shape[0])):
        for x_index in range(int(MB_index_mat.shape[1])):
            stats_img[y_index*16:(y_index+1)*16, x_index*16:(x_index+1)*16] = MB_stat_dict[MB_index_mat[y_index, x_index]]
    return stats_img

def MB_map_generate(MB_index_mat, MB_stat_dict):
    #Same as plot_MB_stats, but not resized to full size of the frame
    MB_map = np.zeros((int(MB_index_mat.shape[0]), int(MB_index_mat.shape[1])), dtype="float64")
    for y_index in range(int(MB_index_mat.shape[0])):
        for x_index in range(int(MB_index_mat.shape[1])):
            MB_map[y_index, x_index] = MB_stat_dict[MB_index_mat[y_index, x_index]]
    return MB_map

def get_ac_list(MB_index_mat, MB_AC_dict):
    #Not used, only for outputing AC energy statistics
    AC_list = []
    for y_index in range(int(MB_index_mat.shape[0])):
        for x_index in range(int(MB_index_mat.shape[1])):
            AC_list.append(MB_AC_dict[MB_index_mat[y_index, x_index]])
    return np.array(AC_list)

def get_mapped_map(input_map, vmin, vmax):
    cmap = matplotlib.colormaps.get_cmap('bwr')
    norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)
    true_colored_map = cmap(norm_obj(input_map))
    rgb_image = (true_colored_map[:,:,0:3]*255).astype("uint8")
    fullsize_img = np.zeros((int(rgb_image.shape[0]*16), int(rgb_image.shape[1]*16), 3), dtype="uint8")
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            fullsize_img[16*i:16*(i+1), 16*j:16*(j+1),:] = rgb_image[i,j,:]
    return fullsize_img

def qp_maps_stats(MB_index_mat, MB_AC_dict, BIT_DEPTH: int, mode: int = 1, aq_strength: float = 1.0):
    #Begins AQ algorithm
    bit_depth_correction = 1 / (1 << (2*(BIT_DEPTH-8)))
    bias_strength = aq_strength #aq-strength
    
    
    MB_ac_map = MB_map_generate(MB_index_mat, MB_AC_dict) #small MB map for calculation
    qp_adj_map = np.power(MB_ac_map * bit_depth_correction + 1, 0.125)
    avg_adj_raw = qp_adj_map.mean()
    avg_adj_pow2 = np.square(qp_adj_map).mean()
    avg_adj = avg_adj_raw - 0.5 * (avg_adj_pow2 - 14.) / avg_adj_raw
    #print(f"avg_adj_raw: {avg_adj_raw:.2f}, avg_adj:{avg_adj:.2f}\n")
    #AQ-mode = 1
    strength_1 = bias_strength * 1.0397
    energy_helper_mat = np.ones(qp_adj_map.shape, dtype=qp_adj_map.dtype)
    qp_adj_map_mode1 = strength_1 * (np.log2( np.fmax(MB_ac_map, energy_helper_mat) ) - (14.427 + 2*(BIT_DEPTH-8)))
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
    y = padimage_plane(np.asarray(source[0])[:],0)
    if("YUV444" in source.format.name):
        u = padimage_plane(np.asarray(source[1])[:],0)
        v = padimage_plane(np.asarray(source[2])[:],0)
        MB_index_mat, MB_dict = MB_list_generate([y, u, v], 0)
        MB_AC_dict = calculate_AC_energy_whole_img(MB_index_mat, MB_dict, planes = [0, 1, 2])
    elif("YUV420" in source.format.name):
        u = padimage_plane(np.asarray(source[1])[:],1)
        v = padimage_plane(np.asarray(source[2])[:],1)
        MB_index_mat, MB_dict = MB_list_generate([y, u, v], 1)
        MB_AC_dict = calculate_AC_energy_whole_img(MB_index_mat, MB_dict, planes = [0, 1, 2])
    (avg_adj_raw, avg_adj), (qp_adj_map_mode1, fin_qp_offset_map_mode1), (qp_adj_map_mode2, fin_qp_offset_map_mode2), (qp_adj_map_mode3, fin_qp_offset_map_mode3),\
    (qp_map_diff_2to1, qp_map_diff_3to1, qp_map_diff_3to2) = qp_maps_stats(MB_index_mat, MB_AC_dict, aq_strength = 1.0, BIT_DEPTH = BIT_DEPTH)
    if(aq_mode == 0):
        ac_map = MB_map_generate(MB_index_mat, MB_AC_dict)
        rgb_image = get_mapped_map(ac_map, 0, ac_map.max()*0.5)
    else:
        if(diff==2):
            v = np.max([(np.abs(qp_map_diff_2to1.min())+qp_map_diff_2to1.max())/2,qp_map_diff_2to1.max()])
            if(vmin!=0):
                rgb_image = get_mapped_map(qp_map_diff_2to1, vmin, vmax)
            else:
                rgb_image = get_mapped_map(qp_map_diff_2to1, -v, v)
        elif(diff==3):
            v = np.max([(np.abs(qp_map_diff_3to1.min())+qp_map_diff_3to1.max())/2,qp_map_diff_3to1.max()])
            if(vmin!=0):
                rgb_image = get_mapped_map(qp_map_diff_3to1, vmin, vmax)
            else:
                rgb_image = get_mapped_map(qp_map_diff_3to1, -v, v)
        elif(diff==6):
            v = np.max([(np.abs(qp_map_diff_3to2.min())+qp_map_diff_3to2.max())/2,qp_map_diff_3to2.max()])
            if(vmin!=0):
                rgb_image = get_mapped_map(qp_map_diff_3to2, vmin, vmax)
            else:
                rgb_image = get_mapped_map(qp_map_diff_3to2, -v, v)
        else:
            if(aq_mode == 1):
                if(show_fin_delta_qp_map == 1):
                    v = np.min([np.abs(fin_qp_offset_map_mode1.min()),fin_qp_offset_map_mode1.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode1, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode1, -v, v)
                else:
                    v = np.min([np.abs(qp_adj_map_mode1.min()),qp_adj_map_mode1.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(qp_adj_map_mode1, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(qp_adj_map_mode1, -v, v)
            elif(aq_mode == 2):
                if(show_fin_delta_qp_map == 1):
                    v = np.min([np.abs(fin_qp_offset_map_mode2.min()),fin_qp_offset_map_mode2.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode2, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode2, -v, v)
                else:
                    v = np.min([np.abs(qp_adj_map_mode2.min()),qp_adj_map_mode2.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(qp_adj_map_mode2, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(qp_adj_map_mode2, -v, v)
            elif(aq_mode == 3):
                if(show_fin_delta_qp_map == 1):
                    v = np.min([np.abs(fin_qp_offset_map_mode3.min()),fin_qp_offset_map_mode3.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode3, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(fin_qp_offset_map_mode3, -v, v)
                else:
                    v = np.min([np.abs(qp_adj_map_mode3.min()),qp_adj_map_mode3.max()])
                    if(vmin!=0):
                        rgb_image = get_mapped_map(qp_adj_map_mode3, vmin, vmax)
                    else:
                        rgb_image = get_mapped_map(qp_adj_map_mode3, -v, v)
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

def wrapper(core, clip, aq_mode: int = 1, diff: int = 0, show_fin_delta_qp_map: int = 0, display_stats: int = 0, vmin: float = 0.0, vmax: float = 0.0, matrix = None):
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
    AQ final Δqp =  Δqp0 + Δqp
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
    clip2 = mvf.ToRGB(clip)
    clip2 = core.std.SetFrameProp(clip2, prop="_aqmode", intval = aq_mode)
    clip2 = core.std.SetFrameProp(clip2, prop="_diff", intval = diff)
    clip2 = core.std.SetFrameProp(clip2, prop="_finqp", intval = show_fin_delta_qp_map)
    clip2 = core.std.SetFrameProp(clip2, prop="_display_stats", intval = display_stats)
    clip2 = core.std.SetFrameProp(clip2, prop="_stats", data = "") #Used to deliver qp stats if needed
    clip2 = core.std.SetFrameProp(clip2, prop="_vmin", floatval = vmin)
    clip2 = core.std.SetFrameProp(clip2, prop="_vmax", floatval = vmax)
    rgb_clip = core.std.ModifyFrame(padded_clip, [clip, clip2], fill_frame)
    clip_out = mvf.ToYUV(rgb_clip, full=True, css=css, matrix=matrix)
    return clip_out