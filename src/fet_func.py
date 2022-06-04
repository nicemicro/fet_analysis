# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:14:09 2021

@author: Nice Micro
(C) Nice Micro 2021-2022
Licenced under the GPL v3.0
"""

import pandas as pd
import numpy as np
import glob
import re
import pylab as pl
from scipy import signal

def file_convert(filename, header=2):
    data = pd.read_csv(filename, delimiter=",",
                       header=header, encoding="korean")
    header = {"VDS": " Drain_Voltage (V)",
              "VGS": " Gate_Voltage (V)",
              "IDS": " Drain_Current (A)"}
    if (not header["VDS"] in data.columns) or \
        (not header["VGS"] in data.columns) or \
        (not header["IDS"] in data.columns):
        return []
    data = data[[header["VDS"], header["VGS"], header["IDS"]]]
    gt_vltgs = data[header["VGS"]].unique()
    dr_vltgs = data[header["VDS"]].unique()
    if len(gt_vltgs) > len(dr_vltgs):
        # transfer curve
        # Check for the double sweep
        x_val = data[[header["VGS"]]].to_numpy()[:,0]
        sw_reverse = (np.convolve(np.convolve(x_val, [1, -1], mode="valid"),
                                  [1, -1], mode="valid") != 0).argmax()
        # Due to test datasets not being available this is not done yet.
        # TODO implement handling of double sweep data
        if sw_reverse == 0:
            new_data = pd.DataFrame(gt_vltgs)
            new_data.columns = [header["VGS"]]
            for Vds in dr_vltgs:
                new_data = new_data.merge(
                    data[(data[header["VDS"]] == Vds)][[header["VGS"],
                                                      header["IDS"]]],
                    how="left")
                new_data.rename(columns={header["IDS"]: "VDS = "+str(Vds)+" V"},
                                inplace=True)
        else:
            new_data = pd.DataFrame(gt_vltgs[0:sw_reverse+2])
            new_data.columns = [header["VGS"]]
            for Vds in dr_vltgs:
                new_data = new_data.merge(
                    data[(data[header["VDS"]] == Vds)][[header["VGS"],
                                                      header["IDS"]]][0:sw_reverse+2],
                    how="left")
                new_data.rename(columns={header["IDS"]: "VDS = "+str(Vds)+" V"},
                                inplace=True)
    else:
        # output curve
        new_data = pd.DataFrame(dr_vltgs)
        new_data.columns = [header["VDS"]]
        for Vgs in gt_vltgs:
            new_data = new_data.merge(
                data[(data[header["VGS"]] == Vgs)][[header["VDS"],
                                                  header["IDS"]]],
                how="left")
            new_data.rename(columns={header["IDS"]: "VGS = "+str(Vgs)+" V"},
                            inplace=True)
    return [new_data.rename(columns={header["VDS"]: "VDS",
                                    header["VGS"]: "VGS"})]

def read_excel(filename):
    raw_data = pd.read_excel(filename)
    blocks = {}
    data = pd.DataFrame([])
    pattern = re.compile(r"(?P<name>GateV|DrainV|DrainI)[(]?(?P<tag>[0-9]*)[)]?")
    renamer = {"GateV": "VGS", "DrainV": "VDS", "DrainI": "IDS"}
    for colname in raw_data.columns:
        matching = pattern.search(colname)
        if matching is None:
            continue
        coltype = matching.group("name")
        blockid = matching.group("tag")
        if blockid == "":
            blockid = "0"
        if blockid not in blocks:
            blocks[blockid] = raw_data[colname].rename(renamer[coltype])
        else:
            blocks[blockid] = pd.concat([blocks[blockid], raw_data[colname].rename(renamer[coltype])], axis=1)
    data_type = ""
    for block_id, block in blocks.items():
        new_data = pd.DataFrame([])
        if "VGS" not in block.columns:
            new_data = block.rename(columns={"IDS": f"IDS ({block_id})"})
            data_type = "out"
        elif "VDS" not in block.columns:
            new_data = block.rename(columns={"IDS": f"IDS ({block_id})"})
            data_type = "tr"
        elif block["VGS"].std() == 0:
            new_data = pd.concat(
                [
                    block[["VDS"]].round(decimals=2),
                    block["IDS"].rename("VGS = " + str(int(block["VGS"].mean())) + " V")
                ]
                , axis=1
            )
            data_type = "out"
        elif block["VDS"].std() == 0:
            new_data = pd.concat(
                [
                    block[["VGS"]].round(decimals=2),
                    block["IDS"].rename("VDS = " + str(int(block["VDS"].mean())) + " V")
                ],
                axis=1
            )
            data_type = "tr"
        if data.empty:
            data = new_data
        elif data_type == "tr":
            data = pd.merge(data, new_data, on="VGS")
        elif data_type == "out":
            data = pd.merge(data, new_data, on="VDS")
    x_val = data[data.columns[0]].to_numpy()
    sw_reverse = (np.convolve(x_val, [1, -1], mode="valid") == 0).argmax()
    processed_data = []
    if sw_reverse == 0:
        processed_data = [data]
    else:
        processed_data = [data[0:sw_reverse+1], data[sw_reverse+1:]]
    return processed_data

def read_simple_file(filename, voltages=None):
    if voltages is None:
        voltages = setvtgs
    raw_data = pd.read_csv(filename, delimiter=',', header=None)
    ren = {}
    x_val = raw_data[[0]].to_numpy()[:,0]
    sw_reverse = (np.convolve(x_val, [1, -1], mode="valid") == 0).argmax()
    if sw_reverse == 0:
        data = [raw_data]
    else:
        data = [raw_data[0:sw_reverse+1], raw_data[sw_reverse+1:]]
    processed_data = []
    for data_part in data:
        if filename[-len(outend):] == outend:
            ren[0] = "VDS"
            for index in range(len(data_part.columns) - 1):
                ren[data_part.columns[index+1]] = "VGS = " + str(voltages[index]) \
                    + " V"
        if filename[-len(trend):] == trend:
            ren[0] = "VGS"
            for index in range(len(data_part.columns) - 1, 0, -1):
                ren[data_part.columns[index]] = "VDS = " + \
                    str(voltages[len(voltages)-len(data_part.columns)+index]) + " V"
        data_part = data_part.rename(columns=ren)
        processed_data.append(data_part)
    return processed_data

#%%
def smooth_data(data_all):
    for data_part in data_all:
        for index in data_part.columns:
            if index == data_part.columns[0]: continue
            y_raw = data_part[index].to_numpy()
            y_raw = np.pad(y_raw, 2, mode="symmetric")
            y_mat = np.array([y_raw[0:-4], y_raw[1:-3], y_raw[2:-2], 
                              y_raw[3:-1], y_raw[4:]])
            y_denoise = (y_mat.sum(axis=0) - y_mat.max(axis=0) - 
                         y_mat.min(axis=0)) / 3
            data_part.loc[:, index] = y_denoise
    return data_all

#%%
def draw_output(book, axes, limit=0, label_font_s=14):
    if limit == 0 or limit == np.nan:
        limit = np.absolute(book.to_numpy()[:,1:]).max() * 1.2
    x = np.array([])
    axes.set_xlabel(r'$V_{\mathrm{DS}}$ / V', fontsize=label_font_s)
    axes.set_ylabel(r'$I_{\mathrm{DS}}$ / A', fontsize=label_font_s)
    for column_name in book.columns:
        if len(x) == 0:
            x = book[[column_name]].to_numpy()
        else:
            axes.plot(x, book[[column_name]].to_numpy())
    axes.set_ylim(-limit*0.1, limit)

def draw_transfer(book, axes, axes2, limit=0, low_lim=0, label_font_s=14):
    if limit == 0:
        limit = np.absolute(book.to_numpy()[:,1:]).max() * 8
    if low_lim == 0:
        low_lim = max(np.absolute(book.to_numpy()[:,1:]).min() / 4, 1E-13)
    x = np.array([])
    axes.set_xlabel(r'$V_{\mathrm{GS}}$ / V', fontsize=label_font_s)
    axes.set_ylabel(r'$|I_{\mathrm{DS}}|$ / A', fontsize=label_font_s)
    axes2.set_ylabel(r'$|I_{\mathrm{DS}}|^{-1/2}$ / A$^{-1/2}$', fontsize=label_font_s)
    for column_name in book.columns:
        if len(x) == 0:
            x = book[[column_name]].to_numpy()            
        else:
            y = np.absolute(book[[column_name]].to_numpy())
            axes.plot(x, y + (y == 0) * y[(y > 0)].min())
            axes2.plot(x, np.sqrt(y), "s", ms=3)
    axes.set_yscale("log")
    axes.set_ylim(low_lim, limit)
    axes2.set_ylim(0, np.sqrt(limit / 5))
   
#%%
def find_on_index(data: pd.DataFrame) -> int:
    avg_kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
    avg_k_os =int((len(avg_kernel) - 1) / 2 ) # overshoot of the averaging kernel
    sl_kernel = [0.125, 0.25, 0, -0.25, -0.125]
    #sl_kernel = [1/18, 1/12, 1/6, 0, -1/6, -1/12, -1/18]
    sl_k_os = int((len(sl_kernel) - 1) / 2) # overshoot of the slope kernel
    y_val = np.absolute(data.to_numpy()[:, -1])
    # Remove potential 0 values from the y
    y_val = y_val + (y_val == 0) * y_val[(y_val > 0)].min()
    # Calculating Turn On Voltage, IOn, IOff and SS
    logy = np.log10(y_val)
    logavg = np.convolve(np.pad(logy, avg_k_os, mode="symmetric"), avg_kernel,
                        mode="valid")
    slope = np.convolve(np.pad(logavg, sl_k_os, mode="symmetric"), sl_kernel,
                        mode="valid")
    slope = slope * (slope > 0)  # we don't care about decreasing current
    slslope = np.convolve(np.pad(slope, sl_k_os, mode="symmetric"), sl_kernel,
                          mode="valid")
    # Trying to check whether there is a valid turn-on voltage / subthreshold
    # regime in our graph
    #only_pos = ((slope[:-sl_k_os] <= 0) * \
    #            np.array(range(len(slope)-sl_k_os))).max()
    #only_pos = min(only_pos, len(slope) - sl_k_os * 2)
    #min_slsl = slslope[only_pos:-3].argmin() + only_pos
    #findstart = max(4, min(int(only_pos * 0.9), only_pos - 4))
    #findstart = sl_k_os * 2
    y_sl_p = signal.find_peaks(slope[sl_k_os:-sl_k_os],
                               prominence=(slope[sl_k_os:-sl_k_os].max() \
                                           -slope[sl_k_os:-sl_k_os].min())/3)
    y_slsl_p = signal.find_peaks(slslope[sl_k_os*2:-sl_k_os*2], \
        prominence=(slslope[sl_k_os*2:-sl_k_os*2].max() \
        -slslope[sl_k_os*2:-sl_k_os*2].min())/10)
    sl_peaks = y_sl_p[0] + sl_k_os
    slsl_peaks = y_slsl_p[0] + sl_k_os*2
    if len(sl_peaks) == 0:
        slsl_peaks = []
    else:
        slsl_peaks = slsl_peaks[(slsl_peaks < sl_peaks[-1])]
    #on_index = slslope[findstart:-3].argmax() + findstart
    if len(slsl_peaks) > 0:
        # Turn On Voltage
        on_index = slsl_peaks[-1]
    else:
        on_index = 0
    return on_index

def find_fitting_boundaries(data: pd.DataFrame) -> tuple[int, int]:
    fit_from: int
    fit_to: int
    
    y_val = np.absolute(data.to_numpy()[:, -1])
    # Remove potential 0 values from the y
    y_val = y_val + (y_val == 0) * y_val[(y_val > 0)].min()
    # Calculating Mobility and Threshold Voltage
    sqrty = np.sqrt(y_val)
    sl_kernel = [0.125, 0.25, 0, -0.25, -0.125]
    #sl_kernel = [1/18, 1/12, 1/6, 0, -1/6, -1/12, -1/18]
    sl_k_os = int((len(sl_kernel) - 1) / 2) # overshoot of the slope kernel
    slope2 = np.convolve(np.pad(sqrty, sl_k_os, mode="symmetric"), sl_kernel,
                        mode="valid")
    # Finding the range in which to proceed with linear fitting
    rep_slope = sorted(slope2.tolist())[int(len(slope2)*0.95)]
    fit_index = (slope2 > rep_slope * 0.9) * np.arange(len(slope2))
    fit_from = fit_index[(fit_index>0)].min()
    fit_to = fit_index[(fit_index>0)].max() + 2
    return fit_from, fit_to

def sort_transfer_curve(data: pd.DataFrame) -> pd.DataFrame:
    return data.sort_values("VGS", ascending=data.to_numpy()[:,-1].mean() > 0)

def param_eval_core(data, debug_on=False, on_index=None, fitting_boundaries=None):
    # if the average y value is positive, we have n-type device, so the
    # data should be ordered in ascending order by VGS.
    data = sort_transfer_curve(data)
    if on_index is None:
        on_index = find_on_index(data)
    
    x_val = data.to_numpy()[:, 0]
    y_val = np.absolute(data.to_numpy()[:, -1])
    # Remove potential 0 values from the y
    y_val = y_val + (y_val == 0) * y_val[(y_val > 0)].min()
    logy = np.log10(y_val)
    avg_kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
    avg_k_os =int((len(avg_kernel) - 1) / 2 ) # overshoot of the averaging kernel
    logavg = np.convolve(np.pad(logy, avg_k_os, mode="symmetric"), avg_kernel,
                        mode="valid")
    avg = 10 ** logavg
    if on_index == 0:
        # Turn on voltage does not seem to be valid, the device turns on at
        # lower voltage than measured, no subthreshold regime.
        v_turnon = None
        ss = None
        # Off current is simply the lowest smoothed current
        i_off = avg.min()
    else:
        # The minimum is after the maximum value, the turn on voltage is valid
        v_turnon = x_val[on_index]
        # This comparision is made so as to decide whether to use raw or smoothed
        # data based on the points next to and after the turn on voltage point
        if (y_val[on_index+2:] < y_val[on_index+1]).sum() + \
            (y_val[on_index+3:] < y_val[on_index+2]).sum() == 0:
            # Checking how many data points are there after the turn_on point that are
            # less than the 10x of the current at the turn_on point.
            subthres_point = (y_val[on_index:] < y_val[on_index] * 10).sum()
            subthres_point = min(subthres_point, len(y_val) - on_index - 1)
            # Subthreshold Swing
            ss = x_val[on_index + subthres_point] - x_val[on_index]
            ss = ss / (logy[on_index + subthres_point] - logy[on_index])
            # Off current is the real measured current at the turn on voltage
            i_off = y_val[on_index]
        else:
            # Checking how many data points are there after the turn_on point that are
            # less than the 10x of the current at the turn_on point.
            subthres_point = (avg[on_index:] < avg[on_index] * 10).sum()
            subthres_point = min(subthres_point, len(y_val) - on_index - 1)
            # Subthreshold Swing
            ss = x_val[on_index + subthres_point] - x_val[on_index]
            ss = ss / (logavg[on_index + subthres_point] - logavg[on_index])
            # Off current is the averaged current at the turn on voltage
            i_off = avg[on_index]
    # On Current
    i_on = y_val[on_index:].max()
    # Linear fitting
    if fitting_boundaries is None:
        fit_from, fit_to = find_fitting_boundaries(data)
    else:
        fit_from, fit_to = fitting_boundaries
    sqrty = np.sqrt(y_val)
    model = np.polyfit(x_val[fit_from:fit_to], sqrty[fit_from:fit_to], 1)
    # Slope for mobility calculations
    mob_slope = model[0]
    # Threshold Voltage
    v_th = -model[1] / model[0]
    debug = pd.DataFrame()
    if debug_on:
        logy = np.log10(y_val)
        logavg = np.convolve(np.pad(logy, avg_k_os, mode="symmetric"), avg_kernel,
                            mode="valid")
        avg = 10 ** logavg
        sl_kernel = [0.125, 0.25, 0, -0.25, -0.125]
        #sl_kernel = [1/18, 1/12, 1/6, 0, -1/6, -1/12, -1/18]
        sl_k_os = int((len(sl_kernel) - 1) / 2) # overshoot of the slope kernel
        slope = np.convolve(np.pad(logavg, sl_k_os, mode="symmetric"), sl_kernel,
                            mode="valid")
        slope = slope * (slope > 0)  # we don't care about decreasing current
        slslope = np.convolve(np.pad(slope, sl_k_os, mode="symmetric"), sl_kernel,
                              mode="valid")
        slope2 = np.convolve(np.pad(sqrty, sl_k_os, mode="symmetric"), sl_kernel,
                            mode="valid")
        debug = pd.DataFrame(np.dstack((x_val, y_val, logy, logavg, avg, slope,
                                        slslope, sqrty, slope2))[0,:,:],
                             columns=["x", "y", "log(y)", "avg(log(y))", 
                                      "avg(y)", "log(y)'", "log(y)''",
                                      "sqrt(y)", "sqrt(y)'"])
    return mob_slope, v_th, v_turnon, ss, i_off, i_on, debug

def param_eval(data, on_index=None, fitting_boundaries=None):
    mob_slope, v_th, v_turnon, ss, i_off, i_on, debug = param_eval_core(
        data, on_index=on_index, fitting_boundaries=fitting_boundaries
    )
    return mob_slope, v_th, v_turnon, ss, i_off, i_on

#%%
def draw_analyzed_graph(
    book: pd.DataFrame,
    axes: pl.Axes,
    axes2: pl.Axes,
    mob: float,
    params: tuple[float, float, float, float, float, float],
    fontsize: int = 8
) -> None:
    draw_transfer(book, axes, axes2)
    # calculate things to draw
    x_min = book[book.columns[0]].min()
    x_max = book[book.columns[0]].max()
    mob_slope, v_th, v_turnon, ss, i_off, i_on = params
    onoff = i_on / i_off
    # show fitted line for mobility
    axes2.plot(
        np.array([v_th, x_max]),
        np.array([0, mob_slope*(x_max-v_th)]),
        "-r"
    )
    axes2.text(
        v_th + (x_max - v_th) * 0.25,
        (x_max - v_th) * 0.15 * mob_slope,
        r'$\mu\ = $' + f'{float(f"{mob:.3g}"):g}' + " cm$^2$/Vs",
        fontsize=fontsize
    )
    axes2.text(
        v_th + (x_max - v_th) * 0.25,
        (x_max - v_th) * 0.05 * mob_slope,
        r'$V_{\mathrm{Th}}$ = ' + f'{float(f"{v_th:.1f}"):g}' + " V",
        fontsize=fontsize
    )
    axes.text(
        x_min,
        i_on,
        r'$I_{\mathrm{On}} / I_{\mathrm{Off}}$ = ' + f'{float(f"{onoff:.1E}"):g}',
        fontsize=fontsize
    )
    if not v_turnon is None:
        # show off current and subthreshold swing
        x_s = [x_min, v_turnon + ss, v_turnon + ss, v_turnon, v_turnon]
        y_s = [i_off, i_off, i_off * 10, i_off * 10, i_off]
        axes.plot(x_s, y_s, "-r")
        axes.text(x_min, i_on / 10,
                  r'$V_{\mathrm{On}}$ = ' + \
                      f'{float(f"{v_turnon:.1f}"):g}' + " V",
                   fontsize=fontsize)
        axes.text(x_min, i_on / 100,
                  r'$S.S$ = ' + f'{float(f"{ss:.1f}"):g}' + " V/dec",
                   fontsize=fontsize)

def mobility_calc(mob_slope: float, ci: float, w: float, l: float) -> float:
    mob_multip = 2 * l / w / (ci / 10000)
    return mob_slope ** 2 * mob_multip

def create_record(
    name: str,
    mob: float,
    params: tuple[float, float, float, float, float, float]
) -> pd.DataFrame:
    _, v_th, v_turnon, ss, i_off, i_on = params
    onoff = i_on / i_off
    new_record = pd.DataFrame(
        [[
            name,
            mob,
            v_th,
            v_turnon,
            ss, onoff
        ]],
        columns=["Name", "Mobility", "VTh", "VOn", "S.S", "IOn/IOff"])
    return new_record

def evaluate(data_all, namelist: list[str], path: str):
    # IMPORTANT CONSTANTS
    ci = 3.45E-4 # F/m^2 (areal capacitance) SiO2 100 nm
    #ci = 1.18E-3 # F/m^2 (areal capacitance) Al2O3 50 nm
    #ci = 5.78E-5 # pF/mm^2 crosslinked PVP solution processed
    w = 2000 # micrometers (channel length)
    l = 200 # micrometers (channel width)
    # CHANGE THE CONSTANTS IF NEEDED
    data_table = pd.DataFrame(columns=["Name", "Mobility", "VTh", "VOn",
                                       "S.S", "IOn/IOff"])
    for index, name in enumerate(namelist):
        if list(data_all[index].columns.values)[0] == "VGS":
            # only taransfer curves are of interest
            book = data_all[index]
            book = book[["VGS", book.columns.values[-1]]]
            # evaluate parameters
            params = param_eval(book)
            mob = mobility_calc(params[0], ci, w, l)
            new_record = create_record(name, mob, params)
            data_table = pd.concat([data_table, new_record], axis=0)
            # create figures
            pl.rc('font', size=10)
            fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
            axes = fig.add_axes([0,0,1,1])
            axes.set_title(namelist[index], fontsize=16)
            axes2 = axes.twinx()
            draw_analyzed_graph(
                book, axes, axes2, mob, params
            )
            pl.savefig(path + SEP + namelist[index] + "_eval.png",
                       bbox_inches = 'tight')
            pl.show()
    return data_table

#%%
def draw_graphs(data_all, namelist, path):
    out_max_list: list[float] = []
    tr_max_list: list[float] = []
    tr_min_list: list[float] = []
    for index in range(len(namelist)):
        book = data_all[index]
        data = book.to_numpy()
        if list(book.columns.values)[0] == "VDS":
            out_max_list.append(data[-1, -1])
        elif list(book.columns.values)[0] == "VGS":
            tr_max_list.append(max(abs(data[-1, -1]), abs(data[0, -1])))
            tr_min_list.append(np.min(np.absolute(data[:, -1])))
    out_max = np.array(sorted(out_max_list), dtype=float)
    tr_max = np.array(sorted(tr_max_list), dtype=float)
    tr_min = np.array(sorted(tr_min_list), dtype=float)
    if len(out_max) == 0:
        outlimit = 1
    else:
        outlimit = min((np.median(out_max) + np.std(out_max) * 2.5) * 1.2,
                       out_max.max() * 1.2)
    if len(tr_max) == 0:
        trlimit = 1
    else:
        trlimit = min((np.median(tr_max) + np.std(tr_max) * 2.5) * 10,
                      tr_max.max() * 10)
    trlowlim = tr_min[(tr_min > 0)].min() / 2
    for index, name in enumerate(namelist):
        book = data_all[index]
        # create figures
        pl.rc('font', size=10)
        fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
        axes = fig.add_axes([0,0,1,1])
        axes.set_title(name, fontsize=16)
        if list(book.columns.values)[0] == "VDS":
            # output curve
            draw_output(book, axes, outlimit)
        elif list(book.columns.values)[0] == "VGS":
            # transfer curve
            axes2 = axes.twinx()
            draw_transfer(book, axes, axes2, trlimit, trlowlim)
        pl.savefig(path + SEP + name + ".png", bbox_inches = 'tight')
        pl.show()
    return

def extract_data(filename):
    if filename[-len(outend):] == outend or filename[-len(trend):] == trend:
        data = read_simple_file(filename)
        # COMMENT OUT THE FOLLOWING LINES IF YOU WANT TO WORK WITH THE
        # RAW DATA
        #if filename[-6:] == "tr.csv":
        #    conv_data = smooth_data(conv_data)
    elif filename[-len(excelend):] == excelend:
        data = read_excel(filename)
    else:
        data = file_convert(filename)
    return data

def process_directory(path):
    """
    Goes through all csv files in the directory, converts the data files, and
    creates the graphs.

    Parameters
    ----------
    path : string
        The path.
    """
    print(f"Processing directory {path}")
    filelist = glob.glob(path + SEP + "*.csv") + glob.glob(path + SEP + "*.xls")
    newnamelist = []
    data_all = []
    path_array = path.split(SEP)
    if len(path_array ) >= 1:
        dirname = path_array[-1] + " "
    else:
        dirname = ""
    for filename in filelist:
        print(f"  > {filename}")
        #data = pd.read_csv(filename, header=2)        
        conv_data = extract_data(filename)
        ind = 0
        fn_end = ["", " rev"]
        for data in conv_data:
            data_all.append(data)
            shortname = filename[filename.rfind(SEP)+1: ]
            newname = shortname[:-4]
            if len(shortname) > 13 and filename.find("CH") > -1:
                place = filename.find("CH")
                if list(data.columns.values)[0] == "VDS":
                    newname = filename[place: place+3] + " out" + fn_end[ind]
                elif list(data.columns.values)[0] == "VGS":
                    newname = filename[place: place+3] + " tr" + fn_end[ind]
                else:
                    newname = filename[place: place+3] + " NA" + fn_end[ind]
                newname =  dirname + newname
            else:
                newname = shortname[0: shortname.rfind(".")] + fn_end[ind]
            newnamelist.append(newname)
            ind += 1
            #data.to_csv(path + SEP + "c_" + newname + ".csv", index=False)
    draw_graphs(data_all, newnamelist, path)
    table = evaluate(data_all, newnamelist, path)
    table.to_csv(path + SEP + "characteristics.csv", index=False)
#%%
def graph_debug(data, name):
    pl.rc('font', size=10)
    fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
    axes = fig.add_axes([0,0,1,1])
    axes.set_title(name, fontsize=16)
    axes2 = axes.twinx()
    x = data[["x"]].to_numpy()[:,0]
    y_slope = data[["log(y)'"]].to_numpy()[:,0]
    y_slslope = data[["log(y)''"]].to_numpy()[:,0]
    y_sl_p = signal.find_peaks(y_slope[2:-2],
                               prominence=(y_slope[2:-2].max()-y_slope[2:-2].min())/5)
    y_slsl_p = signal.find_peaks(y_slslope[4:-4],
                                 prominence=(y_slslope[4:-4].max()-y_slslope[4:-4].min())/10)
    axes.plot(x, data[["log(y)"]], "b-", ms=3)
    axes.plot(x, data[["avg(log(y))"]], "r-", ms=3)
    axes2.plot(x, y_slope, "k-", ms=3)
    axes2.plot(x, y_slslope, "g-", ms=3)
    axes2.plot(x[y_sl_p[0]+2], y_slope[y_sl_p[0]+2], "bo", ms=3)
    axes2.plot(x[y_slsl_p[0]+4], y_slslope[y_slsl_p[0]+4], "ro", ms=3)
    pl.show()

#%%
def load_data(filename):
    raw_data = pd.read_csv(filename, delimiter=',', header=None)
    x_values = raw_data.to_numpy()[: , 0]
    y_values = raw_data.to_numpy()[:, 1: ]
    return x_values, y_values

trend = "tr.csv"
outend = "out.csv"
excelend = ".xls"
SEP = "/"
setvtgs = [0, 10, 20, 30, 40, 50, 60]

if __name__ == '__main__':
    print("")
    #for dirname in glob.glob("D:\\Gergely\\Documents\\CBNU files\\Data\\04. Oxide transistors\\04 gradient annealing of IGZO\\220516 grad\\*"):
    #    process_directory(dirname)
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Data\\01. Inorganic-organic hybrid phototransistor\\01. IGZO + PTCDI+C13\\210831\\D")
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Data\\04. Oxide transistors\\01 Indium oxide stability\\210831\\A")
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Data\\01. Inorganic-organic hybrid phototransistor\\01. IGZO + PTCDI+C13\\210902")
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Data\\04. Oxide transistors\\04 gradient annealing of IGZO\\220519 grad\\A")
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Data\\04. Oxide transistors\\04 gradient annealing of IGZO\\220519 grad\\B")
    #process_directory("D:\\Gergely\\Documents\\CBNU files\\Papers and Presentations\\In2O3 - IGZO sandwich layer\\All data\\9-60-9")
    #data_all = {}
    #for fn in ["A-7", "B-3", "B-5", "B-7", "C-2", "E-1", "E-2", "E-8"]:
    #    data = read_simple_file("data\\sandwich\\" + fn + " tr.csv", [0, 1, 2, 3])
    #    graph_debug(param_eval_core(data[0], True)[6], fn)
    #    graph_debug(param_eval_core(data[1], True)[6], fn + " rev")
    #    data_all[fn] = data
    #E = file_convert("data\\21-06-08_CH4_T_896.csv")
    #process_directory("data\\sandwich")
    
