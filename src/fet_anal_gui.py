#!/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:07:03 2022

@author: Nice Micro
(C) Nice Micro 2021-2022
Licenced under the GPL v3.0
"""

import pandas as pd
from math import sqrt
import glob
import tkinter as tk
from tkinter import ttk
import pylab as pl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import fet_func as fet

SEP = "/"

class AnalParams():
    def __init__(self, turn_on: int, fit_left: int, fit_right: int) -> None:
        self.turn_on: int = turn_on
        self.fit_left: int = fit_left
        self.fit_right: int = fit_right

class FileListWindow(ttk.Frame):
    def __init__(self, path: str, name: str, master: ttk.Notebook, *args, **kwargs) -> None:
        ttk.Frame.__init__(self, *args, **kwargs)

        self.path: str = path
        self.name: str = name
        self.filelist: list[str] = []
        self.all_data: dict[int, list[pd.DataFrame]] = {}
        self.anal_param: dict[int, dict[int, dict[int, AnalParams]]] = {}
        self.anal_res: dict[int, dict[int, dict[int, pd.Series]]] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self)
        main_frame.grid(column=0, row=0, sticky="nswe")
        main_frame.columnconfigure(0, weight=10)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=10)

        self.filelist_view = ttk.Treeview(main_frame, selectmode="browse")
        self.filelist_view.heading('#0', text='Filename')
        scrollbar = ttk.Scrollbar(
            main_frame,
            orient="vertical",
            command=self.filelist_view.yview
        )
        self.filelist_view.configure(yscrollcommand=scrollbar.set)
        self.filelist_view.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="nsw")
        self.filelist = (
            glob.glob(self.path + SEP + "*.csv") +
            glob.glob(self.path + SEP + "*.xls")
        )
        self.filelist.sort()
        self.all_data = {}
        self.anal_param = {}
        self.anal_res = {}
        for itemnum in self.filelist_view.get_children():
            self.filelist_view.delete(itemnum)
        for index, filename in enumerate(self.filelist):
            self.filelist_view.insert(
                "",
                "end",
                iid=str(index),
                text=filename[len(self.path)+1:]
            )
    def get_selection_nums(self) -> tuple[int, int, int]:
        selection: list[str] = (self.filelist_view.selection()[0]).split("-")
        sel_num: int = int(selection[0])
        subsel: int = -1
        colnum: int = -1
        if len(selection) > 1:
            subsel = int(selection[1])
        if len(selection) > 2:
            colnum = int(selection[2])
        return sel_num, subsel, colnum

    def get_selection(self) -> tuple[int, int, int, str, pd.DataFrame]:
        new_data: bool = False
        sel_num, subsel, col_ind = self.get_selection_nums()
        selected: str = self.filelist[sel_num]
        if sel_num in self.all_data:
            data_storage = self.all_data[sel_num]
        else:
            data_storage = fet.extract_data_from_file(selected)
            self.all_data[sel_num] = data_storage
            new_data = True
        if subsel >= 0:
            book = data_storage[subsel]
        elif subsel == -1:
            if len(data_storage) > 0:
                book = pd.concat(data_storage)
            else:
                return -1, -1, -1, "", [pd.DataFrame]
        else:
            assert False, f"Invalid subsel value {subsel}"

        datatype: str
        if list(book.columns.values)[0] == "VDS":
            datatype = "out"
            if new_data and len(data_storage) > 1:
                for index, _ in enumerate(data_storage):
                    self.filelist_view.insert(
                        f"{sel_num}",
                        "end",
                        iid=f"{sel_num}-{index}",
                        text=f"Sub-{index+1}"
                    )
                self.filelist_view.see(f"{sel_num}-0")
        elif list(book.columns.values)[0] == "VGS":
            datatype = "tr"
            if new_data:
                for index, sub_data in enumerate(data_storage):
                    self.filelist_view.insert(
                        f"{sel_num}",
                        "end",
                        iid=f"{sel_num}-{index}",
                        text=f"Sub-{index+1}"
                    )
                    for i2, colname in enumerate(sub_data.columns):
                        if colname == "VGS":
                            continue
                        self.filelist_view.insert(
                            f"{sel_num}-{index}",
                            "end",
                            iid=f"{sel_num}-{index}-{i2}",
                            text=f"Calculations: {colname}"
                        )
                    self.filelist_view.see(f"{sel_num}-{index}-1")
                self.filelist_view.see(f"{sel_num}-0")
        else:
            return -1, -1, -1, "", [pd.DataFrame()]
        return sel_num, subsel, col_ind, datatype, book

    def get_name_selected(self) -> str:
        selection: list[str] = (self.filelist_view.selection()[0]).split("-")
        sel_num: int = int(selection[0])
        return self.filelist[sel_num][len(self.path)+1]

    def analyze_transfer_curve(self) -> tuple[int, int, int]:
        sel, subsel, col_ind = self.get_selection_nums()
        if not sel in self.anal_param:
            self.anal_param[sel] = {}
            self.anal_res[sel] = {}
        if not subsel in self.anal_param[sel]:
            self.anal_param[sel][subsel] = {}
            self.anal_res[sel][subsel] = {}
        if not col_ind in self.anal_param[sel][subsel]:
            table: pd.DataFrame = self.all_data[sel][subsel]
            data: pd.DataFrame = table[[table.columns[0], table.columns[col_ind]]]
            turn_on = fet.find_on_index(data)
            fit_left, fit_right = fet.find_fitting_boundaries(data)
            self.anal_param[sel][subsel][col_ind] = AnalParams(turn_on, fit_left, fit_right)
        turn_on = self.anal_param[sel][subsel][col_ind].turn_on
        fit_left = self.anal_param[sel][subsel][col_ind].fit_left
        fit_right = self.anal_param[sel][subsel][col_ind].fit_right
        return turn_on, fit_left, fit_right

    def save_analysis_results(
        self,
        mob: float,
        result: tuple[float, float, float, float, float, float]
    ) -> None:
        name = self.get_name_selected()
        sel, subsel, col_ind = self.get_selection_nums()
        self.anal_res[sel][subsel][col_ind] = fet.create_record(name, mob, result)

    def save_analysis_params(self, parameters: AnalParams) -> None:
        sel, subsel, col = self.get_selection_nums()
        self.anal_param[sel][subsel][col] = parameters

class AppContainer(tk.Tk):
    """The main window of the App"""
    
    def make_slider(self, master: ttk.Frame, text: str, rownum: int) -> ttk.Scale:
        ttk.Label(master, text=text).grid(row=rownum, column=0, sticky="nse")
        new_scale = ttk.Scale(
            master,
            orient="horizontal",
            length=300,
            from_=0,
            to=100,
            state="disabled",
            command=lambda x: self.set_slider(text, x)
        )
        new_scale.grid(row=rownum, column=1, sticky="nswe")
        return new_scale
    
    def __init__(self, *args, **kwargs) -> None:
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.columnconfigure(self, 0, weight=1)
        tk.Tk.rowconfigure(self, 0, weight=0)
        tk.Tk.rowconfigure(self, 1, weight=10)
        self.title("FET Analysis")

        self.folders: dict[str, FileListWindow] = {}

        inputcontainer = ttk.Frame(self)
        inputcontainer.grid(row=0, column=0, sticky="nsew")
        inputcontainer.columnconfigure(0, weight=1)
        inputcontainer.rowconfigure(0, weight=10)
        inputcontainer.rowconfigure(1, weight=0)
        actioncontainer = ttk.Frame(self)
        actioncontainer.grid(row=1, column=0, sticky="nsew")
        actioncontainer.rowconfigure(0, weight=10)
        actioncontainer.columnconfigure(0, weight=0)
        actioncontainer.columnconfigure(1, weight=10)
        graph_and_analyzers = ttk.Frame(actioncontainer)
        graph_and_analyzers.grid(row=0, column=0, sticky="nsew")
        graph_and_analyzers.columnconfigure(0, weight=1)
        self.filelistplace = ttk.Notebook(actioncontainer)
        self.filelistplace.grid(row=0, column=1, sticky="nsew")

        self.folder_text = tk.StringVar()
        ttk.Entry(
            inputcontainer, textvariable=self.folder_text, width=60
        ).grid(row=0, column=0, sticky="nsew")
        ttk.Button(
            inputcontainer, text="Search", command=self.add_folder
        ).grid(row=0, column=1, sticky="nsew")

        pl.rc('font', size=8)
        self.graph = pl.figure(figsize=(7/2.54, 5/2.54), dpi=180)
        self.canvas = FigureCanvasTkAgg(self.graph, master=graph_and_analyzers)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)

        self.sliders: dict[str, ttk.Scale] = {}
        for index, text in enumerate(["Turn-ON", "Fit from", "Fit to"]):
            self.sliders[text] = self.make_slider(
                graph_and_analyzers, f"{text}:", index + 2
            )
        self.sliding: int = 0
        self.bind("<<TreeviewSelect>>", self.file_selected)
    
    def add_folder(self):
        path = self.folder_text.get()
        name = path.split(SEP)[-1]
        while name in self.folders:
            name += "+"
        child = FileListWindow(path, name, self.filelistplace)
        #child = ttk.Frame(self.filelistplace)
        self.folders[name] = child
        self.filelistplace.add(child, text=name)

    def file_selected(self, _: tk.Event):
        name = self.filelistplace.tab(self.filelistplace.select(), "text")
        filelist_view = self.folders[name]
        sel_num, subsel, plottingtype, datatype, book = filelist_view.get_selection()
        if sel_num == -1:
            return

        for axes in self.graph.get_axes():
            self.graph.delaxes(axes)

        if plottingtype == -1 or datatype == "out":
            for slider in self.sliders.values():
                slider.state(['disabled'])
        else:
            for slider in self.sliders.values():
                slider.state(['!disabled'])

        if datatype == "out":
            # output curve
            axes = self.graph.add_axes([0.3,0.25,0.65,0.55])
            axes.set_title(name, fontsize=14)
            fet.draw_output(book, axes, label_font_s=10)
        elif datatype == "tr":
            # transfer curve
            axes = self.graph.add_axes([0.25,0.25,0.52,0.55])
            axes.set_title(name, fontsize=14)
            axes2 = axes.twinx()
            if plottingtype == -1:
                fet.draw_transfer(book, axes, axes2, label_font_s=10)
            else:
                self.transfer_analysis(
                    sel_num,
                    subsel,
                    plottingtype,
                    book,
                    axes,
                    axes2,
                    label_font_s=10
                )
        self.canvas.draw()

    def transfer_analysis(
        self,
        sel: int,
        subsel: int,
        col_ind: int,
        table: pd.DataFrame,
        axes: pl.Axes,
        axes2: pl.Axes,
        label_font_s: int = 14
    ) -> None:
        turn_on: int
        fit_left: int
        fit_right: int
        foldername = self.filelistplace.tab(self.filelistplace.select(), "text")
        filelist_view = self.folders[foldername]
        name: str = filelist_view.get_name_selected()
        # IMPORTANT CONSTANTS
        #TODO: constants should be changed in the UI
        ci = 3.45E-4 # F/m^2 (areal capacitance) SiO2 100 nm
        #ci = 1.18E-3 # F/m^2 (areal capacitance) Al2O3 50 nm
        #ci = 5.78E-5 # pF/mm^2 crosslinked PVP solution processed
        w = 2000 # micrometers (channel length)
        l = 200 # micrometers (channel width)
        data: pd.DataFrame = table[[table.columns[0], table.columns[col_ind]]]
        turn_on, fit_left, fit_right = filelist_view.analyze_transfer_curve()
        self.sliding = -3
        self.sliders["Turn-ON"].set(turn_on / len(data) * 100)
        self.sliders["Fit from"].set(fit_left / len(data) * 100)
        self.sliders["Fit to"].set(fit_right / len(data) * 100)
        result = fet.param_eval(
            data,
            on_index=turn_on,
            fitting_boundaries=(fit_left, fit_right)
        )
        mob = fet.mobility_calc(result[0], ci, w, l)
        filelist_view.save_analysis_results(mob, result)
        fet.draw_analyzed_graph(
            data,
            axes,
            axes2,
            mob,
            result,
            fontsize=5,
            label_font_s=label_font_s
        )
        self.add_fitting_range(axes2, data, fit_left, fit_right)

    def add_fitting_range(
        self,
        axes: pl.Axes,
        data: pd.DataFrame,
        fit_left: int,
        fit_right: int
    ) -> None:
        data = fet.sort_transfer_curve(data)
        start_v: float = data.iat[fit_left, 0]
        end_v: float = data.iat[fit_right-1, 0]
        top: float = sqrt(data[data.columns[1]].max()) * 1.2
        axes.plot([start_v, start_v], [0, top], "-g", linewidth=0.5)
        axes.plot([end_v, end_v], [0, top], "-g", linewidth=0.5)

    def set_slider(self, _: str, __: str) -> None:
        if self.sliding == 0:
            self.after(100, self.read_slider_pos)
        self.sliding += 1

    def read_slider_pos(self) -> None:
        foldername = self.filelistplace.tab(self.filelistplace.select(), "text")
        filelist_view = self.folders[foldername]
        sel, sub_sel, col, _, table = filelist_view.get_selection()
        name = filelist_view.get_name_selected()
        turn_on = int(round(self.sliders["Turn-ON"].get() * len(table) / 100))
        fit_left = int(round(self.sliders["Fit from"].get() * len(table) / 100))
        fit_right = int(round(self.sliders["Fit to"].get() * len(table) / 100))
        filelist_view.save_analysis_params(AnalParams(turn_on, fit_left, fit_right))

        for axes in self.graph.get_axes():
            self.graph.delaxes(axes)
        axes = self.graph.add_axes([0.25,0.25,0.52,0.55])
        axes.set_title(name, fontsize=14)
        axes2 = axes.twinx()
        self.transfer_analysis(sel, sub_sel, col, table, axes, axes2, label_font_s=10)
        self.canvas.draw()

        self.sliding = 0

def main() -> None:
    app = AppContainer()
    app.mainloop()


if __name__ == "__main__":
    main()
