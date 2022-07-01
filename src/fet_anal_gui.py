#!/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:07:03 2022

@author: Nice Micro
(C) Nice Micro 2021-2022
Licenced under the GPL v3.0
"""

import glob
import os
import tkinter as tk
from math import sqrt
from tkinter import ttk
from typing import Optional, Literal

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as pl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import fet_func as fet
from fet_gui_filelist import AnalParams, FileListWindow
from fet_gui_datatree import DataTree
from fet_gui_boxplot import BoxplotCtrl


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

    def __init__(self) -> None:
        tk.Tk.__init__(self)
        tk.Tk.columnconfigure(self, 0, weight=0)
        tk.Tk.columnconfigure(self, 1, weight=1)
        tk.Tk.rowconfigure(self, 0, weight=0)
        tk.Tk.rowconfigure(self, 1, weight=0)
        tk.Tk.rowconfigure(self, 2, weight=10)
        self.title("FET Analysis")

        self.folders: dict[str, FileListWindow] = {}
        self.sep = "/"
        self.ci: float = 3.45E-4
        self.l: float = 200
        self.w: float = 2000

        self.constantcontainer =ttk.Frame(self)
        self.constantcontainer.columnconfigure(2, weight=1)
        self.constantcontainer.columnconfigure(4, weight=1)
        self.constantcontainer.columnconfigure(6, weight=1)
        self.constantcontainer.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.inputcontainer = ttk.Frame(self)
        self.inputcontainer.columnconfigure(0, weight=1)
        graph_and_analyzers = ttk.Frame(self)
        graph_and_analyzers.grid(row=1, column=0, sticky="nsew")
        graph_and_analyzers.columnconfigure(0, weight=1)
        self.filelistplace = ttk.Notebook(self)
        self.filelistplace.grid(row=1, column=1, rowspan=2, sticky="nsew")
        self.datalistplace = ttk.Notebook(self)
        self.datalistplace.grid(row=2, column=0, sticky="nsew")

        self.folder_text = tk.StringVar()
        self._folder_entry = ttk.Entry(
            self.inputcontainer, textvariable=self.folder_text, width=60
        )
        self._folder_entry.grid(row=0, column=0, sticky="nsew")
        ttk.Button(
            self.inputcontainer, text="Open Folder", command=self.add_folder
        ).grid(row=0, column=1, sticky="nsew")

        self.capacitance_text = tk.StringVar()
        self.capacitance_text.set("3.45E-4")
        self.width_text = tk.StringVar()
        self.width_text.set("2000")
        self.length_text = tk.StringVar()
        self.length_text.set("200")
        ttk.Label(
            self.constantcontainer, text="Areal capacitance (pF/m^2):"
        ).grid(row=0, column=0, sticky="nsew")
        ttk.Entry(
            self.constantcontainer, textvariable=self.capacitance_text
        ).grid(row=0, column=2, sticky="nsew")
        ttk.Label(
            self.constantcontainer, text="Channel width (um):"
        ).grid(row=0, column=3, sticky="nsew", padx=(15, 0))
        ttk.Entry(
            self.constantcontainer, textvariable=self.width_text
        ).grid(row=0, column=4, sticky="nsew")
        ttk.Label(
            self.constantcontainer, text="Channel length (um):"
        ).grid(row=0, column=5, sticky="nsew", padx=(15, 0))
        ttk.Entry(
            self.constantcontainer, textvariable=self.length_text
        ).grid(row=0, column=6, sticky="nsew")
        ttk.Button(
            self.constantcontainer, text="Set", command=self.constants_set
        ).grid(row=0, column=7, sticky="nsew")

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

        self.alldata_frame = DataTree(self.datalistplace)
        self.datalistplace.add(self.alldata_frame, text="Data analysis results")
        self.boxplot_ctrl = BoxplotCtrl(self.datalistplace)
        self.datalistplace.add(self.boxplot_ctrl, text="Create summary graphs")

        self.bind("<<TreeviewSelect>>", self.treeview_element_selected)
        self.bind("<<NotebookTabChanged>>", self.notebook_tab_changed)
        self.bind("<Return>", self.enter_pressed)
        self.bind("<<NewAnalysisResults>>", self.anal_results_updated)
        self.bind("<<DrawBoxplot>>", self.draw_boxplot)

    def constants_set(self) -> None:
        ci: str = self.capacitance_text.get()
        w: str = self.width_text.get()
        l: str = self.length_text.get()
        try:
            self.ci = float(ci)
        except ValueError:
            return
        try:
            self.w = float(w)
        except ValueError:
            return
        try:
            self.l = float(l)
        except ValueError:
            return
        self.constantcontainer.grid_forget()
        self.inputcontainer.grid(row=0, column=0, columnspan=2, sticky="nsew")

    def enter_pressed(self, event: tk.Event) -> None:
        if event.widget == self._folder_entry:
            self.add_folder()

    def notebook_tab_changed(self, event: tk.Event) -> None:
        if event.widget == self.filelistplace:
            self.file_selected()

    def treeview_element_selected(self, event: tk.Event) -> None:
        caller = event.widget
        if caller in [frame.filelist_view for frame in self.folders.values()]:
            self.file_selected()

    def add_folder(self) -> None:
        path = self.folder_text.get()
        if not os.path.isdir(path):
            return
        if path[-1] == self.sep:
            path = path[:-1]
        name = path.split(self.sep)[-1]
        while name in self.folders:
            name += "+"
        child = FileListWindow(
            path,
            name,
            self.filelistplace,
            self.ci,
            self.w,
            self.l,
            self.sep
        )
        self.folders[name] = child
        self.filelistplace.add(child, text=name)
        self.filelistplace.select(len(self.folders)-1)
        self.alldata_frame.add_folder(name)

    def file_selected(self) -> None:
        foldername = self.filelistplace.tab(self.filelistplace.select(), "text")
        filelist_view = self.folders[foldername]
        sel_num, subsel, plottingtype, datatype, book = filelist_view.get_selection()
        name = filelist_view.get_selected_name()

        for axes in self.graph.get_axes():
            self.graph.delaxes(axes)

        if sel_num == -1:
            self.canvas.draw()
            return

        if plottingtype == -1 or datatype == "out":
            for slider in self.sliders.values():
                slider.state(['disabled'])
        else:
            for slider in self.sliders.values():
                slider.state(['!disabled'])

        if datatype == "out":
            # output curve
            axes = self.graph.add_axes([0.3, 0.25, 0.65, 0.55])
            axes.set_title(name, fontsize=14)
            fet.draw_output(book, axes, label_font_s=10)
        elif datatype == "tr":
            # transfer curve
            axes = self.graph.add_axes([0.25, 0.25, 0.52, 0.55])
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

    def anal_results_updated(self, event: tk.Event) -> None:
        caller = event.widget
        assert isinstance(caller, FileListWindow)
        if caller not in self.folders.values():
            return
        foldername: str = caller.name
        result: Optional[fet.FetResult]
        while True:
            changed_value = caller.get_changed_result()
            if changed_value is None:
                break
            (result, filenum, sweepnum, col_num) = changed_value
            if result is None:
                self.alldata_frame.del_data(
                    foldername, filenum, sweepnum, col_num
                )
            else:
                self.alldata_frame.add_data(
                    result, foldername, filenum, sweepnum, col_num
                )

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
        data: pd.DataFrame = table[[table.columns[0], table.columns[col_ind]]]
        turn_on, fit_left, fit_right = filelist_view.analyze_selected_transfer()
        self.sliding = -3
        self.sliders["Turn-ON"].set(turn_on / len(data) * 100)
        self.sliders["Fit from"].set(fit_left / len(data) * 100)
        self.sliders["Fit to"].set(fit_right / len(data) * 100)
        result = fet.param_eval(
            data,
            on_index=turn_on,
            fitting_boundaries=(fit_left, fit_right)
        )
        mob = fet.mobility_calc(result[0], self.ci, self.w, self.l)
        filelist_view.save_selected_results(mob, result)
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
        name = filelist_view.get_selected_name()
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

    def draw_boxplot(self, _: tk.Event) -> None:
        data_to_plot: fet.Parameters
        to_file: bool
        data_to_plot, to_file = self.boxplot_ctrl.graph_info()
        points: list[npt.ArrayLike] = []
        dataset_names: list[str] = []
        path: Optional[str] = None
        for group, folder_data in self.folders.items():
            if path is None:
                path = folder_data.path
            else:
                newpath: str = ""
                for folder1, folder2 in zip(
                    path.split(self.sep),
                    folder_data.path.split(self.sep)
                ):
                    if folder1 != folder2:
                        break
                    newpath = newpath + self.sep + folder1
                path = newpath
            column: Optional[npt.ArrayLike] = None
            for result_index in folder_data.list_all_calculated():
                result = folder_data.get_results(*result_index)
                assert result is not None
                result_point = result[data_to_plot]
                if result_point is None:
                    continue
                assert isinstance(result_point, float)
                if column is None:
                    column = np.array([result_point])
                    continue
                column = np.append(column, result_point)
            if column is not None:
                points.append(column)
                dataset_names.append(group)

        for axes in self.graph.get_axes():
            self.graph.delaxes(axes)
        if len(points) == 0:
            return
        assert path is not None
        if to_file:
            self.export_boxplot(
                fet.PARAMNAMES[data_to_plot],
                path,
                points,
                dataset_names
            )
        axes = self.graph.add_axes([0.25,0.25,0.52,0.55])
        axes.set_title(fet.PARAMNAMES[data_to_plot], fontsize=14)
        fet.draw_boxplot(points, axes, labels=dataset_names)
        self.canvas.draw()

    def export_boxplot(
        self,
        title: str,
        path: str,
        points: list[npt.ArrayLike],
        dataset_names: list[str]
    ) -> None:
        fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
        axes = fig.add_axes([0,0,1,1])
        axes.set_title(title, fontsize=16)
        fet.draw_boxplot(points, axes, labels=dataset_names)
        pl.savefig(
            f"{path}{self.sep}{title}.png",
            bbox_inches = 'tight'
        )
        pl.close(fig)

def main() -> None:
    app = AppContainer()
    app.mainloop()


if __name__ == "__main__":
    main()
