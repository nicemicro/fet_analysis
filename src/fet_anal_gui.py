#!/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:07:03 2022

@author: Nice Micro
(C) Nice Micro 2021-2022
Licenced under the GPL v3.0
"""

import glob
import tkinter as tk
from math import sqrt
from tkinter import ttk
from typing import Optional, Literal

import pandas as pd
import pylab as pl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import fet_func as fet

SEP = "/"
# IMPORTANT CONSTANTS
#TODO: constants should be changed in the UI
ci = 3.45E-4 # F/m^2 (areal capacitance) SiO2 100 nm
#ci = 1.18E-3 # F/m^2 (areal capacitance) Al2O3 50 nm
#ci = 5.78E-5 # pF/mm^2 crosslinked PVP solution processed
w = 2000 # micrometers (channel length)
l = 200 # micrometers (channel width)

class AnalParams():
    def __init__(self, turn_on: int, fit_left: int, fit_right: int) -> None:
        self.turn_on: int = turn_on
        self.fit_left: int = fit_left
        self.fit_right: int = fit_right

class FileListWindow(ttk.Frame):
    def __init__(
        self,
        path: str,
        name: str,
        master: ttk.Notebook,
        *args, **kwargs
    ) -> None:
        ttk.Frame.__init__(self, *args, **kwargs)

        self.path: str = path
        self.name: str = name
        self.filelist: list[Optional[str]] = []
        self.namelist: dict[int, str] = {}
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
        for rownum in [1, 2, 3, 4]:
            main_frame.rowconfigure(rownum, weight=0)

        self.filelist_view = ttk.Treeview(
            main_frame,
            selectmode="browse",
            columns=("name", "load")
        )
        self.filelist_view.heading("#0", text="Filename")
        self.filelist_view.column("load", minwidth=20, width=30, stretch=tk.NO)
        self.filelist_view.heading("name", text="Name")
        self.filelist_view.heading("load", text="Load")
        scrollbar = ttk.Scrollbar(
            main_frame,
            orient="vertical",
            command=self.filelist_view.yview
        )
        hscrollbar = ttk.Scrollbar(
            main_frame,
            orient="horizontal",
            command=self.filelist_view.xview
        )
        self.filelist_view.configure(yscrollcommand=scrollbar.set)
        self.filelist_view.configure(xscrollcommand=hscrollbar.set)
        self.filelist_view.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="nsw")
        hscrollbar.grid(row=1, column=0, sticky="new")
        ttk.Button(
            main_frame, text="Remove", command=self.delete_selected
        ).grid(row=2, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            main_frame, text="Cache all", command=self.cache_all
        ).grid(row=3, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            main_frame, text="Export cached data", command=self.export_cached
        ).grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.filelist += glob.glob(self.path + SEP + "*.csv")
        self.filelist += glob.glob(self.path + SEP + "*.xls")
        self.filelist.sort()

        for itemnum in self.filelist_view.get_children():
            self.filelist_view.delete(itemnum)
        for index, full_filename in enumerate(self.filelist):
            if full_filename is None:
                continue
            filename = full_filename[len(self.path)+1:]
            self.namelist[index] = ".".join(filename.split(".")[:-1])
            self.filelist_view.insert(
                "",
                "end",
                iid=str(index),
                text=filename,
                values=(self.namelist[index],"")
            )

    def export_cached(self) -> None:
        result_list: list[pd.DataFrame] = []
        file_res_indeces = list(self.anal_res.keys())
        file_res_indeces.sort()
        for file_index in file_res_indeces:
            sub_indeces = list(self.anal_res[file_index].keys())
            sub_indeces.sort()
            for sub_index in sub_indeces:
                column_indeces = list(self.anal_res[file_index][sub_index].keys())
                column_indeces.sort()
                for column_index in column_indeces:
                    result_list.append(self.anal_res[file_index][sub_index][column_index])
        if len(result_list) == 0:
            return
        result = pd.concat(result_list)
        result = result.reset_index(drop=True)
        result.to_csv(f"{self.path}{SEP}result.csv", index=False)

    def delete_selected(self) -> None:
        sel_num, subsel, colnum = self.get_selection_nums()
        if sel_num == -1:
            return
        if subsel == -1:
            self.filelist_view.delete(f"{sel_num}")
            self.filelist[sel_num] = None
            if sel_num in self.all_data:
                self.all_data.pop(sel_num)
            if sel_num in self.anal_param:
                self.anal_param.pop(sel_num)
                self.anal_res.pop(sel_num)
            return
        if colnum == -1:
            self.filelist_view.delete(f"{sel_num}-{subsel}")
            self.filelist_view.selection_add(f"{sel_num}")
            if subsel in self.anal_param[sel_num]:
                self.anal_param[sel_num].pop(subsel)
                self.anal_res[sel_num].pop(subsel)
            return
        self.filelist_view.delete(f"{sel_num}-{subsel}-{colnum}")
        self.filelist_view.selection_add(f"{sel_num}-{subsel}")
        if colnum in self.anal_param[sel_num][subsel]:
            self.anal_param[sel_num][subsel].pop(colnum)
            self.anal_res[sel_num][subsel].pop(colnum)
        return

    def get_selection_nums(self) -> tuple[int, int, int]:
        if len(self.filelist_view.selection()) == 0:
            return -1, -1, -1
        selection: list[str] = (self.filelist_view.selection()[0]).split("-")
        if len(selection) == 0:
            return -1, -1, -1
        sel_num: int = int(selection[0])
        subsel: int = -1
        colnum: int = -1
        if len(selection) > 1:
            subsel = int(selection[1])
        if len(selection) > 2:
            colnum = int(selection[2])
        return sel_num, subsel, colnum

    def _cache_transfer_curve(self, index: int, subindex: int, col_ind: int) -> None:
        turn_on, fit_left, fit_right = self._analyze_transfer_curve(index, subindex, col_ind)
        if col_ind in self.anal_res[index][subindex]:
            return
        self.anal_param[index][subindex][col_ind] = AnalParams(turn_on, fit_left, fit_right)
        table = self.all_data[index][subindex]
        data: pd.DataFrame = table[[table.columns[0], table.columns[col_ind]]]
        result = fet.param_eval(
            data,
            on_index=turn_on,
            fitting_boundaries=(fit_left, fit_right)
        )
        mob = fet.mobility_calc(result[0], ci, w, l)
        self._save_analysis_results(index, subindex, col_ind, mob, result)

    def cache_all(self) -> None:
        for index, filename in enumerate(self.filelist):
            if filename is None:
                continue
            if index not in self.all_data:
                data_storage = self._get_file_data(index)
            else:
                data_storage = self.all_data[index]
            if len(data_storage) == 0:
                continue
            assert fet.data_type(data_storage[0]) != ""
            if (fet.data_type(data_storage[0]) == "tr"):
                for subindex, _ in enumerate(data_storage):
                    for col_ind, colname in enumerate(data_storage[subindex].columns):
                        if colname == "VGS":
                            continue
                        self._cache_transfer_curve(index, subindex, col_ind)
                        self.filelist_view.item(
                            f"{index}-{subindex}-{col_ind}",
                            values=("", "✓")
                        )

    def _get_file_data(self, file_ind: int) -> list[pd.DataFrame]:
        new_data: bool = False
        selected: Optional[str] = self.filelist[file_ind]
        if file_ind in self.all_data:
            data_storage = self.all_data[file_ind]
        else:
            data_storage = fet.extract_data_from_file(selected)
            self.all_data[file_ind] = data_storage
            self.filelist_view.item(
                f"{file_ind}",
                values=(self.namelist[file_ind], "✓")
            )
            new_data = True
        if len(data_storage) == 0:
            return data_storage
        if fet.data_type(data_storage[0]) == "out":
            if new_data and len(data_storage) > 1:
                for index, _ in enumerate(data_storage):
                    self.filelist_view.insert(
                        f"{file_ind}",
                        "end",
                        iid=f"{file_ind}-{index}",
                        text=f"Sweep-{index+1}",
                        values=(f"{self.name[file_ind]}-{index+1}","")
                    )
                self.filelist_view.see(f"{file_ind}-0")
        elif fet.data_type(data_storage[0]) == "tr":
            if new_data:
                for index, sub_data in enumerate(data_storage):
                    self.filelist_view.insert(
                        f"{file_ind}",
                        "end",
                        iid=f"{file_ind}-{index}",
                        text=f"Sweep-{index+1}",
                        values=(f"{self.namelist[file_ind]}-{index+1}","")
                    )
                    for i2, colname in enumerate(sub_data.columns):
                        if colname == "VGS":
                            continue
                        self.filelist_view.insert(
                            f"{file_ind}-{index}",
                            "end",
                            iid=f"{file_ind}-{index}-{i2}",
                            text=f"Calc: {colname}"
                        )
                    self.filelist_view.see(f"{file_ind}-{index}-1")
                self.filelist_view.see(f"{file_ind}-0")
        return data_storage

    def get_selection(self) -> tuple[int, int, int, Literal["", "out", "tr"], pd.DataFrame]:
        sel_num, subsel, col_ind = self.get_selection_nums()
        if sel_num == -1:
            return -1, -1, -1, "", [pd.DataFrame]
        selected: Optional[str] = self.filelist[sel_num]
        assert selected is not None, "Deleted item selected somehow"
        data_storage: list[pd.DataFrame] = self._get_file_data(sel_num)
        if subsel >= 0:
            book = data_storage[subsel]
        elif subsel == -1:
            if len(data_storage) > 0:
                book = pd.concat(data_storage)
            else:
                return -1, -1, -1, "", [pd.DataFrame]
        else:
            assert False, f"Invalid subsel value {subsel}"
        datatype: Literal["", "out", "tr"] = fet.data_type(book)
        if datatype == "":
            return -1, -1, -1, "", [pd.DataFrame()]
        return sel_num, subsel, col_ind, datatype, book

    def get_selected_filename(self) -> str:
        if len(self.filelist_view.selection()) == 0:
            return ""
        selection: list[str] = (self.filelist_view.selection()[0]).split("-")
        sel_num: int = int(selection[0])
        filename = self.filelist[sel_num]
        assert filename is not None, "Somehow deleted item got selected"
        return filename[len(self.path)+1:]

    def get_selected_name(self) -> str:
        if len(self.filelist_view.selection()) == 0:
            return ""
        selection: list[str] = (self.filelist_view.selection()[0]).split("-")
        sel_num: int = int(selection[0])
        return self.namelist[sel_num]

    def analyze_selected_transfer(self) -> tuple[int, int, int]:
        sel, subsel, col_ind = self.get_selection_nums()
        return self._analyze_transfer_curve(sel, subsel, col_ind)

    def _analyze_transfer_curve(self, index, sub_ind, col_ind) -> tuple[int, int, int]:
        if not index in self.anal_param:
            self.anal_param[index] = {}
            self.anal_res[index] = {}
        if not sub_ind in self.anal_param[index]:
            self.anal_param[index][sub_ind] = {}
            self.anal_res[index][sub_ind] = {}
        if not col_ind in self.anal_param[index][sub_ind]:
            table: pd.DataFrame = self.all_data[index][sub_ind]
            data: pd.DataFrame = table[[table.columns[0], table.columns[col_ind]]]
            turn_on = fet.find_on_index(data)
            fit_left, fit_right = fet.find_fitting_boundaries(data)
            self.anal_param[index][sub_ind][col_ind] = AnalParams(turn_on, fit_left, fit_right)
        turn_on = self.anal_param[index][sub_ind][col_ind].turn_on
        fit_left = self.anal_param[index][sub_ind][col_ind].fit_left
        fit_right = self.anal_param[index][sub_ind][col_ind].fit_right
        return turn_on, fit_left, fit_right

    def save_selected_results(
        self,
        mob: float,
        result: tuple[float, float, float, float, float, float]
    ) -> None:
        sel, subsel, col_ind = self.get_selection_nums()
        self._save_analysis_results(sel, subsel, col_ind, mob, result)

    def _save_analysis_results(
        self,
        index: int,
        subindex: int,
        col_ind: int,
        mob: float,
        result: tuple[float, float, float, float, float, float]
    ) -> None:
        desc: dict[str, str] = {"Name": self.namelist[index]}
        if len(self.all_data[index]) > 1:
            desc["Sweep"] = f"{subindex}"
        if len(self.all_data[index][subindex].columns) > 2:
            desc["VDS"] = self.all_data[index][subindex].columns[col_ind].replace("VDS = ", "")
        self.anal_res[index][subindex][col_ind] = fet.create_record(desc, mob, result)
        self.filelist_view.item(
            f"{index}-{subindex}-{col_ind}",
            values=("","✓")
        )

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
            inputcontainer, text="Open Folder", command=self.add_folder
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
        self.bind("<<NotebookTabChanged>>", self.file_selected)

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
        mob = fet.mobility_calc(result[0], ci, w, l)
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

def main() -> None:
    app = AppContainer()
    app.mainloop()


if __name__ == "__main__":
    main()
