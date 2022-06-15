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
from tkinter import ttk
from typing import Optional, Literal

import pandas as pd
import matplotlib.pyplot as pl

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
        master: ttk.Notebook
    ) -> None:
        ttk.Frame.__init__(self, master=master)

        self.path: str = path
        self.name: str = name
        self.filelist: list[Optional[str]] = []
        self.namelist: dict[int, str] = {}
        self.all_data: dict[int, dict[int, pd.DataFrame]] = {}
        self.anal_param: dict[int, dict[int, dict[int, AnalParams]]] = {}
        self.anal_res: dict[int, dict[int, dict[int, fet.FetResult]]] = {}

        self.recently_generated: list[tuple[int, int, int]] = []

        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=10)
        for rownum in [1, 2, 3, 4, 5, 6]:
            self.rowconfigure(rownum, weight=0)

        self.filelist_view = ttk.Treeview(
            self,
            selectmode="browse",
            columns=("name", "load")
        )
        self.filelist_view.heading("#0", text="Filename")
        self.filelist_view.heading("name", text="Name")
        self.filelist_view.heading("load", text="Load")
        self.filelist_view.column("#0", minwidth=100, width=200)
        self.filelist_view.column("name", minwidth=100, width=200)
        self.filelist_view.column("load", minwidth=20, width=30, stretch=tk.NO)
        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.filelist_view.yview
        )
        hscrollbar = ttk.Scrollbar(
            self,
            orient="horizontal",
            command=self.filelist_view.xview
        )
        self.filelist_view.configure(yscrollcommand=scrollbar.set)
        self.filelist_view.configure(xscrollcommand=hscrollbar.set)
        self.filelist_view.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="nsw")
        hscrollbar.grid(row=1, column=0, sticky="new")
        ttk.Button(
            self, text="Remove", command=self.delete_selected
        ).grid(row=2, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            self, text="Cache all", command=self.cache_all
        ).grid(row=3, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            self, text="Export cached data", command=self.export_cached
        ).grid(row=4, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            self, text="Save cached graphs to png", command=self.export_graphs
        ).grid(row=5, column=0, columnspan=2, sticky="nsew")

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
                    result_list.append(
                        self.anal_res[file_index][sub_index][column_index].record
                    )
        if len(result_list) == 0:
            return
        result = pd.concat(result_list)
        result = result.reset_index(drop=True)
        result.to_csv(f"{self.path}{SEP}result.csv", index=False)

    def export_graphs(self) -> None:
        file_res_indeces = list(self.all_data.keys())
        file_res_indeces.sort()
        for file_index in file_res_indeces:
            sub_indeces = list(self.all_data[file_index].keys())
            sub_indeces.sort()
            name = self.namelist[file_index]
            data = self.all_data[file_index]
            assert len(data) > 0
            fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
            axes = fig.add_axes([0,0,1,1])
            axes.set_title(name, fontsize=16)
            if fet.data_type(data[0]) == "out":
                # output curve
               fet.draw_output(pd.concat(data.values()), axes)
            elif fet.data_type(data[0]) == "tr":
                # transfer curve
                axes2 = axes.twinx()
                fet.draw_transfer(pd.concat(data.values()), axes, axes2)
            pl.savefig(f"{self.path}{SEP}{name}.png", bbox_inches = 'tight')
            pl.close(fig)
            for sub_index in sub_indeces:
                sub_data = data[sub_index]
                if fet.data_type(sub_data) == "out" and len(data) > 1:
                    fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
                    axes = fig.add_axes([0,0,1,1])
                    axes.set_title(name, fontsize=16)
                    fet.draw_output(sub_data, axes)
                    pl.savefig(
                        f"{self.path}{SEP}{name}-sweep{sub_index+1}.png",
                        bbox_inches = 'tight'
                    )
                    pl.close(fig)
                if file_index not in self.anal_res or sub_index not in self.anal_res[file_index]:
                    continue
                column_indeces = list(self.anal_res[file_index][sub_index].keys())
                column_indeces.sort()
                for col_ind in column_indeces:
                    data = sub_data[[sub_data.columns[0], sub_data.columns[col_ind]]]
                    turn_on, fit_left, fit_right = (
                        self._analyze_transfer_curve(file_index, sub_index, col_ind)
                    )
                    result = fet.param_eval(
                        data,
                        on_index=turn_on,
                        fitting_boundaries=(fit_left, fit_right)
                    )
                    mob = fet.mobility_calc(result[0], ci, w, l)
                    fig = pl.figure(figsize=(7/2.54, 5/2.54), dpi=600)
                    axes = fig.add_axes([0,0,1,1])
                    axes2 = axes.twinx()
                    axes.set_title(name, fontsize=16)
                    fet.draw_analyzed_graph(
                        data,
                        axes,
                        axes2,
                        mob,
                        result,
                        fontsize=8,
                        label_font_s=14
                    )
                    pl.savefig(
                        f"{self.path}{SEP}{name}-calc{col_ind}.png", bbox_inches = 'tight'
                    )
                    pl.close(fig)

    def delete_selected(self) -> None:
        sel_num, subsel, colnum = self.get_selection_nums()
        if sel_num == -1:
            return
        self.event_generate("<<NewAnalysisResults>>", when="tail")
        if subsel == -1:
            self.filelist_view.delete(f"{sel_num}")
            self.filelist[sel_num] = None
            if sel_num in self.all_data:
                self.all_data.pop(sel_num)
            if sel_num in self.anal_param:
                self.anal_param.pop(sel_num)
                self.anal_res.pop(sel_num)
                self.recently_generated.append((sel_num, subsel, colnum))
            return
        if colnum == -1:
            self.filelist_view.delete(f"{sel_num}-{subsel}")
            self.filelist_view.selection_add(f"{sel_num}")
            if subsel in self.anal_param[sel_num]:
                self.anal_param[sel_num].pop(subsel)
                self.anal_res[sel_num].pop(subsel)
                self.recently_generated.append((sel_num, subsel, colnum))
            return
        self.filelist_view.delete(f"{sel_num}-{subsel}-{colnum}")
        self.filelist_view.selection_add(f"{sel_num}-{subsel}")
        assert colnum in self.anal_param[sel_num][subsel]
        self.anal_param[sel_num][subsel].pop(colnum)
        self.anal_res[sel_num][subsel].pop(colnum)
        self.recently_generated.append((sel_num, subsel, colnum))
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
        self.event_generate("<<NewAnalysisResults>>", when="tail")

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

    def _get_file_data(self, file_ind: int) -> dict[int, pd.DataFrame]:
        new_data: bool = False
        selected: Optional[str] = self.filelist[file_ind]
        if file_ind in self.all_data:
            data_storage = self.all_data[file_ind]
        else:
            assert selected is not None
            raw_data = fet.extract_data_from_file(selected)
            data_storage = {ind: data for ind, data in enumerate(raw_data)}
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
                for index, sub_data in list(data_storage.items()):
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
        data_storage: dict[int, pd.DataFrame] = self._get_file_data(sel_num)
        if subsel >= 0:
            book = data_storage[subsel]
        elif subsel == -1:
            if len(data_storage) > 0:
                book = pd.concat(data_storage.values())
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

    def _analyze_transfer_curve(self,
        index: int,
        sub_ind: int,
        col_ind: int
    ) -> tuple[int, int, int]:
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
        result: tuple[float, float, Optional[float], Optional[float], float, float]
    ) -> None:
        sel, subsel, col_ind = self.get_selection_nums()
        self._save_analysis_results(sel, subsel, col_ind, mob, result)
        self.event_generate("<<NewAnalysisResults>>", when="tail")

    def _save_analysis_results(
        self,
        index: int,
        subindex: int,
        col_ind: int,
        mob: float,
        result: tuple[float, float, Optional[float], Optional[float], float, float]
    ) -> None:
        desc: dict[str, str] = {"Name": self.namelist[index]}
        if len(self.all_data[index]) > 1:
            desc["Sweep"] = f"{subindex}"
        if len(self.all_data[index][subindex].columns) > 2:
            desc["VDS"] = self.all_data[index][subindex].columns[col_ind].replace("VDS = ", "")
        self.anal_res[index][subindex][col_ind] = fet.FetResult(desc, mob, *result)
        self.filelist_view.item(
            f"{index}-{subindex}-{col_ind}",
            values=("","✓")
        )
        self.recently_generated.append((index, subindex, col_ind))

    def save_analysis_params(self, parameters: AnalParams) -> None:
        sel, subsel, col = self.get_selection_nums()
        self.anal_param[sel][subsel][col] = parameters

    def get_changed_result(self) -> Optional[tuple[Optional[fet.FetResult], int, int, int]]:
        if len(self.recently_generated) == 0:
            return None
        file, sweep, col_num = self.recently_generated.pop(0)
        if (
            file not in self.anal_res or
            sweep not in self.anal_res[file] or
            col_num not in self.anal_res[file][sweep]
        ):
            return None, file, sweep, col_num
        return self.anal_res[file][sweep][col_num], file, sweep, col_num
