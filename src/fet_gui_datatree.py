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
import matplotlib.pyplot as pl
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import fet_func as fet


class DataTree(ttk.Frame):
    def __init__(self, master: ttk.Widget) -> None:
        ttk.Frame.__init__(self, master=master)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.folders: list[str] = []
        self.data_list: list[list[tuple[int, int, int]]] = []

        colnames = ("mob", "von", "ss", "vth", "onoff")
        self.alldata_view = ttk.Treeview(
            self,
            columns=colnames
        )
        self.alldata_view.heading("#0", text="Name")
        self.alldata_view.heading("mob", text="Mobility")
        self.alldata_view.heading("von", text="VOn")
        self.alldata_view.heading("ss", text="Ss")
        self.alldata_view.heading("vth", text="VTh")
        self.alldata_view.heading("onoff", text="I(On/Off)")
        self.alldata_view.column("#0", minwidth=100, width=200)
        for col_name in colnames:
            self.alldata_view.column(col_name, minwidth=50, width=60)
        self.alldata_view.grid(column=0, row=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.alldata_view.yview
        )
        hscrollbar = ttk.Scrollbar(
            self,
            orient="horizontal",
            command=self.alldata_view.xview
        )
        self.alldata_view.configure(yscrollcommand=scrollbar.set)
        self.alldata_view.configure(xscrollcommand=hscrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="nsw")
        hscrollbar.grid(row=1, column=0, sticky="new")

    def add_data(
        self,
        res: fet.FetResult,
        foldername: str,
        filenum: int,
        sweepnum: int,
        col_num: int
    ) -> None:
        foldernum: int = self.folders.index(foldername)
        visiblename: str = ""
        if isinstance(res.name, str):
            visiblename = res.name
        else:
            for thing in res.name.values():
                if len(visiblename) == 0:
                    visiblename = thing
                    continue
                visiblename += f" - {thing}"
        if (filenum, sweepnum, col_num) in self.data_list[foldernum]:
            self.alldata_view.item(
                f"{foldernum}-{filenum}-{sweepnum}-{col_num}",
                values=(f"{res.mob}", f"{res.v_on}", f"{res.ss}", f"{res.v_th}", f"{res.onoff}")
            )
            return
        self.data_list[foldernum].append((filenum, sweepnum, col_num))
        self.data_list[foldernum].sort()
        self.alldata_view.insert(
            foldername,
            index=self.data_list[foldernum].index((filenum, sweepnum, col_num)),
            iid=f"{foldernum}-{filenum}-{sweepnum}-{col_num}",
            text=visiblename,
            values=(f"{res.mob}", f"{res.v_on}", f"{res.ss}", f"{res.v_th}", f"{res.onoff}")
        )
        self.alldata_view.see(f"{foldernum}-{filenum}-{sweepnum}-{col_num}")

    def del_data(
        self,
        foldername: str,
        filenum: int,
        sweepnum: int,
        col_num: int
    ) -> None:
        foldernum: int = self.folders.index(foldername)
        if col_num != -1:
            if (filenum, sweepnum, col_num) in self.data_list[foldernum]:
                index = self.data_list[foldernum].index((filenum, sweepnum, col_num))
                self.data_list[foldernum].pop(index)
                self.alldata_view.delete(f"{foldernum}-{filenum}-{sweepnum}-{col_num}")
                return
        indexes_to_pop: list[int] = []
        for index, identifiers in enumerate(self.data_list[foldernum]):
            if identifiers[0] != filenum:
                continue
            if identifiers[1] != sweepnum and sweepnum != -1:
                continue
            indexes_to_pop = [index] + indexes_to_pop
            self.alldata_view.delete(f"{foldernum}-{identifiers[0]}-{identifiers[1]}-{identifiers[2]}")
        for to_pop in indexes_to_pop:
            self.data_list[foldernum].pop(to_pop)

    def add_folder(self, foldername: str) -> None:
        self.alldata_view.insert(
            "",
            "end",
            iid=foldername,
            text=foldername
        )
        self.folders.append(foldername)
        self.data_list.append([])
