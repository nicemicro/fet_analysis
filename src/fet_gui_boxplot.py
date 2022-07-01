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


class BoxplotCtrl(ttk.Frame):
    def __init__(self, master: ttk.Widget) -> None:
        ttk.Frame.__init__(self, master=master)
        ttk.Label(self, text="Data to plot:").grid(row=0, column=0)
        self.params_to_list: list[fet.Parameters] = [
            fet.Parameters.MOBILITY,
            fet.Parameters.VTH,
            fet.Parameters.VON,
            fet.Parameters.SS,
            fet.Parameters.ONOFF,
            fet.Parameters.ION,
            fet.Parameters.IOFF
        ]
        self._selected_param_name = tk.StringVar()
        self._selected_param_name.set(fet.PARAMNAMES[fet.Parameters.MOBILITY])
        ttk.OptionMenu(
            self,
            self._selected_param_name,
            fet.PARAMNAMES[fet.Parameters.MOBILITY],
            *list(fet.PARAMNAMES[param] for param in self.params_to_list)
        ).grid(row=0, column=1)
        self._save_to_file = tk.IntVar()
        self._save_to_file.set(0)
        ttk.Label(self, text="Export graph to file").grid(row=1, column=0)
        ttk.Checkbutton(
            self,
            variable=self._save_to_file,
            offvalue=0,
            onvalue=1
        ).grid(row=1, column=1, sticky="nsw")
        ttk.Button(
            self,
            text="Create Graph",
            command=lambda: self.event_generate("<<DrawBoxplot>>", when="tail")
        ).grid(row=10, column=0, columnspan=2, sticky="se")

    def graph_info(self) -> tuple[fet.Parameters, bool]:
        name = self._selected_param_name.get()
        index = list(fet.PARAMNAMES.values()).index(name)
        return (
            list(fet.PARAMNAMES.keys())[index],
            self._save_to_file.get() == 1
        )
