# Copyright (c) 2020 Sarthak Mittal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import glob
import random
import pdf2image
import simplejson
import numpy as np
from tqdm import tqdm
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk

from .. import FIELDS, FIELD_TYPES
from ..common import util
from ..acp.acp import AttendCopyParse
from ..acp.data import InvoiceData
from .custom_widgets import HoverButton, Logger, StoppableThread


class Trainer(Frame):

    def __init__(self, master=None, **kw):
        Frame.__init__(self, master, **kw)
        self.background = '#303030'
        self.border_color = '#404040'
        self.args = {
            "data_dir": "",
            "prepared_data": "processed_data",
            "field": list(FIELDS.keys())[0],
            "batch_size": 4
        }
        self.textboxes = {}
        self.thread = None
        self.running = False
        self._init_ui()

    def _init_ui(self):
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        h = hs - 100
        w = int(h / 1.414) + 100
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.master.maxsize(w, h)
        self.master.minsize(w, h)
        self.master.title("InvoiceNet - Trainer")

        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)

        self.configure(bg=self.background, bd=0)

        logo_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN,
                           highlightbackground=self.border_color, highlightthickness=1)
        param_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN,
                            highlightbackground=self.border_color, highlightthickness=1)
        progress_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN,
                               highlightbackground=self.border_color, highlightthickness=0)
        main_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN,
                           highlightbackground=self.border_color, highlightthickness=1)

        logo_frame.grid(row=0, column=0, sticky='news')
        param_frame.grid(row=1, column=0, sticky='news')
        progress_frame.grid(row=2, column=0, sticky='news', padx=50, pady=(0, 20))
        main_frame.grid(row=3, column=0, sticky='news')

        # Logo Frame
        logo_frame.columnconfigure(0, weight=1)
        logo_frame.columnconfigure(1, weight=0)
        logo_frame.columnconfigure(2, weight=0)
        logo_frame.columnconfigure(3, weight=1)
        logo_frame.rowconfigure(0, weight=1)

        self.logo_img = ImageTk.PhotoImage(Image.open(r'widgets/logo.png'))
        Label(logo_frame, bg=self.background, image=self.logo_img).grid(row=0, column=1, sticky='news', pady=10)
        Label(logo_frame, text="InvoiceNet", bg=self.background,
              fg="white", font=("Arial", 24, "bold")).grid(row=0, column=2, sticky='news', padx=20, pady=10)

        # Param Frame
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=0)
        param_frame.columnconfigure(2, weight=0)
        param_frame.columnconfigure(3, weight=1)
        param_frame.rowconfigure(0, weight=1)
        param_frame.rowconfigure(1, weight=0)
        param_frame.rowconfigure(2, weight=0)
        param_frame.rowconfigure(3, weight=0)
        param_frame.rowconfigure(4, weight=1)

        data_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN,
                           highlightbackground=self.border_color, highlightthickness=0)
        out_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN,
                          highlightbackground=self.border_color, highlightthickness=0)
        field_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN,
                            highlightbackground=self.border_color, highlightthickness=0)
        batch_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN,
                            highlightbackground=self.border_color, highlightthickness=0)

        data_param.grid(row=1, column=1, pady=(0, 20), padx=20)
        out_param.grid(row=2, column=1, pady=20, padx=20)
        field_param.grid(row=1, column=2, pady=(0, 20), padx=20)
        batch_param.grid(row=2, column=2, pady=20, padx=20)

        df = Frame(data_param, bg=self.background, bd=0, relief=SUNKEN,
                   highlightbackground=self.border_color, highlightthickness=0)
        df.pack(side=TOP, fill=BOTH)

        Label(df, text="Data Folder:", bg=self.background,
              fg="white", font=("Arial", 8, "bold"), anchor='w').pack(side=LEFT, fill=BOTH)
        HoverButton(df, image_path=r'widgets/open_dir_small.png', command=lambda: self._open_dir("data_dir"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["data_dir"] = Text(data_param, height=1, width=20)
        self.textboxes["data_dir"].insert('1.0', self.args["data_dir"])
        self.textboxes["data_dir"].pack(side=BOTTOM)

        of = Frame(out_param, bg=self.background, bd=0, relief=SUNKEN,
                   highlightbackground=self.border_color, highlightthickness=0)
        of.pack(side=TOP, fill=BOTH)

        Label(of, text="Processed Data Folder:", bg=self.background,
              anchor='w', fg="white", font=("Arial", 8, "bold")).pack(side=LEFT, fill=BOTH)
        HoverButton(of, image_path=r'widgets/open_dir_small.png', command=lambda: self._open_dir("prepared_data"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["prepared_data"] = Text(out_param, height=1, width=20)
        self.textboxes["prepared_data"].insert('1.0', self.args["prepared_data"])
        self.textboxes["prepared_data"].pack(side=BOTTOM)

        Label(field_param, text="Field:", bg=self.background,
              anchor='w', fg="white", font=("Arial", 8, "bold")).pack(side=TOP, fill=BOTH)
        self.field_text = StringVar(field_param)
        self.field_text.set(list(FIELDS.keys())[0])

        keys = list(FIELDS.keys())
        field_list = OptionMenu(field_param, self.field_text, *keys)
        field_list.configure(highlightthickness=0, width=20, bg='#ffffff')
        field_list.pack(side=BOTTOM)

        for key in keys:
            field_list['menu'].entryconfigure(key, state="normal")

        Label(batch_param, text="Batch Size:", bg=self.background,
              anchor='w', fg="white", font=("Arial", 8, "bold")).pack(side=TOP, fill=BOTH)
        self.batch_text = StringVar(batch_param)
        self.batch_text.set("4")
        batch_list = OptionMenu(batch_param, self.batch_text, *[str(2 ** i) for i in range(8)])
        batch_list.configure(highlightthickness=0, width=20, bg='#ffffff')
        batch_list.pack(side=BOTTOM)

        HoverButton(param_frame, image_path=r'widgets/prepare.png', command=self._prepare_data,
                    text='Prepare Data', compound='center', font=("Arial", 10, "bold"), bg=self.background,
                    bd=0, highlightthickness=0, activebackground=self.background).grid(row=3, column=1, columnspan=2,
                                                                                       padx=20, pady=(20, 0),
                                                                                       sticky='news')

        # Progress Frame
        self.progress_label = Label(progress_frame, text="Preparing data:", bg=self.background,
                                    anchor='w', fg="white", font=("Arial", 8, "bold"), bd=0, highlightthickness=0)
        self.progress_label.pack(side=TOP, expand=True, fill=X, pady=(10, 5))
        self.progressbar = Progressbar(progress_frame, orient=HORIZONTAL, length=100, mode='determinate')
        self.progressbar.pack(side=BOTTOM, expand=True, fill=X)

        # Main Frame
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        button_frame = Frame(main_frame, bg=self.background, bd=0, relief=SUNKEN,
                             highlightbackground=self.border_color, highlightthickness=0)
        button_frame.grid(row=0, column=0, sticky='news')

        button_frame.rowconfigure(0, weight=1)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=0)
        button_frame.columnconfigure(2, weight=1)

        self.start_button = HoverButton(button_frame, image_path=r'widgets/begin.png', command=self._start,
                                        text='Start', compound='center', font=("Arial", 10, "bold"), bg=self.background,
                                        bd=0, highlightthickness=0, activebackground=self.background)
        self.stop_button = HoverButton(button_frame, image_path=r'widgets/stop.png', command=self._stop,
                                       text='Stop', compound='center', font=("Arial", 10, "bold"), bg=self.background,
                                       bd=0, highlightthickness=0, activebackground=self.background)

        self.start_button.grid(row=0, column=1)
        self.stop_button.grid(row=0, column=1)
        self.stop_button.grid_forget()

        self.logger = Logger(main_frame, height=18, bg=self.background, bd=0, relief=SUNKEN)
        self.logger.grid(row=1, column=0, sticky='news')

    def _train(self):
        train_data = InvoiceData.create_dataset(
            field=self.args["field"],
            data_dir=os.path.join(self.args["prepared_data"], 'train/'),
            batch_size=self.args["batch_size"]
        )
        val_data = InvoiceData.create_dataset(
            field=self.args["field"],
            data_dir=os.path.join(self.args["prepared_data"], 'val/'),
            batch_size=self.args["batch_size"]
        )

        restore = None
        if os.path.exists(os.path.join('./models/invoicenet/', self.args["field"])):
            restore = messagebox.askyesno(
                title="Restore",
                message="A checkpoint was found! Do you want to restore checkpoint for training?")

        restore = True if restore else False

        model = AttendCopyParse(field=self.args["field"], restore=restore)

        print_interval = 20
        early_stop_steps = 0
        best = float("inf")

        train_iter = iter(train_data)
        val_iter = iter(val_data)

        self.logger.log("Initializing training!")
        start = time.time()
        step = 0

        while True:

            try:
                train_loss = model.train_step(next(train_iter))
            except StopIteration:
                self.logger.log("Couldn't find any training data! Have you prepared your training data?")
                self.logger.log("Terminating...")
                self.thread.stop()
                break

            if not np.isfinite(train_loss):
                raise ValueError("NaN loss")

            if step % print_interval == 0:
                took = time.time() - start

                try:
                    val_loss = model.val_step(next(val_iter))
                except StopIteration:
                    self.logger.log("Couldn't find any validation data! Have you prepared your training data?")
                    self.logger.log("Terminating...")
                    self.thread.stop()
                    break

                self.logger.log("[step: %d | %.2f steps/s]: train loss: %.4f val loss: %.4f" % (
                    step, (step + 1) / took, train_loss, val_loss))
                if not np.isfinite(val_loss):
                    self.logger.log("ERROR: NaN loss")
                    self.thread.stop()
                if val_loss < best:
                    early_stop_steps = 0
                    best = val_loss
                    model.save("best")
                else:
                    early_stop_steps += print_interval
                    if early_stop_steps >= 500:
                        self.logger.log("Validation loss has not improved for 500 steps")
                        self.thread.stop()

            step += 1

            if self.thread.stopped():
                self.logger.log("Training terminated!")
                break

        self.running = False
        self.stop_button.grid_forget()
        self.start_button.grid(row=0, column=1)

    def _get_inputs(self):
        self.args["field"] = self.field_text.get()
        self.args["batch_size"] = int(self.batch_text.get())
        self.args["data_dir"] = self.textboxes["data_dir"].get("1.0", 'end-1c')
        self.args["prepared_data"] = self.textboxes["prepared_data"].get("1.0", 'end-1c')
        if not self.args["prepared_data"].endswith('/'):
            self.args["prepared_data"] += '/'
        if self.args["data_dir"] == '':
            return
        if not self.args["data_dir"].endswith('/'):
            self.args["data_dir"] += '/'

    def _start(self):
        self._get_inputs()

        if not os.path.exists(self.args["prepared_data"]):
            messagebox.showerror("Error", "Prepared data folder does not exist!")
            return

        files = glob.glob(self.args["prepared_data"] + "**/*.json", recursive=True)
        if not files:
            messagebox.showerror("Error",
                                 "Could not find processed data in \"{}\". Did you prepare training data?".format(
                                     self.args["prepared_data"]))
            return
        if not self.running:
            self.running = True
            self.thread = StoppableThread(target=self._train)
            self.thread.daemon = True
            self.thread.start()
            self.start_button.grid_forget()
            self.stop_button.grid(row=0, column=1)

    def _stop(self):
        if self.running:
            self.thread.stop()
            self.running = False
            self.logger.log("Stopping training...")

    def _open_dir(self, key):
        dir_name = filedialog.askdirectory(initialdir='.', title="Select Directory Containing Invoices")
        if not dir_name:
            return
        self.args[key] = dir_name
        self.textboxes[key].delete('1.0', END)
        self.textboxes[key].insert('1.0', self.args[key])

    def _prepare_data(self):
        self._get_inputs()

        if self.args["data_dir"] == '':
            messagebox.showerror("Error", "Data folder does not exist!")
            return

        if not os.path.exists(self.args["data_dir"]):
            messagebox.showerror("Error", "Data folder does not exist!")
            return

        self.progressbar["value"] = 0
        self.progress_label.configure(text="Preparing Data:")

        os.makedirs(os.path.join(self.args["prepared_data"], 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.args["prepared_data"], 'val'), exist_ok=True)

        filenames = [os.path.abspath(f) for f in glob.glob(self.args["data_dir"] + "**/*.pdf", recursive=True)]
        random.shuffle(filenames)

        idx = int(len(filenames) * 0.2)
        train_files = filenames[idx:]
        val_files = filenames[:idx]

        self.logger.log("Total: {}".format(len(filenames)))
        self.logger.log("Training: {}".format(len(train_files)))
        self.logger.log("Validation: {}".format(len(val_files)))

        if len(train_files) == 0 or len(val_files) == 0:
            messagebox.showwarning("Warning", "Training data is not enough to create training/validation splits")

        total_samples = len(filenames)
        sample_idx = 0
        for phase, filenames in [('train', train_files), ('val', val_files)]:
            self.logger.log("Preparing {} data...".format(phase))
            for filename in tqdm(filenames):
                # try:
                page = pdf2image.convert_from_path(filename)[0]
                page.save(os.path.join(self.args["prepared_data"], phase, os.path.basename(filename)[:-3] + 'png'))

                height = page.size[1]
                width = page.size[0]

                ngrams = util.create_ngrams(page, height=height, width=width)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

                with open(filename[:-3] + 'json', 'r') as fp:
                    labels = simplejson.loads(fp.read())

                fields = {}
                for field in FIELDS:
                    if field in labels:
                        if FIELDS[field] == FIELD_TYPES["amount"]:
                            fields[field] = util.normalize(labels[field], key="amount")
                        elif FIELDS[field] == FIELD_TYPES["date"]:
                            fields[field] = util.normalize(labels[field], key="date")
                        else:
                            fields[field] = labels[field]
                    else:
                        fields[field] = ''

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.abspath(
                        os.path.join(self.args["prepared_data"], phase, os.path.basename(filename)[:-3] + 'png'))
                }

                with open(os.path.join(self.args["prepared_data"], phase, os.path.basename(filename)[:-3] + 'json'),
                          'w') as fp:
                    fp.write(simplejson.dumps(data, indent=2))

                # except Exception as exp:
                #     self.logger.log("Skipping {} : {}".format(filename, exp))

                sample_idx += 1
                self.progress_label.configure(text="Preparing data [{}/{}]:".format(sample_idx, total_samples))
                self.progressbar["value"] = (sample_idx / total_samples) * 100
                self.progressbar.update()

        self.progress_label.configure(text="Completed!")
        self.progressbar["value"] = 100
        self.progressbar.update()
        self.logger.log("Prepared data stored in '{}'".format(self.args["prepared_data"]))
