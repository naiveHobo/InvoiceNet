import os
import glob
import pdf2image
import simplejson
from tqdm import tqdm
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar

from invoicenet import FIELDS
from invoicenet.common import util
from invoicenet.acp.acp import AttendCopyParse
from invoicenet.acp.data import RealData
from invoicenet.common.custom_widgets import *


class Extractor(Frame):

    def __init__(self, master=None, **kw):
        Frame.__init__(self, master, **kw)
        self.background = '#303030'
        self.args = {
            "data_dir": "",
            "data_file": "",
            "pred_dir": "predictions",
            "prepared_data": "processed_data",
        }
        self.textboxes = {}
        self.checkboxes = {}
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
        self.master.title("InvoiceNet - Extractor")

        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)

        self.configure(bg=self.background, bd=0)

        title_frame = Frame(self, height=100, bg=self.background, bd=1, relief=SUNKEN)
        param_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN)
        progress_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN)
        main_frame = Frame(self, bg=self.background, bd=0, relief=SUNKEN)

        title_frame.grid(row=0, column=0, sticky='news')
        param_frame.grid(row=1, column=0, sticky='news')
        progress_frame.grid(row=2, column=0, sticky='news', padx=50, pady=(0, 20))
        main_frame.grid(row=3, column=0, sticky='news')

        # Title Frame
        title_frame.columnconfigure(0, weight=1)
        title_frame.rowconfigure(0, weight=1)
        title_label = Label(title_frame, text="Extractor", bg=self.background, fg="white", font="Arial 24")
        title_label.grid(row=0, column=0, sticky='nws', padx=10, pady=5)

        # Param Frame
        param_frame.columnconfigure(0, weight=1)
        param_frame.columnconfigure(1, weight=0)
        param_frame.columnconfigure(2, weight=0)
        param_frame.columnconfigure(3, weight=1)
        param_frame.rowconfigure(0, weight=1)
        param_frame.rowconfigure(1, weight=0)
        param_frame.rowconfigure(2, weight=0)
        param_frame.rowconfigure(3, weight=0)
        param_frame.rowconfigure(4, weight=0)
        param_frame.rowconfigure(5, weight=0)
        param_frame.rowconfigure(6, weight=1)

        data_file_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN)
        data_dir_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN)
        prepare_dir_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN)
        pred_dir_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN)
        field_param = Frame(param_frame, bg=self.background, bd=0, relief=SUNKEN)

        data_file_param.grid(row=1, column=1, pady=10, padx=20)
        data_dir_param.grid(row=2, column=1, pady=10, padx=20)
        prepare_dir_param.grid(row=3, column=1, pady=10, padx=20)
        pred_dir_param.grid(row=4, column=1, pady=10, padx=20)
        field_param.grid(row=1, column=2, rowspan=4, pady=10, padx=10, sticky='news')

        # Invoice File
        data_file_frame = Frame(data_file_param, bg=self.background, bd=0, relief=SUNKEN)
        data_file_frame.pack(side=TOP, fill=BOTH)

        Label(data_file_frame, text="Invoice File:", bg=self.background,
              fg="white", font="Arial 8", anchor='w').pack(side=LEFT, fill=BOTH)
        HoverButton(data_file_frame, image_path=r'widgets/open_file.png',
                    command=lambda: self._open("data_file"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["data_file"] = Text(data_file_param, height=1, width=20)
        self.textboxes["data_file"].insert('1.0', self.args["data_file"])
        self.textboxes["data_file"].pack(side=BOTTOM)

        # Invoice Directory
        data_dir_frame = Frame(data_dir_param, bg=self.background, bd=0, relief=SUNKEN)
        data_dir_frame.pack(side=TOP, fill=BOTH)

        Label(data_dir_frame, text="Invoice Folder:", bg=self.background,
              anchor='w', fg="white", font="Arial 8").pack(side=LEFT, fill=BOTH)
        HoverButton(data_dir_frame, image_path=r'widgets/open_dir.png',
                    command=lambda: self._open("data_dir"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["data_dir"] = Text(data_dir_param, height=1, width=20)
        self.textboxes["data_dir"].insert('1.0', self.args["data_dir"])
        self.textboxes["data_dir"].pack(side=BOTTOM)

        # Prepared Data Directory
        prepare_dir_frame = Frame(prepare_dir_param, bg=self.background, bd=0, relief=SUNKEN)
        prepare_dir_frame.pack(side=TOP, fill=BOTH)

        Label(prepare_dir_frame, text="Processed Data Folder:", bg=self.background,
              anchor='w', fg="white", font="Arial 8").pack(side=LEFT, fill=BOTH)
        HoverButton(prepare_dir_frame, image_path=r'widgets/open_dir.png',
                    command=lambda: self._open("prepared_data"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["prepared_data"] = Text(prepare_dir_param, height=1, width=20)
        self.textboxes["prepared_data"].insert('1.0', self.args["prepared_data"])
        self.textboxes["prepared_data"].pack(side=BOTTOM)

        # Prediction Directory
        pred_dir_frame = Frame(pred_dir_param, bg=self.background, bd=0, relief=SUNKEN)
        pred_dir_frame.pack(side=TOP, fill=BOTH)

        Label(pred_dir_frame, text="Prediction Folder:", bg=self.background,
              anchor='w', fg="white", font="Arial 8").pack(side=LEFT, fill=BOTH)
        HoverButton(pred_dir_frame, image_path=r'widgets/open_dir.png',
                    command=lambda: self._open("pred_dir"),
                    width=18, height=18, bg=self.background, bd=0,
                    highlightthickness=0, activebackground='#558de8').pack(side=RIGHT)

        self.textboxes["pred_dir"] = Text(pred_dir_param, height=1, width=20)
        self.textboxes["pred_dir"].insert('1.0', self.args["pred_dir"])
        self.textboxes["pred_dir"].pack(side=BOTTOM)

        # Field Checkboxes
        field_frame = Frame(field_param, bg='#353535', bd=1, relief=SUNKEN)
        field_frame.pack(expand=True, fill=BOTH, pady=10)

        Label(field_frame, text="Field:", width=30, bg='#353535',
              anchor='w', fg="white", font="Arial 8").pack(side=TOP, fill=X, padx=5, pady=5)

        checkbox_frame = Frame(field_frame, bg='#353535', bd=1, relief=SUNKEN)
        checkbox_frame.pack(expand=True, fill=BOTH, side=BOTTOM)

        checkbox_frame.columnconfigure(0, weight=1)
        checkbox_frame.columnconfigure(1, weight=1)
        checkbox_frame.columnconfigure(2, weight=1)
        checkbox_frame.columnconfigure(3, weight=1)
        for i in range(len(FIELDS) // 2):
            checkbox_frame.rowconfigure(i, weight=1)
        for idx, key in enumerate(FIELDS):
            self.checkboxes[key] = BooleanVar(checkbox_frame, value=False)
            state = False
            if os.path.exists('./models/invoicenet/'):
                state = key in os.listdir('./models/invoicenet/')

            Checkbutton(checkbox_frame, fg="black", bg='#353535',
                        activebackground=self.background, variable=self.checkboxes[key], anchor='w',
                        state="normal" if state else "disabled", highlightthickness=0).grid(row=idx // 2,
                                                                                            column=2 if idx % 2 else 0,
                                                                                            sticky='news', padx=(10, 0))
            Label(checkbox_frame, text=key, bg='#353535',
                  anchor='w', fg="white", font="Arial 8").grid(row=idx // 2, column=3 if idx % 2 else 1, sticky='news')

        # Prepare Data Button
        HoverButton(param_frame, image_path=r'widgets/prepare.png', command=self._prepare_data,
                    text='Prepare Data', compound='center', font='Arial 10 bold', bg=self.background,
                    bd=0, highlightthickness=0, activebackground=self.background).grid(row=5, column=1, columnspan=2,
                                                                                       padx=20, pady=(20, 0),
                                                                                       sticky='news')

        # Progress Frame
        self.progress_label = Label(progress_frame, text="Preparing data:", bg=self.background,
                                    anchor='w', fg="white", font="Arial 8")
        self.progress_label.pack(side=TOP, expand=True, fill=X)
        self.progressbar = Progressbar(progress_frame, orient=HORIZONTAL, length=100, mode='determinate')
        self.progressbar.pack(side=BOTTOM, expand=True, fill=X)

        # Main Frame
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        button_frame = Frame(main_frame, bg=self.background, bd=0, relief=SUNKEN)
        button_frame.grid(row=0, column=0, sticky='news')

        button_frame.rowconfigure(0, weight=1)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=0)
        button_frame.columnconfigure(2, weight=1)

        self.start_button = HoverButton(button_frame, image_path=r'widgets/begin.png', command=self._start,
                                        text='Extract', compound='center', font='Arial 10 bold', bg=self.background,
                                        bd=0, highlightthickness=0, activebackground=self.background)
        self.start_button.grid(row=0, column=1)

        self.logger = Logger(main_frame, bg=self.background, bd=0, relief=SUNKEN)
        self.logger.grid(row=1, column=0, sticky='news')

    def _extract(self):
        self.logger.log("Extracting information from invoices...\n")

        predictions = {}
        for key in FIELDS:
            if self.checkboxes[key].get():
                test_data = RealData(field=key, data_dir=os.path.join(self.args["prepared_data"], 'predict/'))
                model = AttendCopyParse(field=key, test_data=test_data, batch_size=1, restore=True)
                preds = model.test_set(out_path=self.args["pred_dir"])
                for file in preds.keys():
                    if file in predictions:
                        predictions[file][key] = preds[file][key]
                    else:
                        predictions[file] = preds[file]

        for file in predictions:
            self.logger.log("Invoice: {}".format('.'.join([file.split('.')[0], 'pdf'])))
            for key in predictions[file]:
                self.logger.log("  - {}: {}".format(key, predictions[file][key]))
            self.logger.log(" ")

        self.logger.log("Extracted information stored in '{}'".format(self.args["pred_dir"]))
        self.start_button.configure(state='normal')

    def _get_inputs(self):
        self.args["data_file"] = self.textboxes["data_file"].get("1.0", 'end-1c')
        self.args["data_dir"] = self.textboxes["data_dir"].get("1.0", 'end-1c')
        self.args["prepared_data"] = self.textboxes["prepared_data"].get("1.0", 'end-1c')
        self.args["pred_dir"] = self.textboxes["pred_dir"].get("1.0", 'end-1c')
        if not self.args["data_dir"].endswith('/'):
            self.args["data_dir"] += '/'
        if not self.args["prepared_data"].endswith('/'):
            self.args["prepared_data"] += '/'
        if not self.args["pred_dir"].endswith('/'):
            self.args["pred_dir"] += '/'

    def _start(self):
        self._get_inputs()

        if not os.path.exists(self.args["prepared_data"]):
            messagebox.showerror("Error", "Prepared data folder does not exist!")
            return

        files = glob.glob(self.args["prepared_data"] + "**/*.json", recursive=True)
        if not files:
            messagebox.showerror("Error",
                                 "Could not find processed data in \"{}\". Did you prepare data for extraction?".format(
                                     self.args["prepared_data"]))
            return

        selected = False
        for key in FIELDS:
            if self.checkboxes[key].get():
                selected = True
                break

        if not selected:
            messagebox.showerror("Error", "No fields were selected!")
            return

        if not self.running:
            self.running = True
            self.thread = StoppableThread(target=self._extract)
            self.thread.daemon = True
            self.thread.start()
            self.start_button.configure(state='disabled')

    def _open(self, key):
        if key == 'data_file':
            path = filedialog.askopenfilename(initialdir='.', title="Select Invoice File",
                                              filetypes=[("PDF files", "*.pdf")])
        else:
            path = filedialog.askdirectory(initialdir='.', title="Select Directory Containing Invoices")
        if not path:
            return
        self.args[key] = path
        self.textboxes[key].delete('1.0', END)
        self.textboxes[key].insert('1.0', self.args[key])

    def _prepare_data(self):
        self._get_inputs()

        if not (os.path.exists(self.args["data_dir"]) or self.args["data_file"]):
            messagebox.showerror("Error", "No files were selected!")
            return

        self.progressbar["value"] = 0
        self.progress_label.configure(text="Preparing Data:")

        data_dir = os.path.join(self.args["prepared_data"], 'predict')
        os.makedirs(data_dir, exist_ok=True)

        filenames = [os.path.abspath(f) for f in glob.glob(data_dir + "**/*.json", recursive=True)]
        filenames += [os.path.abspath(f) for f in glob.glob(data_dir + "**/*.png", recursive=True)]
        for f in filenames:
            os.remove(f)

        filenames = []
        if self.args["data_dir"] and os.path.exists(self.args["data_dir"]):
            filenames = [os.path.abspath(f) for f in glob.glob(self.args["data_dir"] + "**/*.pdf", recursive=True)]
        if self.args["data_file"] and os.path.exists(self.args["data_file"]):
            filenames += [self.args["data_file"]]

        self.logger.log("Total: {}".format(len(filenames)))
        self.logger.log("Preparing data for extraction...")

        total_samples = len(filenames)
        sample_idx = 0
        for filename in tqdm(filenames):
            try:
                page = pdf2image.convert_from_path(filename)[0]
                page.save(os.path.join(data_dir, os.path.basename(filename)[:-3] + 'png'))

                height = page.size[1]
                width = page.size[0]

                ngrams = util.create_ngrams(page)
                for ngram in ngrams:
                    if "amount" in ngram["parses"]:
                        ngram["parses"]["amount"] = util.normalize(ngram["parses"]["amount"], key="amount")
                    if "date" in ngram["parses"]:
                        ngram["parses"]["date"] = util.normalize(ngram["parses"]["date"], key="date")

                fields = {field: '0' for field in FIELDS}

                data = {
                    "fields": fields,
                    "nGrams": ngrams,
                    "height": height,
                    "width": width,
                    "filename": os.path.abspath(os.path.join(data_dir, os.path.basename(filename)[:-3] + 'png'))
                }

                with open(os.path.join(data_dir, os.path.basename(filename)[:-3] + 'json'),
                          'w') as fp:
                    fp.write(simplejson.dumps(data, indent=2))

            except Exception as exp:
                self.logger.log("Skipping {} : {}".format(filename, exp))

            sample_idx += 1
            self.progress_label.configure(text="Preparing data [{}/{}]:".format(sample_idx, total_samples))
            self.progressbar["value"] = (sample_idx / total_samples) * 100
            self.progressbar.update()

        self.progress_label.configure(text="Completed!")
        self.progressbar["value"] = 100
        self.progressbar.update()
        self.logger.log("Prepared data stored in '{}'".format(data_dir))


def main():
    root = Tk()
    Extractor(root)
    root.mainloop()


if __name__ == '__main__':
    main()
