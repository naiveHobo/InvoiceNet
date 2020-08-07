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

from tkinter import *
from PIL import Image, ImageTk

from.custom_widgets import HoverButton


class HelpBox(Frame):

    def __init__(self, master=None, background='#303030', **kw):
        Frame.__init__(self, master, **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)

        Label(self, text="Meet InvoiceNet!", anchor='nw', width=100,
              font="OpenSans 22 bold", fg='white', bg=background, bd=2).grid(row=0, column=0, padx=20, pady=20)

        Label(self, text="Made with ❤ by naiveHobo", anchor='nw', width=100,
              font="OpenSans 10 bold", fg='white', bg=background, bd=2).grid(row=2, column=0, padx=20, pady=20)

        text_frame = Frame(self, height=440, width=550, bg=background, bd=2, relief=SUNKEN)
        text_frame.grid(row=1, column=0)

        text_frame.grid_propagate(False)

        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        text_box = Text(text_frame, borderwidth=3, relief="sunken", bg=background,
                        fg='white', font="OpenSans 12", wrap='word')

        with open(r'widgets/help.txt', 'r') as infile:
            texts = infile.read()

        texts = [text.strip() + '\n\n\n' for text in texts.split('---')]

        text_box.insert('1.0', texts[0])
        texts = texts[1:]

        paths = ['open_file.png', 'open_dir.png', 'save_as.png', 'clear_page.png',
                 'search.png', 'extract.png', 'ocr.png', 'clear_all.png']
        self.images = [ImageTk.PhotoImage(Image.open(r'widgets/' + path)) for path in paths]

        for text, image in zip(texts, self.images):
            text_box.image_create(END, image=image)
            text_box.insert(END, ' ' + text)

        self.images.extend([ImageTk.PhotoImage(Image.open(r'widgets/' + path))
                            for path in ['prev_file.png', 'next_file.png']])

        text_box.image_create(END, image=self.images[-2])
        text_box.image_create(END, image=self.images[-1])
        text_box.insert(END, ' ' + texts[-5])

        viewer_text = texts[-4].split('\n\n')
        text_box.insert(END, viewer_text[0] + '\n')
        self.images.append(ImageTk.PhotoImage(Image.open(r'widgets/toolbar.png')))
        text_box.image_create(END, image=self.images[-1])
        text_box.insert(END, '\n\n' + '\n\n'.join(viewer_text[1:]))

        for text, image in zip(texts[-3:], ['begin.png', 'labels.png', 'labels.png']):
            splits = text.strip().split('\n\n')
            btn = HoverButton(text_box, image_path=r'widgets/' + image, text=splits[0],
                              compound='center', font=("Arial", 10, "bold"), bd=0, bg=background,
                              highlightthickness=0, activebackground=background)
            text_box.window_create(END, window=btn)
            text_box.insert(END, '\n\n' + '\n\n'.join(splits[1:]) + '\n\n\n')

        text_box.config(state=DISABLED)
        text_box.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        scroll_bar = Scrollbar(text_frame, command=text_box.yview, bg=background)
        scroll_bar.grid(row=0, column=1, sticky='nsew')

        text_box['yscrollcommand'] = scroll_bar.set
