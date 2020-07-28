import threading
from tkinter import *
from tkinter import scrolledtext
from PIL import Image, ImageTk


class StoppableThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class HoverButton(Button):

    def __init__(self, master, image_path=None, keep_pressed=False, **kw):
        Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        if keep_pressed:
            self.bind("<Button-1>", self.on_click)
        if image_path:
            self.image = ImageTk.PhotoImage(Image.open(image_path))
            self.configure(image=self.image)

    def on_click(self, e):
        if self['background'] == self.defaultBackground:
            self['background'] = self['activebackground']
        else:
            self['background'] = self.defaultBackground

    def on_enter(self, e):
        self['background'] = self['activebackground']

    def on_leave(self, e):
        self['background'] = self.defaultBackground


class Logger(Frame):

    def __init__(self, master=None, **kw):
        Frame.__init__(self, master, **kw)
        self.text = scrolledtext.ScrolledText(self, height=15, bg='#002b36', fg='#eee8d5')
        self.text.pack(expand=True, padx=50)
        self.text.configure(state='disabled')

    def log(self, msg):
        self.text.configure(state='normal')
        self.text.insert(END, msg + '\n')
        self.text.configure(state='disabled')
        self.text.yview(END)
