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


class ToolTip:

    def __init__(self, widget, text):
        self.waittime = 500
        self.wraplength = 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.idx = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.idx = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        idx = self.idx
        self.idx = None
        if idx:
            self.widget.after_cancel(idx)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                      background="#ffffff", relief='solid', borderwidth=1,
                      wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()


class MenuBox(Frame):

    def __init__(self, master=None, image_path=None, background='#303030', highlight='#558de8', **kw):
        Frame.__init__(self, master, **kw)

        self.menu_button = Menubutton(self, width=50, height=50, bg=background, bd=0,
                                      highlightthickness=0, activebackground=highlight)

        if image_path:
            self.image = ImageTk.PhotoImage(Image.open(image_path))
            self.menu_button.configure(image=self.image)

        self.menu = Menu(self.menu_button, tearoff=False, bg=background,
                         fg='white', bd=2, activebackground=highlight)

        self.menu_button.config(menu=self.menu)
        self.menu_button.pack(side=LEFT)

        self.menu_button.bind("<Button-1>", lambda e: self.menu_button.event_generate('<<Invoke>>'))

    def add_item(self, title, func, seperator=False):
        self.menu.add_command(label=title, command=func)
        if seperator:
            self.menu.add_separator()


class HoverButton(Button):

    def __init__(self, master, tool_tip=None, image_path=None, **kw):
        Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        if image_path:
            self.image = ImageTk.PhotoImage(Image.open(image_path))
            self.configure(image=self.image)
        if tool_tip:
            ToolTip(self, text=tool_tip)

    def on_click(self, e):
        if self['background'] == self.defaultBackground:
            self['background'] = self['activebackground']
        else:
            self['background'] = self.defaultBackground

    def on_enter(self, e):
        self['background'] = self['activebackground']

    def on_leave(self, e):
        self['background'] = self.defaultBackground


class DisplayCanvas(Frame):

    def __init__(self, master, background='#404040', highlight='#558de8', **kw):
        Frame.__init__(self, master, **kw)
        self.x = self.y = 0

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        self.canvas = Canvas(self, bg=background, highlightbackground='#353535')
        self.canvas.grid(row=0, column=0, sticky='news')

        self.sbarv = Scrollbar(self, orient=VERTICAL, bg=background, highlightbackground=highlight)
        self.sbarh = Scrollbar(self, orient=HORIZONTAL, bg=background, highlightbackground=highlight)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.sbarv.grid(row=0, column=1, stick=N+S)
        self.sbarh.grid(row=1, column=0, sticky=E+W)

        self.canvas.bind('<Left>', self.on_left)
        self.canvas.bind('<Right>', self.on_right)
        self.canvas.bind('<Up>', self.on_up)
        self.canvas.bind('<Down>', self.on_down)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None
        self.image = None
        self.image_obj = None
        self.pil_image = None
        self.draw = False

        self.start_x = None
        self.start_y = None

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_button_press(self, event):
        self.canvas.focus_set()
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if not self.rect and self.draw:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9*w:
            self.on_right()
        elif event.x < 0.1*w:
            self.on_left()
        if event.y > 0.9*h:
            self.on_down()
        elif event.y < 0.1*h:
            self.on_up()

        if self.draw:
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_left(self, event=None):
        self.canvas.xview_scroll(-1, 'units')

    def on_right(self, event=None):
        self.canvas.xview_scroll(1, 'units')

    def on_up(self, event=None):
        self.canvas.yview_scroll(-1, 'units')

    def on_down(self, event=None):
        self.canvas.yview_scroll(1, 'units')

    def on_button_release(self, event):
        pass

    def update_image(self, image):
        self.draw = True
        self.pil_image = image
        self.image = ImageTk.PhotoImage(image)
        if self.image_obj is None:
            self.image_obj = self.canvas.create_image(1, 1, image=self.image, anchor=CENTER)
        else:
            self.canvas.itemconfig(self.image_obj, image=self.image)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def clear(self):
        self.canvas.delete("all")
        self.image_obj = self.canvas.create_image(1, 1, image=self.image, anchor=CENTER)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.rect = None

    def reset(self):
        self.canvas.delete("all")
        self.rect = None
        self.image = None
        self.image_obj = None
        self.pil_image = None
        self.draw = False

    def get_rect(self):
        w, h = self.pil_image.size
        x0, y0 = self.canvas.coords(self.image_obj)
        minx = x0 - w / 2.0
        miny = y0 - h / 2.0
        if self.rect:
            rect = self.canvas.coords(self.rect)
            rect = [rect[0] + abs(minx), rect[1] + abs(miny), rect[2] + abs(minx), rect[3] + abs(miny)]
            return rect
        else:
            return None


class Logger(Frame):

    def __init__(self, master=None, height=18, disable=True, **kw):
        Frame.__init__(self, master, **kw)
        self.text = scrolledtext.ScrolledText(self, height=height,
                                              bg='#002b36', fg='#eee8d5', insertbackground='#eee8d5')
        self.text.pack(expand=True, padx=50)
        self.disable = disable
        if self.disable:
            self.text.configure(state='disabled')

    def log(self, msg):
        self.text.configure(state='normal')
        self.text.insert(END, msg + '\n')
        if self.disable:
            self.text.configure(state='disabled')
        self.text.yview(END)

    def get(self):
        return self.text.get("1.0", END)

    def clear(self):
        self.text.configure(state='normal')
        self.text.delete('1.0', END)
        if self.disable:
            self.text.configure(state='disabled')
