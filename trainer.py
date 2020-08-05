from tkinter import Tk
from invoicenet.gui.trainer import Trainer


def main():
    root = Tk()
    Trainer(root)
    root.mainloop()


if __name__ == '__main__':
    main()
