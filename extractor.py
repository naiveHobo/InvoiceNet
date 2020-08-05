from tkinter import Tk
from invoicenet.gui.extractor import Extractor


def main():
    root = Tk()
    Extractor(root)
    root.mainloop()


if __name__ == '__main__':
    main()
