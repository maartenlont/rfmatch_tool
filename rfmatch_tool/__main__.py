import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QSize
from PyQt5.QtGui import *

from gui.main_window import Ui_MainWindow
from components.Circuit import Circuit

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)

        # Connect the circuit list
        self.circuit = Circuit()
        self.listView.setModel(self.circuit)
#        self.listView.setIconSize(QSize(125,150))
        self.listView.setIconSize(QSize(75,90))

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
