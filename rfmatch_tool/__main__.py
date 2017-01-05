import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from .gui.main_window import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    pass

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
