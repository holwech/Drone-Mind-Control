#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QGridLayout, QApplication, QLabel, QTextEdit, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication


class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        review = QLabel('Log')
        self.reviewEdit = QTextEdit()
        self.reviewEdit.insertPlainText("TEST\n")
        grid = QGridLayout()
        grid.setSpacing(10)


        btn = QPushButton('Button', self)
        close_btn = QPushButton('Quit', self)
        close_btn.clicked.connect(QCoreApplication.instance().quit)

        grid.addWidget(close_btn, 12, 2)
        grid.addWidget(btn, 12,1)
        grid.addWidget(review, 0, 0)
        grid.addWidget(self.reviewEdit, 1, 0, 10, 3)

        self.setLayout(grid) 

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle('Review')
        self.show()

    def build_menu(self):
        #Create menu bar and add a file option
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        # Create an exit act
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        # Add the action to the file option
        fileMenu.addAction(exitAct)

        # Create and import mail act
        importAct = QAction('Import mail', self)

        # Create an import menu option
        importMenu = QMenu('Import', self)
        # Addt the import act as a suboption to the import option
        importMenu.addAction(importAct)

        # Create a new act
        newAct = QAction('New',self)
        # Add the new act to the file option
        fileMenu.addAction(newAct)
        #Add the import menu to the file option
        fileMenu.addMenu(importMenu)




    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
