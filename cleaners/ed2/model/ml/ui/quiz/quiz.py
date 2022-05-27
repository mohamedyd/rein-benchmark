#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import sys
import time
from functools import partial

from PyQt4.QtGui import *


class Example(QWidget):
    def __init__(self, data):
        super(Example, self).__init__()
        self.data = data
        self.current_row = 0
        self.current_column = 5

        self.timer = time.time()

        self.time_sum = 0
        self.label_count = 0


        self.initUI()

    def initUI(self):


        number_columns = self.data.clean_pd.shape[1]

        self.table = QTableWidget(self)

        # initiate table
        self.table.resize(1250, 150)
        self.table.setRowCount(1)
        self.table.setColumnCount(number_columns)

        self.table.setHorizontalHeaderLabels(self.data.clean_pd.columns)
        self.table.verticalHeader().hide()


        self.setData()

        label = QLabel("Please, identify whether the marked cell is erroneous:", self)
        label.move(350, 70)

        qbtnCorrect = QPushButton('correct', self)
        qbtnCorrect.clicked.connect(partial(self.pushButton, False))
        qbtnCorrect.resize(qbtnCorrect.sizeHint())
        qbtnCorrect.move(350, 100)

        qbtnErr = QPushButton('error', self)
        qbtnErr.clicked.connect(partial(self.pushButton, True))
        qbtnErr.resize(qbtnErr.sizeHint())
        qbtnErr.move(620, 100)

        self.resize(1250, 150)
        self.center()

        self.setWindowTitle('Domain knowledge teacher')
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def cleanFormat(self):
        self.table.item(0, self.current_column).setBackground(QColor(255, 255, 255))
        self.table.item(0, self.current_column).setTextColor(QColor(0, 0, 0))


    def setData(self):
        self.current_row= random.randint(0, self.data.shape[0]-1)
        self.current_column = random.randint(0, self.data.shape[1] - 1)
        #self.current_column = 8


        val = self.data.dirty_pd.values

        # set data
        for c in range(self.data.shape[1]):
            self.table.setItem(0, c, QTableWidgetItem(str(val[self.current_row][c])))

        self.table.item(0, self.current_column).setBackground(QColor(100, 100, 150))
        self.table.item(0, self.current_column).setTextColor(QColor(255, 255, 255))

        header = self.table.horizontalHeader()
        for c in range(self.data.shape[1]):
            header.setResizeMode(c, QHeaderView.ResizeToContents)

    def pushButton(self, clicked):

        current_time_period = time.time() - self.timer
        self.time_sum += current_time_period
        self.label_count += 1
        self.timer = time.time()

        print "current: " + str(current_time_period) + " avg: " + str(self.time_sum / float(self.label_count)) + " count: " + str(self.label_count)

        is_error = self.data.matrix_is_error[self.current_row, self.current_column]
        clean_cell = str(self.data.clean_pd.values[self.current_row, self.current_column])
        dirty_cell = str(self.data.dirty_pd.values[self.current_row, self.current_column])

        if clicked == is_error:
            print "well done"
        else:
            print "wrong"

            if is_error:
                QMessageBox.about(self, "Wrong", "Clean = '%s', dirty = '%s'" % (clean_cell, dirty_cell))
            else:
                QMessageBox.about(self, "Wrong", "'%s' was a correct entry" % (clean_cell))

        self.table.setItem(0, 1, QTableWidgetItem(str("test")))

        self.cleanFormat()

        self.setData()


def main():
    #data = BlackOakDataSet()
    #data = FlightHoloClean()

    #data = BlackOakDataSetUppercase()

    from ml.datasets.mohammad.MohammadDataSet import MohammadDataSet
    data = MohammadDataSet("tax", 20, 30, 10)

    app = QApplication(sys.argv)
    ex = Example(data)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()