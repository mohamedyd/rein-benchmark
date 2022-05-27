#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from functools import partial
import time
import numpy as np
import pandas as pd

class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()

        self.initUI()

    def fillna_df(self, df):
        for i in range(df.shape[1]):
            if df[df.columns[i]].dtypes.name == "object":
                df[df.columns[i]] = df[df.columns[i]].fillna('')
            else:
                raise Exception('not implemented')
            #todo if numeric
        return df

    def initUI(self):

        filename = QFileDialog.getOpenFileName(self, 'Open File', '/', "CSV files (*.csv *.txt)")
        print filename

        self.dirty_pd = self.fillna_df(pd.read_csv(str(filename), header=0, dtype=object))

        self.column_has_no_error = np.zeros(self.dirty_pd.shape[1], dtype=bool)
        self.is_error = np.ones(self.dirty_pd.shape) * -1

        number_columns = self.dirty_pd.shape[1]

        self.N = 1000

        self.table = QTableWidget(self)

        # initiate table
        self.table.resize(1250, 500)
        self.table.setRowCount(self.N)
        self.table.setColumnCount(number_columns)

        self.table.setHorizontalHeaderLabels(self.dirty_pd.columns)
        self.table.verticalHeader().hide()
        self.table.move(0, 70)

        self.table.cellClicked.connect(partial(self.selectCell))

        header = self.table.horizontalHeader()
        self.connect(header, SIGNAL("sectionClicked(int)"), self.header_clicked)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)


        self.setData()

        label = QLabel("Please, identify whether the marked cell is erroneous:", self)
        label.move(50, 10)

        self.resize(1250, 1000)
        self.center()

        self.setWindowTitle('Algorithm initializer')
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def header_clicked(self, column):
        print "column selected: " + str(column)
        if self.column_has_no_error[column]:
            self.column_has_no_error[column] = False
        else:
            self.column_has_no_error[column] = True

        self.format_column(column, self.column_has_no_error[column])


    def format_column(self, column, no_error):
        for i in range(self.N):
            if no_error:
                self.table.item(i, column).setBackground(QColor(0, 255, 0))
                self.table.item(i, column).setTextColor(QColor(255, 255, 255))
            else:
                self.table.item(i, column).setBackground(QColor(255, 255, 255))
                self.table.item(i, column).setTextColor(QColor(0, 0, 0))

    def setData(self):
        val = self.dirty_pd.values

        # set data
        for r in range(self.N):
            for c in range(self.dirty_pd.shape[1]):
                item = QTableWidgetItem(str(val[r][c]))
                item.setFlags(Qt.ItemIsSelectable)
                self.table.setItem(r, c, item)

                self.table.item(r, c).setBackground(QColor(255, 255, 255))
                self.table.item(r, c).setTextColor(QColor(0, 0, 0))

        header = self.table.horizontalHeader()
        for c in range(self.dirty_pd.shape[1]):
            header.setResizeMode(c, QHeaderView.ResizeToContents)

        self.time = time.time()

    def selectCell(self, row, column):
        print("Row %d and Column %d was clicked" % (row, column))

        if self.is_error[row,column] == -1:
            self.is_error[row, column] = 1
            self.table.item(row, column).setBackground(QColor(255, 0, 0))
            self.table.item(row, column).setTextColor(QColor(255, 255, 255))
        elif self.is_error[row,column] == 1:
            self.is_error[row, column] = 0
            self.table.item(row, column).setBackground(QColor(0, 255, 0))
            self.table.item(row, column).setTextColor(QColor(255, 255, 255))
        else:
            self.is_error[row, column] = -1
            self.table.item(row, column).setBackground(QColor(255, 255, 255))
            self.table.item(row, column).setTextColor(QColor(0, 0, 0))

        self.check_whether_done()


    def check_whether_done(self):
        done = 0
        for c in range(self.dirty_pd.shape[1]):
            if self.column_has_no_error[c] or \
                    (len(np.where(self.is_error[:,c] == 0)[0]) >= 2 and
                             len(np.where(self.is_error[:,c] == 1)[0]) >= 2):
                done += 1
            else:
                break

        if done == self.dirty_pd.shape[1]:
            print "run-time: " + str(time.time() - self.time)
            self.close()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()