import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from functools import partial
import time
import numpy as np
import pickle

from ml.datasets.flights.FlightHoloClean import FlightHoloClean
from ml.datasets.blackOak.BlackOakDataSetUppercase import BlackOakDataSetUppercase


class MyDialog(QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)

        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Ok)

        self.textBrowser = QTextBrowser(self)
        self.textBrowser.append("Why:")

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.destroy)


class Example(QWidget):
    def __init__(self, data):
        super(Example, self).__init__()

        self.data = data

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



        path = "/home/felix/SequentialPatternErrorDetection/explain_model/"

        data_map={}
        data_map[FlightHoloClean.name]="flights"
        data_map[BlackOakDataSetUppercase.name] = "blackoak"


        self.classifier = pickle.load(open( path + data_map[self.data.name] + "/classifier.p", "rb"))
        self.y_pred = pickle.load(open(path + data_map[self.data.name] + "/pedictions.p", "rb"))
        self.feature_name_list = pickle.load(open(path + data_map[self.data.name] + "/feature_names.p", "rb"))
        self.feature_matrix = pickle.load(open(path + data_map[self.data.name] + "/feature_matrix.p", "rb"))


        self.dirty_pd = self.data.dirty_pd

        self.column_has_no_error = np.zeros(self.dirty_pd.shape[1], dtype=bool)
        self.is_error = self.data.matrix_is_error

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

        self.dialogTextBrowser = MyDialog(self)

        self.resize(1250, 1000)
        self.center()

        self.setWindowTitle('Prediction Explainer')
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

    def explain_prediction(self, x, model):
        from eli5.explain import explain_prediction
        params = {}
        params['feature_names'] = self.feature_name_list
        params['top'] = 5
        expl = explain_prediction(model, x, **params)
        from eli5.formatters import format_as_text
        params_text = {}
        params_text['show_feature_values'] = True
        return format_as_text(expl, **params_text)

    def setData(self):
        val = self.dirty_pd.values

        # set data
        for r in range(self.N):
            for c in range(self.dirty_pd.shape[1]):
                item = QTableWidgetItem(str(val[r][c]))
                item.setFlags(Qt.ItemIsSelectable)
                self.table.setItem(r, c, item)



                if c in self.y_pred and self.y_pred[c][r] > 0.5:
                    #print self.y_pred[c]
                    self.table.item(r, c).setBackground(QColor(255, 0, 0))
                    self.table.item(r, c).setTextColor(QColor(255, 255, 255))
                else:
                    self.table.item(r, c).setBackground(QColor(255, 255, 255))
                    self.table.item(r, c).setTextColor(QColor(0, 0, 0))


        header = self.table.horizontalHeader()
        for c in range(self.dirty_pd.shape[1]):
            header.setResizeMode(c, QHeaderView.ResizeToContents)

        self.time = time.time()

    def selectCell(self, row, column):
        print("Row %d and Column %d was clicked" % (row, column))

        self.dialogTextBrowser.textBrowser.setText(self.explain_prediction(self.feature_matrix[row,:], self.classifier[column]))

        self.dialogTextBrowser.exec_()


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

    #data = BlackOakDataSetUppercase()
    data = FlightHoloClean()

    ex = Example(data)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()