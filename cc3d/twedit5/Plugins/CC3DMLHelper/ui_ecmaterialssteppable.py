# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ecmaterialssteppabledlg.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ECMaterialsSteppableDlg(object):
    def setupUi(self, ECMaterialsSteppableDlg):
        ECMaterialsSteppableDlg.setObjectName("ECMaterialsSteppableDlg")
        ECMaterialsSteppableDlg.resize(823, 585)
        self.gridLayout = QtWidgets.QGridLayout(ECMaterialsSteppableDlg)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(ECMaterialsSteppableDlg)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(ECMaterialsSteppableDlg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox_mtlC = QtWidgets.QComboBox(self.tab)
        self.comboBox_mtlC.setObjectName("comboBox_mtlC")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_mtlC)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.comboBox_mtlR = QtWidgets.QComboBox(self.tab)
        self.comboBox_mtlR.setObjectName("comboBox_mtlR")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_mtlR)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEdit_mtl = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_mtl.setObjectName("lineEdit_mtl")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_mtl)
        self.verticalLayout_2.addLayout(self.formLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_mtlAdd = QtWidgets.QPushButton(self.tab)
        self.pushButton_mtlAdd.setObjectName("pushButton_mtlAdd")
        self.horizontalLayout_2.addWidget(self.pushButton_mtlAdd)
        self.pushButton__mtlDel = QtWidgets.QPushButton(self.tab)
        self.pushButton__mtlDel.setObjectName("pushButton__mtlDel")
        self.horizontalLayout_2.addWidget(self.pushButton__mtlDel)
        self.pushButton__mtlClear = QtWidgets.QPushButton(self.tab)
        self.pushButton__mtlClear.setObjectName("pushButton__mtlClear")
        self.horizontalLayout_2.addWidget(self.pushButton__mtlClear)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.tableWidget_mtl = QtWidgets.QTableWidget(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget_mtl.sizePolicy().hasHeightForWidth())
        self.tableWidget_mtl.setSizePolicy(sizePolicy)
        self.tableWidget_mtl.setObjectName("tableWidget_mtl")
        self.tableWidget_mtl.setColumnCount(3)
        self.tableWidget_mtl.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_mtl.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_mtl.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_mtl.setHorizontalHeaderItem(2, item)
        self.horizontalLayout.addWidget(self.tableWidget_mtl)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.comboBox_fldC = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_fldC.setObjectName("comboBox_fldC")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_fldC)
        self.comboBox_fldR = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_fldR.setObjectName("comboBox_fldR")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_fldR)
        self.lineEdit_fld = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_fld.setObjectName("lineEdit_fld")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_fld)
        self.verticalLayout_3.addLayout(self.formLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_fldAdd = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_fldAdd.setObjectName("pushButton_fldAdd")
        self.horizontalLayout_4.addWidget(self.pushButton_fldAdd)
        self.pushButton_fldDel = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_fldDel.setObjectName("pushButton_fldDel")
        self.horizontalLayout_4.addWidget(self.pushButton_fldDel)
        self.pushButton_fldClear = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_fldClear.setObjectName("pushButton_fldClear")
        self.horizontalLayout_4.addWidget(self.pushButton_fldClear)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.tableWidget_fld = QtWidgets.QTableWidget(self.tab_2)
        self.tableWidget_fld.setObjectName("tableWidget_fld")
        self.tableWidget_fld.setColumnCount(3)
        self.tableWidget_fld.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fld.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fld.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_fld.setHorizontalHeaderItem(2, item)
        self.horizontalLayout_3.addWidget(self.tableWidget_fld)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 2)
        self.gridLayout_3.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_7 = QtWidgets.QLabel(self.tab_3)
        self.label_7.setObjectName("label_7")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setObjectName("label_10")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.comboBox_cellM = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_cellM.setObjectName("comboBox_cellM")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_cellM)
        self.comboBox_cellT = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_cellT.setObjectName("comboBox_cellT")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_cellT)
        self.comboBox_cellR = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_cellR.setObjectName("comboBox_cellR")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_cellR)
        self.lineEdit_cell = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_cell.setObjectName("lineEdit_cell")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_cell)
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.comboBox_cellT_New = QtWidgets.QComboBox(self.tab_3)
        self.comboBox_cellT_New.setObjectName("comboBox_cellT_New")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.comboBox_cellT_New)
        self.verticalLayout_4.addLayout(self.formLayout_3)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_cellAdd = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_cellAdd.setObjectName("pushButton_cellAdd")
        self.horizontalLayout_6.addWidget(self.pushButton_cellAdd)
        self.pushButton_cellDel = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_cellDel.setObjectName("pushButton_cellDel")
        self.horizontalLayout_6.addWidget(self.pushButton_cellDel)
        self.pushButton_cellClear = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_cellClear.setObjectName("pushButton_cellClear")
        self.horizontalLayout_6.addWidget(self.pushButton_cellClear)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.tableWidget_cell = QtWidgets.QTableWidget(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget_cell.sizePolicy().hasHeightForWidth())
        self.tableWidget_cell.setSizePolicy(sizePolicy)
        self.tableWidget_cell.setObjectName("tableWidget_cell")
        self.tableWidget_cell.setColumnCount(5)
        self.tableWidget_cell.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_cell.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_cell.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_cell.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_cell.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_cell.setHorizontalHeaderItem(4, item)
        self.horizontalLayout_5.addWidget(self.tableWidget_cell)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 2)
        self.gridLayout_4.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableWidget_diff = QtWidgets.QTableWidget(self.tab_4)
        self.tableWidget_diff.setObjectName("tableWidget_diff")
        self.tableWidget_diff.setColumnCount(3)
        self.tableWidget_diff.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_diff.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_diff.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_diff.setHorizontalHeaderItem(2, item)
        self.gridLayout_5.addWidget(self.tableWidget_diff, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.retranslateUi(ECMaterialsSteppableDlg)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ECMaterialsSteppableDlg)

    def retranslateUi(self, ECMaterialsSteppableDlg):
        _translate = QtCore.QCoreApplication.translate
        ECMaterialsSteppableDlg.setWindowTitle(_translate("ECMaterialsSteppableDlg", "ECMaterials Steppable: Please define extracellular material interactions"))
        self.label.setText(_translate("ECMaterialsSteppableDlg", "Catalyst"))
        self.label_2.setText(_translate("ECMaterialsSteppableDlg", "Reactant"))
        self.label_3.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.pushButton_mtlAdd.setText(_translate("ECMaterialsSteppableDlg", "Add Interaction"))
        self.pushButton__mtlDel.setText(_translate("ECMaterialsSteppableDlg", "Delete Interaction"))
        self.pushButton__mtlClear.setText(_translate("ECMaterialsSteppableDlg", "Clear Table"))
        item = self.tableWidget_mtl.horizontalHeaderItem(0)
        item.setText(_translate("ECMaterialsSteppableDlg", "Catalyst"))
        item = self.tableWidget_mtl.horizontalHeaderItem(1)
        item.setText(_translate("ECMaterialsSteppableDlg", "Reactant"))
        item = self.tableWidget_mtl.horizontalHeaderItem(2)
        item.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("ECMaterialsSteppableDlg", "Material Interactions"))
        self.label_4.setText(_translate("ECMaterialsSteppableDlg", "Catalyst"))
        self.label_5.setText(_translate("ECMaterialsSteppableDlg", "Reactant"))
        self.label_6.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.pushButton_fldAdd.setText(_translate("ECMaterialsSteppableDlg", "Add Interaction"))
        self.pushButton_fldDel.setText(_translate("ECMaterialsSteppableDlg", "Delete Interaction"))
        self.pushButton_fldClear.setText(_translate("ECMaterialsSteppableDlg", "Clear Table"))
        item = self.tableWidget_fld.horizontalHeaderItem(0)
        item.setText(_translate("ECMaterialsSteppableDlg", "Catalyst"))
        item = self.tableWidget_fld.horizontalHeaderItem(1)
        item.setText(_translate("ECMaterialsSteppableDlg", "Reactant"))
        item = self.tableWidget_fld.horizontalHeaderItem(2)
        item.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("ECMaterialsSteppableDlg", "Field Interactions"))
        self.label_7.setText(_translate("ECMaterialsSteppableDlg", "ECMaterial"))
        self.label_8.setText(_translate("ECMaterialsSteppableDlg", "Cell Type"))
        self.label_9.setText(_translate("ECMaterialsSteppableDlg", "Response Type"))
        self.label_10.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.label_11.setText(_translate("ECMaterialsSteppableDlg", "New Cell Type: "))
        self.pushButton_cellAdd.setText(_translate("ECMaterialsSteppableDlg", "Add Interaction"))
        self.pushButton_cellDel.setText(_translate("ECMaterialsSteppableDlg", "Delete Interaction"))
        self.pushButton_cellClear.setText(_translate("ECMaterialsSteppableDlg", "Clear Table"))
        item = self.tableWidget_cell.horizontalHeaderItem(0)
        item.setText(_translate("ECMaterialsSteppableDlg", "ECMaterial"))
        item = self.tableWidget_cell.horizontalHeaderItem(1)
        item.setText(_translate("ECMaterialsSteppableDlg", "Cell Type"))
        item = self.tableWidget_cell.horizontalHeaderItem(2)
        item.setText(_translate("ECMaterialsSteppableDlg", "Response Type"))
        item = self.tableWidget_cell.horizontalHeaderItem(3)
        item.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        item = self.tableWidget_cell.horizontalHeaderItem(4)
        item.setText(_translate("ECMaterialsSteppableDlg", "New Cell Type"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("ECMaterialsSteppableDlg", "Cell Interactions"))
        item = self.tableWidget_diff.horizontalHeaderItem(0)
        item.setText(_translate("ECMaterialsSteppableDlg", "ECMaterial"))
        item = self.tableWidget_diff.horizontalHeaderItem(1)
        item.setText(_translate("ECMaterialsSteppableDlg", "Diffuses"))
        item = self.tableWidget_diff.horizontalHeaderItem(2)
        item.setText(_translate("ECMaterialsSteppableDlg", "Coefficient"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("ECMaterialsSteppableDlg", "Material Diffusion"))
