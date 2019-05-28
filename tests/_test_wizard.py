import sys
import PySide2
from PySide2.QtWidgets import (
    QApplication, 
    QButtonGroup, QComboBox, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QVBoxLayout,
    QWizard, QWizardPage
)

print("Qt Version: " + PySide2.__version__)


# CREATE WIZARD, WATERMARK, LOGO, BANNER
app = QApplication(sys.argv)
wizard = QWizard()
wizard.setWizardStyle(QWizard.ModernStyle)

wizard.setPixmap(QWizard.WatermarkPixmap, "Watermark.png")
wizard.setPixmap(QWizard.LogoPixmap, "Logo.png")
wizard.setPixmap(QWizard.BannerPixmap, "Banner.png")

# CREATE PAGE 1, LINE EDIT, TITLES
page1 = QWizardPage()
page1.setTitle("Import dataset")
page1.setSubTitle("Determine dataset format")
lineEdit = QLineEdit()
button_group = QButtonGroup()
button1 = QRadioButton("SPIM")
button2 = QRadioButton("uManager")
button3 = QRadioButton("Generic")
button_group.addButton(button1)
button_group.addButton(button2)
button_group.addButton(button3)
combo = QComboBox()
combo.addItem("Image collection")
combo.addItem("Volume tiles")
hLayout1 = QVBoxLayout(page1)
hLayout1.addWidget(lineEdit)
hLayout1.addWidget(button3)
hLayout1.addWidget(combo)
hLayout1.addWidget(button1)
hLayout1.addWidget(button2)



page1.registerField("myField*", lineEdit, lineEdit.text(), "textChanged")

# CREATE PAGE 2, LABEL, TITLES
page2 = QWizardPage()
page2.setFinalPage(True)
page2.setTitle("Page 2 is better!")
page2.setSubTitle("Lies!")
label = QLabel()
hLayout2 = QHBoxLayout(page2)
hLayout2.addWidget(label)

# CONNECT SIGNALS AND PAGES
# lineEdit.textChanged.connect(lambda:label.setText(lineEdit.text()))
nxt = wizard.button(QWizard.NextButton)
def func():
    label.setText(page1.field("myField"))
nxt.clicked.connect(func)
wizard.addPage(page1)
wizard.addPage(page2)

wizard.show()
sys.exit(app.exec_())
