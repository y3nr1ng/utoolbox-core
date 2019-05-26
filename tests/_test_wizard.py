import sys
import PySide2
from PySide2.QtWidgets import (
    QApplication, 
    QLabel, QLineEdit, QHBoxLayout,
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
page1.setTitle("Page 1 is best!")
page1.setSubTitle("1111111111")
lineEdit = QLineEdit()
hLayout1 = QHBoxLayout(page1)
hLayout1.addWidget(lineEdit)

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
