#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs,os,sys,platform,subprocess,random,natsort,cv2,easygui
import xml.etree.ElementTree as ET
from copy import deepcopy
from strsimpy.jaro_winkler import JaroWinkler
import numpy as np
from skimage import exposure
from functools import partial

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

sys.path.append('pytorch_yolov5/')
from models.experimental import *
from utils.datasets import *
from utils.utils import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem

__appname__ = 'AutoLabelImg  '

class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.usingPascalVocFormat = True
        self.usingYoloFormat = False

        #define global path for myown use
        self.img_folder_path=''
        self.xml_folder_path=''
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        
        
        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox('Default Label')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        #listLayout.addWidget(self.editButton)
        #listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)
        #test: show local enlarge img
        self.new_test = QLabel(self)
        self.new_test.setGeometry(0,0,360,360)
        listLayout.addWidget(self.new_test)

        # Create and add combobox for showing unique labels in group 
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)
        

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)
        
        self.dock = QDockWidget('Magic Lens', self)   #getStr('boxLabelText')
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setMouseTracking(True)
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        # action()函数的输入参数，第一个str是显示按钮的名称，第二个是对应的触发函数，第三个是快捷键，第四个是icon，第五个目前不知道,不用也不报错
        
        #自己增加的action：
        search_system=action('Search_System', self.search_actions_info,None,'zoom-in')
        batch_rename_img=action('batch_rename_img', self.batch_rename_img, 'Ctrl+1','edit')
        rename_img_xml=action('rename_img_xml', self.rename_img_xml, 'Ctrl+Alt+1','edit')
        duplicate_xml=action('duplicate_xml', self.make_duplicate_xml,'Ctrl+2', 'copy')
        batch_duplicate=action('batch_duplicate', self.batch_duplicate_xml,'Ctrl+Alt+2', 'copy')
        label_pruning=action('label_pruning', self.prune_useless_label,'Ctrl+3', 'delete')
        file_pruning=action('file_pruning', self.remove_extra_img_xml,'Ctrl+Alt+3', 'delete')
        change_label=action('change_label', self.change_label_name, 'Ctrl+4','color_line')
        fix_property=action('fix_property', self.fix_xml_property, 'Ctrl+5','color_line')
        auto_labeling=action('auto_labeling', self.auto_labeling,'Ctrl+6', 'new')
        data_agument=action('data_agument', self.data_auto_agument,'Ctrl+7', 'copy')
        
        folder_info=action('folder_info', self.show_folder_infor,'Alt+1', 'help')
        label_info=action('label_info', self.show_label_info,'Alt+2', 'help')
        
        extract_video=action('extract_video', self.extract_video,'Shift+1', 'new')
        extract_stream=action('extract_stream', self.extract_stream,'Shift+2', 'new')
        batch_resize_img=action('batch_resize_img', self.batch_resize_img,'Shift+3', 'fit-window')
        merge_video=action('merge_video', self.merge_video,'Shift+4', 'open')
        annotation_video=action('annotation_video', self.annotation_video,'Shift+5', 'new')
        
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action('Open Img', self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        changeSavedir = action('Open Xml', self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action('Next Img', self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action('Prev Img', self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action('Verify', self.verifyImg,
                        'space', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        save_format = action('VOC', self.change_format,
                      'Ctrl+', 'format_voc', getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action('Create Box', self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action('Delete Box', self.deleteSelectedShape,
                        'Delete', 'delete', getStr('delBoxDetail'), enabled=False)
        copy = action('Copy Box', self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close, resetAll = resetAll,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll,quit,duplicate_xml,batch_duplicate,label_pruning,folder_info,data_agument),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            annotate=self.menu('&Annotate-Tools'),
            video=self.menu('&Video-Tools'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.useMagnifyingLens = QAction('Magnifying Lens', self)
        self.useMagnifyingLens.setCheckable(True)
        self.useMagnifyingLens.setChecked(settings.get(SETTING_Magnifying_Lens, False))
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+P")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        # 设置菜单栏
        addActions(self.menus.annotate,(batch_rename_img,rename_img_xml,None,
                    duplicate_xml,batch_duplicate,None,
                    label_pruning,file_pruning,change_label,fix_property,None,
                    auto_labeling,data_agument,None,
                    folder_info,label_info))
        addActions(self.menus.video,(extract_video,extract_stream,None,batch_resize_img,merge_video,None,annotation_video))
        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format, saveAs, close, resetAll, quit))
        addActions(self.menus.help, (help, showInfo, None, search_system))
        addActions(self.menus.view, (
            self.useMagnifyingLens,
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.usingPascalVocFormat = True
            self.usingYoloFormat = False
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.usingPascalVocFormat = False
            self.usingYoloFormat = True
            LabelFile.suffix = TXT_EXT

    def change_format(self):
        if self.usingPascalVocFormat: self.set_format(FORMAT_YOLO)
        elif self.usingYoloFormat: self.set_format(FORMAT_PASCALVOC)

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def showInfoDialog(self):
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:

                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                if snapped:
                    self.setDirty()

                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)

            self.addLabel(shape)
        self.updateComboBox()
        self.canvas.loadShapes(s)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]
            
        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append('')
        uniqueTextList.sort()

        self.comboBox.update_items(uniqueTextList)

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb(),
                        fill_color=s.fill_color.getRgb(),
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.usingYoloFormat is True:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)
    
    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == '':
                self.labelList.item(i).setCheckState(2) 
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            # if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(self.filePath)[0])
                xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
                txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

                """Annotation file priority:
                PascalXML > YOLO
                """
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)
            else:
                xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                txtPath = os.path.splitext(filePath)[0] + TXT_EXT
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath)
                elif os.path.isfile(txtPath):
                    self.loadYOLOTXTByFilename(txtPath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_Magnifying_Lens] = self.useMagnifyingLens.isChecked()
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath
        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return
            

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent!=True :
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                         '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                         QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        #print(targetDirPath)
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]
        
        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                #print(savedPath)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0] # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        print (shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())
        
    def function_test(self):
        progress = QProgressDialog(self)
        progress.setWindowTitle("请稍等")  
        progress.setLabelText("正在操作...")
        progress.setCancelButtonText("取消")
        progress.setMinimumDuration(5)
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0,100) 
        for i in range(100):
            progress.setValue(i) 
            import time
            time.sleep(0.2)
            if progress.wasCanceled():
                QMessageBox.warning(self,"提示","操作失败,请检查文件路径！") 
                break
        else:
            progress.setValue(100)
            QMessageBox.information(self,"提示","数据增强成功！")
    
    def test_act(self):
        go=True
        while go:
            if self.show_test():
                QMessageBox.information(self,u'Wrong!',u'!拉出去脱水！')
            else:
                go=False
                QMessageBox.information(self,u'Right!',u'消灭人类暴政，世界属于三体！') 
        
    def show_test(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'三体好不好看?'
        return no == QMessageBox.warning(self, u'Attention:', msg, yes | no)  

    def search_actions_info(self):
        """this is a action information search system.
        input actions name, ti will tell you what it is and how to use it.
        for example, input 'rename_img_xml' or 'ca1' can read actions <rename_img_xml>'s instruction.
        key_word must be actions in menu bar or its shorcut key for now, more intelligent system will update lately. 
        """
        def find_max_similarity(test_str,str_list,threshold=0.7):
            jarowinkler = JaroWinkler()
            max_simi=threshold
            max_str=None
            for string in str_list:
                simi=jarowinkler.similarity(test_str, string)
                if simi > max_simi:
                    max_str=string
                    max_simi = simi
                    
            return max_str
            
        search_key, ok=QInputDialog.getText(self, 'Text Input Dialog', 
                    "Input your search key word：\n(input nothing for search-system's own instruction)")
        if (not ok):
            return
        action_dict={'batch_rename_img':self.batch_rename_img,'c1':self.batch_rename_img,
                     'rename_img_xml':self.rename_img_xml,'ca1':self.rename_img_xml,
                     'duplicate_xml':self.make_duplicate_xml,'c2':self.make_duplicate_xml,
                     'batch_duplicate':self.batch_duplicate_xml,'ca2':self.batch_duplicate_xml,
                     'label_pruning':self.prune_useless_label,'c3':self.prune_useless_label,
                     'file_pruning':self.remove_extra_img_xml,'ca3':self.remove_extra_img_xml,
                     'change_label':self.change_label_name,'c4':self.change_label_name,
                     'fix_property':self.fix_xml_property,'c5':self.fix_xml_property,
                     'auto_labeling':self.auto_labeling,'c6':self.auto_labeling,
                     'data_agument':self.data_auto_agument,'c7':self.data_auto_agument,
                     'folder_info':self.show_folder_infor,'a1':self.show_folder_infor,
                     'label_info':self.show_label_info,'a2':self.show_label_info,
                     'extract_video':self.extract_video,'s1':self.extract_video,
                     'extract_stream':self.extract_stream,'s2':self.extract_stream,
                     'batch_resize_img':self.batch_resize_img,'s3':self.batch_resize_img,
                     'merge_video':self.merge_video,'s4':self.merge_video,
                     'annotation_video':self.annotation_video,'s5':self.annotation_video,
                     'Search_System':self.search_actions_info
                     }
        search_key='Search_System' if search_key=='' else search_key
        if search_key in action_dict.keys():
            search_info=action_dict[search_key].__doc__
            search_info=search_info.replace('  ','')
            QMessageBox.information(self,u'Info!',search_info)
        else:
            vague_key=find_max_similarity(search_key,action_dict.keys())
            if vague_key in action_dict.keys():
                search_info=action_dict[vague_key].__doc__
                search_info=search_info.replace('  ','')
                search_info="here is info about '{}' based on your input: '{}'\n\n".format(vague_key,search_key)+search_info
                QMessageBox.information(self,u'Info!',search_info)
            else:
                QMessageBox.information(self,u'Sorry!',
                u'unkown key word, key word must in(or similar to) actions in menu bar or its shotcut key, please try again.')
        
    def batch_rename_img(self):
        """batch rename img name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty. after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            path=os.path.dirname(self.filePath)
            filelist=natsort.natsorted(os.listdir(path))
            key_word,ok=QInputDialog.getText(self, 'Text Input Dialog',"Input key word：")
            if not ok:
                return
            index,ok=QInputDialog.getInt(self, 'Text Input Dialog',"Input index：",value=1)
            if not ok:
                return
            Fill,ok=QInputDialog.getInt(self, 'Text Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(filelist))):
                QMessageBox.information(self,u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word+'_'
            for item in filelist:
                if item.endswith('.jpg') or item.endswith('.jpeg') or item.endswith('.png'):
                    filepath=os.path.join(os.path.abspath(path), item)
                    if Fill > 1:
                        new_item='{}{}.jpg'.format(key_word,str(index).zfill(Fill))
                    else:
                        new_item='{}{}.jpg'.format(key_word,str(index))
                    new_filepath=os.path.join(os.path.abspath(path), new_item)
                    os.rename(filepath,new_filepath)
                    index+=1
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def rename_img_xml(self):
        """batch rename img's and its corresponding xml's name. 
        new name constructed by key_word, index, if_fill three prospoty,for example,'car_1.jpg'(or 'car_001.jpg' if fill to 3 digit). 
        additionally, '_' will not appear when key_word is empty.after this actions, you may need reopen img folder.
        !!makesure your new name not conflict with your old name!!
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path=os.path.dirname(self.filePath)
            xml_folder_path=self.defaultSaveDir
            imglist = natsort.natsorted(os.listdir(img_folder_path))
            xmllist = natsort.natsorted(os.listdir(xml_folder_path))
            key_word,ok=QInputDialog.getText(self, 'Text Input Dialog',"Input key word：")
            if not ok:
                return
            index,ok=QInputDialog.getInt(self, 'Int Input Dialog',"Input index：",value=1)
            if not ok:
                return
            Fill,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                    'Digit Fill\n(fill means 1->001, 0 means no fill)',value=0)
            if not ok:
                return
            if 1 < Fill < len(str(index+len(imglist))):
                QMessageBox.information(self,u'Waring!',
                u"your Fill is smaller than largest index's digit, try larger Fill or input 0 to use no fill")
                return
            key_word = '' if key_word == '' else key_word+'_'
            for item in xmllist:
                if item.endswith('.xml') and (item[0:-4]+'.jpg' in imglist or item[0:-4]+'.JPG' in imglist):
                    xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                    imgPath=os.path.join(os.path.abspath(img_folder_path), item[0:-4])+'.jpg'
                    if Fill > 1:
                        new_item='{}{}'.format(key_word,str(index).zfill(Fill))
                    else:
                        new_item='{}{}'.format(key_word,str(index))
                    new_xmlPath=os.path.join(os.path.abspath(xml_folder_path), new_item+'.xml')
                    new_imgPath=os.path.join(os.path.abspath(img_folder_path), new_item+'.jpg')
                    os.rename(xmlPath,new_xmlPath)
                    os.rename(imgPath,new_imgPath)
                    index+=1
                else:
                    pass
            QMessageBox.information(self,u'Done!',u'Batch rename done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
            
    def make_duplicate_xml(self):
        """copy last xml file to local img, make sure last xml exist. 
        if local xml exist, you need confirm to overwrite it.
        """
        try:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex - 1 >= 0:
                last_filename = self.mImgList[currIndex - 1]
                imgFileName = os.path.basename(last_filename)
                last_xml = os.path.splitext(imgFileName)[0]
                last_path=os.path.join(ustr(self.defaultSaveDir),last_xml+'.xml')
                
                currfilename = self.mImgList[currIndex]
                imgFileName = os.path.basename(currfilename)
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path=os.path.join(ustr(self.defaultSaveDir),curr_xml+'.xml')

                xml_info={'filename':'none','path':'none'}
                xml_info['filename']=curr_xml+'.jpg'
                xml_info['path']=str(self.filePath)
                if os.path.exists(save_path):
                    if self.question_1():
                        print('over write!')
                        pass
                    else:
                        print('cancled!')
                        return

                tree = ET.ElementTree(file=last_path)
                root=tree.getroot()
                for key in xml_info.keys():
                    root.find(key).text=xml_info[key]
                tree.write(save_path)
            else:
                QMessageBox.information(self,u'Sorry!',u'please ensure the first xml file exists.')
                return
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
            
    def question_1(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'current xml exists,procesing anyway?'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
        
    def batch_duplicate_xml(self):
        """batch copy xml file, make sure at least the first xml exist.
        this action will not overwrite xml file, if local xml exist, it will jump to next and copy local xml to next one.
        """
        if len(self.mImgList) <= 0:
            QMessageBox.information(self,u'Sorry!',u'something is wrong, try load img/xml path again.')
        else:
            for i in range(len(self.mImgList)):
                currfilename = self.mImgList[i]
                imgFileName = os.path.basename(currfilename)
                curr_xml = os.path.splitext(imgFileName)[0]
                save_path=os.path.join(ustr(self.defaultSaveDir),curr_xml+'.xml')
                if i ==0:
                    if os.path.exists(save_path):
                        pass
                    else:
                        QMessageBox.information(self,u'Sorry!',u'please ensure the first xml file exists.')
                        return
                else:
                    last_filename = self.mImgList[i - 1]
                    imgFileName = os.path.basename(last_filename)
                    last_xml = os.path.splitext(imgFileName)[0]
                    last_path=os.path.join(ustr(self.defaultSaveDir),last_xml+'.xml')
                    if os.path.exists(save_path):
                        pass
                    else:
                        xml_info={'filename':'none','path':'none'}
                        xml_info['filename']=curr_xml+'.jpg'
                        xml_info['path']=str(self.filePath)
                        tree = ET.ElementTree(file=last_path)
                        root=tree.getroot()
                        for key in xml_info.keys():
                            root.find(key).text=xml_info[key]
                        tree.write(save_path)          
            QMessageBox.information(self,u'Done!',u'batch duplicate xml file succeed, you can procesing other job now.')
        
    def prune_useless_label(self):
        """delete useless label.
        input label name you want to keep, others will be deleted.
        a img whose xml's object(label) deleted completly, this img and its xml will be deleted.
        after this actions, you may need reopen img folder.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        prune_list=[]
        text, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input labels witch you want keep(split by ',')：")
        text=text.replace(" ","")
        if ok and text:
            for item in text.split(','):
                prune_list.append(item)
            print(prune_list)
        else:
            QMessageBox.information(self,u'Wrong!',u'get empty list, please try again.')
            return
        if self.question_2(prune_list):
            try:
                print(self.defaultSaveDir,os.path.dirname(self.filePath))  #当前实时的img路径和xml路径
                imglist = os.listdir(os.path.dirname(self.filePath))
                xmllist = os.listdir(self.defaultSaveDir)
                if len(imglist)!=len(xmllist):
                    QMessageBox.information(self,u'Wrong!',u'file list length are different({0}/{1}), please check.'.format(len(imglist),len(xmllist)))
                else:
                    for item in xmllist:
                        xmlPath=os.path.join(os.path.abspath(self.defaultSaveDir), item)
                        tree = ET.ElementTree(file=xmlPath)
                        root=tree.getroot()
                        keep=False
                        for obj in root.findall('object'):
                            if obj.find('name').text in prune_list:
                                keep=True
                                pass
                            else:
                                root.remove(obj)
                        tree.write(xmlPath)  
                        if not keep:
                            os.remove(os.path.join(os.path.abspath(self.defaultSaveDir), item))
                            os.remove(os.path.join(os.path.abspath(os.path.dirname(self.filePath)), item[0:-4])+'.jpg')
                    QMessageBox.information(self,u'Done!',u'label pruning done.')
            except:
                return

    def question_2(self,ls):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'these {0} labels will remain ['.format(len(ls))
        for i in range(len(ls)):
            msg=msg+str(ls[i])+'  '
        msg=msg[0:-2]+'], others will be deleted, sure to continue?(personly advise you back up xml files)'
        return yes == QMessageBox.warning(self, u'Attention:', msg, yes | no)
    
    def remove_extra_img_xml(self):
        """remove img who has no corresponding xml or xml who has no corresponding img.
        after this actions, you may need reopen img folder.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:
            img_folder_path=os.path.dirname(self.filePath)
            xml_folder_path=self.defaultSaveDir
            imglist = sorted(os.listdir(img_folder_path))
            xmllist = sorted(os.listdir(xml_folder_path))
            for item in imglist:
                if item.endswith('.jpg') and (item[0:-4]+'.xml') in xmllist:
                    pass
                else:
                    os.remove(os.path.join(os.path.abspath(img_folder_path), item))

            imglist = os.listdir(img_folder_path)
            xmllist = os.listdir(xml_folder_path)
            for item in xmllist:
                if item.endswith('.xml') and (item[0:-4]+'.jpg') in imglist:
                    pass
                else:
                    os.remove(os.path.join(os.path.abspath(xml_folder_path), item))
            imglist = os.listdir(img_folder_path)
            xmllist = os.listdir(xml_folder_path)
            QMessageBox.information(self,u'Info!',u'done, now have {} imgs, and {} xmls'.format(len(imglist),len(xmllist)))
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def show_folder_infor(self):
        """show current img folder path and xml folder path and img's number and xml's number.
        usually img's amount should equal to xml's amount.
        """
        try:
            imglist = os.listdir(os.path.dirname(self.filePath))
            xmllist = os.listdir(self.defaultSaveDir)
            QMessageBox.information(self,u'Haa',u'img path: {0}\nimg nums: {1} imgs\nxml path: {2}\nxml nums: {3} xmls'.format(os.path.dirname(self.filePath),len(imglist),self.defaultSaveDir,len(xmllist)))
           
        except:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            
    def show_label_info(self):
        """show all label's name and it's box amount and img amount. 
        """
        def file_name(file_dir):
            L = []
            for root, dirs, files in os.walk(file_dir):
                for file in files:
                    if os.path.splitext(file)[1] == '.xml':
                        L.append(os.path.join(root, file))
            return L
        try:
            if self.filePath == None:
                QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
                return
            xml_dirs = file_name(self.defaultSaveDir)
            total_Box = 0;total_Pic = 0
            Class = []; box_num=[]; pic_num=[]; flag=[]

            for i in range(0, len(xml_dirs)):
                total_Pic+=1
                annotation_file = open(xml_dirs[i]).read()
                root = ET.fromstring(annotation_file)
                for obj in root.findall('object'):
                    label = obj.find('name').text
                    if label not in Class:
                        Class.append(label)
                        box_num.append(0)
                        pic_num.append(0)
                        flag.append(0)
                    for i in range(len(Class)):
                        if label == Class[i]:
                            box_num[i] += 1
                            flag[i] = 1
                            total_Box += 1
                for i in range(len(Class)):
                    if flag[i] == 1:
                        pic_num[i] += 1
                        flag[i] = 0
            result={}
            for i in range(len(Class)):
                result[Class[i]]= (pic_num[i], box_num[i])
            result['total']=(total_Pic, total_Box)
            info='label | pic_num | box_num \n'
            info += '----------------------------\n'
            for key in result.keys():
                info+='{}:  {}\n'.format(key,result[key])
            QMessageBox.information(self,u'Info',info)
            
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
    
    def extract_video(self):
        """extract imgs from video.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        img will saved in the same path with video.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            video_path,_ = QFileDialog.getOpenFileName(self,'choose video file:')
            if not video_path:
                return
            save_path=os.path.join(os.path.dirname(os.path.abspath(video_path)),os.path.realpath(video_path).split('.')[0])
            os.makedirs(save_path,exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    index=int(cap.get(1))
                    if index%frame_gap!=0:
                        continue
                    cv2.imwrite(save_path+'/'+str(int(cap.get(1)))+'.jpg',frame)
                else:
                    break
            cap.release()
            QMessageBox.information(self,u'Done!',u'video extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))

    def extract_stream(self):
        """extract imgs from stream, 'stream_path' usually start with rtsp or rtmp.
        'frame gap' means save img by this frequency(not save every img in video if frame_gap larger than 1).
        'max save number' means actions will stop after save this amount imgs.
        this action will stop after read stream path failed 3 times.
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            stream_path,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input steam path(start with rtmp、rtsp...):")
            if not(stream_path and ok):
                return
            save_path = QFileDialog.getExistingDirectory()
            if not save_path:
                return
            frame_gap,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input frame gap, img will extract by this frequency",value=1)
            if not ok:
                return
            max_frame,ok=QInputDialog.getInt(self, 'Int Input Dialog',
                "Input max save number, process will end after save cetain number imgs",value=10)
            if not ok:
                return
            cap = cv2.VideoCapture(stream_path)
            drop_times=0
            while True:
                ret, frame = cap.read()
                if ret:
                    index=int(cap.get(1))
                    if index%frame_gap!=0:
                        continue
                    if index>(max_frame*frame_gap):
                        break
                    cv2.imwrite(save_path+'/'+str(int(cap.get(1)))+'.jpg',frame)
                else:
                    cap.release()
                    drop_times+=1
                    if drop_times>=3:
                        QMessageBox.information(self,u'Wrong!',u'stream path not useable.')
                        break
                    cap = cv2.VideoCapture(stream_path)
            cap.release()
            QMessageBox.information(self,u'Done!',u'stream extract done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
           
    def batch_resize_img(self):
        """input Wdith and Height to resize all img to one shape.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return    
        try:
            img_path = os.path.dirname(self.filePath)
            filelist = natsort.natsorted(os.listdir(img_path))
            new_W,ok=QInputDialog.getInt(self,'Integer input dialog','input img wdith :',value=1920)
            if not ok:
                return
            new_H,ok=QInputDialog.getInt(self,'Integer input dialog','input img height :',value=1080)
            if not ok:
                return
            for item in filelist:
                img=cv2.imread(os.path.join(img_path,item))
                img=cv2.resize(img,(new_W,new_H))
                cv2.imwrite(os.path.join(img_path,item),img)
                
            QMessageBox.information(self,u'Done!',u'batch resize done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def merge_video(self):
        """merge all img in one path to one video, video will saved in img's parent path.
        for some restraint, fps must be 25, you can use 'repeat times' to repeat play img if you want slower the video.
        this action may take some time, please don't click mouse too frequently. 
        you can press 'space' if you find bounding box not accurate during auto annotate.
        """
        try:
            img_path = QFileDialog.getExistingDirectory(self,'choose imgs folder:')
            if not img_path:
                return
            filelist = natsort.natsorted(os.listdir(img_path)) #获取该目录下的所有文件名
            img=cv2.imread(img_path+'/'+filelist[0])
            img_size=img.shape
            fps = 25
            repeat_time,ok = QInputDialog.getInt(self, 'Int Input Dialog',
                        "Input each img's repeat times(the bigger, the slower), usually set 1",value=1)
            if not ok:
                return
            file_path = img_path +'_result' + ".avi" #导出路径
            fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
            video = cv2.VideoWriter( file_path, fourcc, fps ,(img_size[1],img_size[0]))
            for item in filelist:
                if item.endswith('.jpg'):   #判断图片后缀是否是.png
                    item = img_path +'/'+item 
                    img = cv2.imread(item)
                    for j in range(repeat_time):
                        video.write(img)        

            video.release()
            QMessageBox.information(self,u'Done!',u'video merge done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))

    def annotation_video(self):
        """ auto annotation video file or local camera.
        select video file, cancle to use local camera.
        img and xml will saved on dir of video path unless you use local camera, and folder will be './' in which case.
        'CSRT' type means more accuracy and low speed(recommend), 'MOSSE' means high speed and low accuracy, 'KCF' is in middle.
        frames are resized for display reason, one better run 'fix_property' after this process.
        press 'space' to re-drawing bounding box during annotation if you find bounding box not accurate.
        """
        try:
            tree = ET.ElementTree(file='./data/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)
            tree.write('./data/template.xml')
            trackerType_selector={'CSRT':cv2.TrackerCSRT_create,
                                  'BOOSTING':cv2.TrackerBoosting_create,
                                  'MIL':cv2.TrackerMIL_create,
                                  'KCF':cv2.TrackerKCF_create,
                                  'TLD':cv2.TrackerTLD_create,
                                  'MEDIANFLOW':cv2.TrackerMedianFlow_create,
                                  'GOTURN':cv2.TrackerGOTURN_create,
                                  'MOSSE':cv2.TrackerMOSSE_create}
            items=tuple(trackerType_selector)
            trackerType , ok = QInputDialog.getItem(self, "Select",
                "Tracker type, usually 'CSRT' is ok:", items, 0, False)
            if not ok:
                return
            videoPath ,_ = QFileDialog.getOpenFileName(self,"choose video file, cancle to use local camera:")
            if not videoPath:
                videoPath = 0
            save_gap,ok=QInputDialog.getInt(self,'Integer input dialog','input save gap, img will saved by this frenquency :',value=25)
            if not ok:
                return
            img_size,ok=QInputDialog.getInt(self,'Integer input dialog','input img size, img resized ti this shape by height:',value=900)
            if not ok:
                return
            process_shape=(int(1.777*img_size),int(img_size))
            cap = cv2.VideoCapture(videoPath)
            ret, frame = cap.read()
            height_K=frame.shape[0]/img_size
            weight_K=frame.shape[1]/(1.777*img_size)
            if not ret:
                print('Failed to read video')
                sys.exit(1)
            else:
                pass
                frame=cv2.resize(frame,process_shape)
            def init_multiTracker(frame):
                bboxes = []
                colors = []
                labels = []
                while True:
                    # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
                    bbox = cv2.selectROI("draw box and press 'SPACE' to affirm, Press 'q' to quit draw box and start tracking-labeling", frame)
                    if min(bbox[2],bbox[3]) >= 10:
                        label_name,ok=QInputDialog.getText(self, 'Text Input Dialog', 
                        "Input label name:")
                        if not(label_name and ok):
                            return
                        labels.append(label_name)
                        bboxes.append(bbox)
                        colors.append((random.randint(30, 240), random.randint(30, 240), random.randint(30, 240)))
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, [10,250,10], 2, 1)
                    else:
                        print("bbox size small than 10, will be abandoned")
                    k=cv2.waitKey(0)
                    print(k)
                    if k==113:
                        break
                print('Selected bounding boxes: {}'.format(bboxes))
                multiTracker = cv2.MultiTracker_create()
                # 初始化多跟踪器
                for bbox in bboxes:
                    tracker=trackerType_selector[trackerType]()
                    multiTracker.add(tracker, frame, bbox)    
                return multiTracker,colors,labels
            multiTracker,colors,labels=init_multiTracker(frame)
            cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
            cv2.moveWindow("MultiTracker", 10, 10)
            # 处理视频并跟踪对象
            index=0
            while cap.isOpened():
                ret, origin_frame = cap.read()
                if not ret:
                    break
                frame=cv2.resize(origin_frame,process_shape)
                draw=frame.copy()
                ret, boxes = multiTracker.update(frame)
                # 绘制跟踪的对象
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(draw, p1, p2, colors[i], 2, 1)
                    info = labels[i]
                    t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.7 , 1)[0]
                    cv2.rectangle(draw, p1, (int(newbox[0]) + t_size[0]+3, int(newbox[1]) + t_size[1]+6), colors[i], -1)
                    cv2.putText(draw, info, (int(newbox[0])+1, int(newbox[1])+t_size[1]+2), cv2.FONT_HERSHEY_TRIPLEX, 0.7, [255,255,255], 1)
                # show frame
                cv2.imshow("MultiTracker, press 'SPACE' to redraw box, press 'q' to quit video labeling", draw)
                # quit on ESC or Q button
                if index%save_gap==0:
                    tree = ET.ElementTree(file='./data/template.xml')
                    root=tree.getroot()
                    for i, newbox in enumerate(boxes):
                        temp_obj=template_obj
                        temp_obj.find('name').text=str(labels[i])
                        temp_obj.find('bndbox').find('xmin').text=str(int(weight_K*newbox[0]))
                        temp_obj.find('bndbox').find('ymin').text=str(int(height_K*newbox[1]))
                        temp_obj.find('bndbox').find('xmax').text=str(int(weight_K*newbox[0]+weight_K*newbox[2]))
                        temp_obj.find('bndbox').find('ymax').text=str(int(height_K*newbox[1]+height_K*newbox[3]))
                        root.append(deepcopy(temp_obj))       #深度复制
                    if videoPath==0:
                        parent_path='./temp'
                    else:
                        parent_path=os.path.dirname(videoPath)
                    os.makedirs(os.path.join(parent_path,'JPEGImages'), exist_ok=True)
                    os.makedirs(os.path.join(parent_path,'Annotations'), exist_ok=True)
                    cv2.imwrite(os.path.join(parent_path,'JPEGImages/','{}.jpg'.format(index)),origin_frame)
                    tree.write(os.path.join(parent_path,'Annotations/','{}.xml'.format(index)))
                index+=1
                k=cv2.waitKey(1)
                if k==32: #press space to reinit box
                    cv2.destroyAllWindows()
                    multiTracker,colors,labels=init_multiTracker(frame)
                    cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("MultiTracker", process_shape[0], process_shape[1])
                    cv2.moveWindow("MultiTracker", 10, 10)
                if k== 27 or k == 113: #press q or esc to quit
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            cap.release()
            cv2.destroyAllWindows()
            QMessageBox.information(self,u'Done!',u'video auto annotation done.')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
    
    def change_label_name(self):
        """change label name. from 'origin' to 'target'
        you can only change one label each time, not support multi-label changing.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return    
        try:
            label_transform={}
            origin, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input origin label name(only sigle label)：")
            if (not ok) and origin !='':
                return
            origin=origin.replace(" ","")
            target, ok=QInputDialog.getText(self, 'Text Input Dialog', "Input target label name(only sigle label)：")
            if (not ok) and target != '':
                return
            target=target.replace(" ","")
            label_transform[origin]=target
            xml_folder_path=self.defaultSaveDir
            img_folder_path=os.path.dirname(self.filePath)
            imglist = natsort.natsorted(os.listdir(img_folder_path))
            xmllist = natsort.natsorted(os.listdir(xml_folder_path))
            for item in xmllist:
                if item.endswith('.xml'):
                    if (item[0:-4]+'.jpg') in imglist:
                        xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                        imgPath=os.path.join(os.path.abspath(img_folder_path), item[0:-4])+'.jpg'
                        tree = ET.ElementTree(file=xmlPath)
                        root=tree.getroot()
                        for obj in root.findall('object'):
                            if obj.find('name').text in label_transform.keys():
                                obj.find('name').text=label_transform[obj.find('name').text]
                        tree.write(xmlPath)
                    else:
                        print(item,'has no corresponding img')
                        os.remove(os.path.join(os.path.abspath(xml_folder_path), item))
                        
            QMessageBox.information(self,u'Done!',u'label name changed!')
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def fix_xml_property(self):
        """fix xml's property such as size,folder,filename,path.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return 
        try:
            xml_folder_path=self.defaultSaveDir
            img_folder_path=os.path.dirname(self.filePath)
            xmllist = os.listdir(xml_folder_path)
            folder_info={'folder':'JPEGImages','filename':'none','path':'none'}
            for item in xmllist:
                if item.endswith('.xml'):
                    folder_info['filename']=item[0:-4]+'.jpg'
                    folder_info['path']=os.path.join(img_folder_path, item[0:-4])+'.jpg'
                    img = cv2.imread(folder_info['path'])
                    size=img.shape
                    xmlPath=os.path.join(os.path.abspath(xml_folder_path), item)
                    tree = ET.ElementTree(file=xmlPath)
                    root=tree.getroot()
                    try:
                        root.find('size').find('width').text=str(size[1])
                        root.find('size').find('height').text=str(size[0])
                        root.find('size').find('depth').text=str(size[2])
                    except:
                        print('xml has no size attribute!')
                    for key in folder_info.keys():
                        try:
                            root.find(key).text=folder_info[key]
                        except:
                            print(item,': attribute',key,'not exist!')
                            pass
                    tree.write(xmlPath)
            QMessageBox.information(self,u"Done!",u"fix xml's property done!")
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
     
    def yolov5_auto_labeling(self):
        try:
            tree = ET.ElementTree(file='./data/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)
            tree.write('./data/template.xml')
            #=====def some function=====
            def change_obj_property(detect_result,template_obj):
                temp_obj=template_obj
                for child in temp_obj:
                    key=child.tag
                    if key in detect_result.keys():
                        child.text=detect_result[key]
                    if key=='bndbox':
                        for gchild in child:
                            gkey=gchild.tag
                            gchild.text=str(detect_result[gkey])
                return temp_obj
                
            def change_result_type_yolov5(boxes,scores,labels):
                result=[]
                for box, score, label in zip(boxes, scores, labels):
                    if score>0.3:
                        try:
                            new_obj={}
                            new_obj['name']=label
                            new_obj['xmin']=int(box[0])
                            new_obj['ymin']=int(box[1])
                            new_obj['xmax']=int(box[2])
                            new_obj['ymax']=int(box[3])
                            result.append(new_obj)
                        except:
                            print('labels_info have no label: '+str(label))
                            pass
                return result

            source=os.path.dirname(self.filePath)
            xml_path=self.defaultSaveDir
            
            weight_path='pytorch_yolov5/weights'
            weight_list=[]
            for item in sorted(os.listdir(weight_path)):
                if item.endswith('.h5') or item.endswith('.pt') or item.endswith('.pth'):
                    weight_list.append(item)
            items = tuple(weight_list)
            if len(weight_list)>0 :
                weights, ok = QInputDialog.getItem(self, "Select",
                "Model weights file(weights file should under 'pytorch_yolov5/weights'):", items, 0, False)
                if not ok:
                    return
                else:
                    weights=os.path.join(weight_path,weights)
            else:
                weights,_ = QFileDialog.getOpenFileName(self,"'pytorch_yolov5/weights' is empty, choose model weights file:")
                if not (weights.endswith('.pt') or weights.endswith('.pth')):
                    QMessageBox.information(self,u'Wrong!',u'weights file must endswith .h5 or .pt or .pth')
                    return
            conf_thres=0.5
            iou_thres=0.5
            # Initialize
            device = torch_utils.select_device('0')
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model and label name.
            model = attempt_load(weights, map_location=device)  # load FP32 model
            names = model.module.names if hasattr(model, 'module') else model.names
            needed_labels=easygui.multchoicebox(msg="select labels you want auto-labeing?",title="Setect labels",choices=tuple(names))
            # set imsize
            imgsz,OK=QInputDialog.getInt(self,'Integer input dialog','input img size(k*32):',value=640)
            if not OK:
                return
            imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
            if half:
                model.half()  # to FP16
            
            # load img and run inference
            dataset = LoadImages(source, img_size=imgsz)
            progress = QProgressDialog(self)
            progress.setWindowTitle(u"Waiting")  
            progress.setLabelText(u"auto-labeling with yolov5 now,Please wait...")
            progress.setCancelButtonText(u"Cancle it")
            progress.setMinimumDuration(1)
            progress.setWindowModality(Qt.WindowModal)
            progress.setRange(0,100)   
            img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            index=-1
            for path, img, im0s, vid_cap in dataset:
                index+=1
                progress.setValue(int(100*index/len(dataset)))
                if progress.wasCanceled():
                    QMessageBox.warning(self,"Attention","auto-labeling canceled！") 
                    return
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
                t2 = torch_utils.time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    labels=[]
                    scores=[]
                    boxes=[]
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        # Write results
                        for *xyxy, conf, cls in det:
                            if names[int(cls)] not in needed_labels:
                                continue
                            labels.append(names[int(cls)])
                            scores.append(conf.item())
                            boxes.append([int(xyxy[0].item()),int(xyxy[1].item()),int(xyxy[2].item()),int(xyxy[3].item())])
                    tree = ET.ElementTree(file='./data/template.xml')
                    root=tree.getroot()
                    common_property={'filename':path.split('\\')[-1],'path':source,'folder':'JPEGImages'}
                    for child in root:
                        key=child.tag
                        if key in common_property.keys():
                            child.text=common_property[key]
                    result=change_result_type_yolov5(boxes,scores,labels)
                    if len(result)>0:
                        for j in range(len(result)):
                            new_obj=change_obj_property(result[j],template_obj)
                            root.append(deepcopy(new_obj))       #深度复制
                            #!!!这块没直接append(new_obj)是因为当增加多个节点的话，new_obj会进行覆盖，必须要用深度复制以进行区分
                    tree.write(os.path.join(xml_path,path[len(os.path.dirname(path))+1:-4]+'.xml'))
            progress.setValue(100)
            QMessageBox.information(self,u'Done!',u'auto labeling done, please reload img folder')  
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
                 
    def auto_labeling(self):
        """use model to labeling unannotated imgs.
        you should choose model type as well as model weights file and input label name.
        supported model type is 'Yolov5' or 'Retinanet', more type will updating later.
        'label name' must sorted by class number, if you do not remenber them, just press 'Enter', 
        box will named by its class number and you can change them one by one using action <change_label>
        for a recorde, yolov5 do not need 'label name', but 'img size' additionally.
        this action may take some time, please don't click mouse too frequently.
        """
        if self.filePath==None:
            QMessageBox.information(self,u'Wrong!',u'have no loaded folder yet, please check again.')
            return
        try:       
            #=====choose model and input label name=====
            # using yolov5 autolabeling   
            with torch.no_grad():
                self.yolov5_auto_labeling()
            return
            # using retinanet autolabeling, not recommended.
            """   
            from keras_retinanet import models
            from keras_retinanet.utils.image import preprocess_image, resize_image
            from keras_retinanet.utils.visualization import remove_certain_label,transform_box
            from keras_retinanet.utils.gpu import setup_gpu
            setup_gpu(0)
            img_path = os.path.dirname(self.filePath)
            xml_path = self.defaultSaveDir
            weight_path='keras_retinanet/weights'
            weight_list=[]
            for item in sorted(os.listdir(weight_path)):
                if item.endswith('.h5') or item.endswith('.pt') or item.endswith('.pth'):
                    weight_list.append(item)
            items = tuple(weight_list)
            if len(weight_list)>0 :
                model_path, ok = QInputDialog.getItem(self, "Select",
                "Model weights file(weights file should under 'keras_retinanet/weights'):", items, 0, False)
                if not ok:
                    return
                else:
                    model_path=os.path.join(weight_path,model_path)
            else:
                model_path,_ = QFileDialog.getOpenFileName(self,"'keras_retinanet/weights' is empty, choose model weights file:")
                if not model_path:
                    return
                if not (model_path.endswith('.h5') or model_path.endswith('.pt') or model_path.endswith('.pth')):
                    QMessageBox.information(self,u'Wrong!',u'weights file must endswith .h5 or .pt or .pth')
                    return
            model = models.load_model(model_path, backbone_name='resnet50')
            draw_info={}
            draw_info['label_name']={}
            draw_info['box_type']=['xmin','ymin','xmax','ymax']
            text, ok=QInputDialog.getText(self, 'Text Input Dialog', 
                                "Input label name sorted by class number(split by ',')：")
            if not ok:
                return
            text='0' if text=='' else text
            text=text.replace(" ","")
            i=0
            for item in text.split(','):
                draw_info['label_name'][i]=item
                i+=1
            print(draw_info['label_name'])
            #=====create template=====
            tree = ET.ElementTree(file='./data/origin.xml')
            root=tree.getroot()
            for child in root.findall('object'):
                template_obj=child#保存一个物体的样板
                root.remove(child)
            tree.write('./data/template.xml')
            #=====def some function=====
            def change_obj_property(detect_result,template_obj):
                temp_obj=template_obj
                for child in temp_obj:
                    key=child.tag
                    if key in detect_result.keys():
                        child.text=detect_result[key]
                    if key=='bndbox':
                        for gchild in child:
                            gkey=gchild.tag
                            gchild.text=str(detect_result[gkey])
                return temp_obj
                
            def change_result_type(boxes,scores,labels):
                result=[]
                for box, score, label in zip(boxes, scores, labels):
                    if score>0.5:
                        try:
                            new_obj={}
                            name=draw_info['label_name'][label] if label in draw_info['label_name'].keys() else str(label)
                            new_obj['name']=name
                            new_obj['xmin']=int(box[0])
                            new_obj['ymin']=int(box[1])
                            new_obj['xmax']=int(box[2])
                            new_obj['ymax']=int(box[3])
                            result.append(new_obj)
                        except:
                            print('labels_info have no label: '+str(label))
                            pass
                return result

            def retina_detect(img,img_size,detect_window=None):
                try:
                    image = img[detect_window[1]:detect_window[3],detect_window[0]:detect_window[2],:].copy()
                except:
                    image = img.copy()
                image = preprocess_image(image)
                image, scale = resize_image(image, min_side=img_size[0], max_side=img_size[1])
                boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
                boxes=boxes[0]
                scores=scores[0]
                labels=labels[0]
                boxes /= (scale/2)
                labels,scores,boxes=remove_certain_label(labels,scores,boxes,[-1])
                boxes=transform_box(boxes,detect_window)
                return labels,scores,boxes

            image_names = sorted(os.listdir(img_path))
            for i in range(len(image_names)):
                if i%50==0:
                    print('now at:',i,'| time:',datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                tree = ET.ElementTree(file='./data/template.xml')
                root=tree.getroot()
                common_property={'filename':image_names[i],'path':img_path,'folder':'JPEGImages'}
                for child in root:
                    key=child.tag
                    if key in common_property.keys():
                        child.text=common_property[key]
                img = cv2.imread(img_path + '/' + image_names[i])
                labels,scores,boxes=retina_detect(img,[411,700])
                result=change_result_type(boxes,scores,labels)
                if len(result)>0:
                    for j in range(len(result)):
                        new_obj=change_obj_property(result[j],template_obj)
                        root.append(deepcopy(new_obj))       #深度复制
                        #!!!这块没直接append(new_obj)是因为当增加多个节点的话，new_obj会进行覆盖，必须要用深度复制以进行区分
                tree.write(xml_path+'/'+image_names[i][0:-4]+'.xml')
            QMessageBox.information(self,u'Done!',u'auto labeling done, please reload img folder')
            """
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
        
    def data_auto_agument(self):
        """data agument, using Affine change, intensity change, contrast change, gama change, Gaussian fillter to agument img data.
        you can select agument multiple(1~4).
        this action may take some time, please don't click mouse too frequently.
        """
        try:
            self.xml_folder_path=self.defaultSaveDir
            self.img_folder_path=os.path.dirname(self.filePath)
            imglist = natsort.natsorted(os.listdir(self.img_folder_path))
            xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
            print(len(imglist),len(xmllist))
            if len(imglist) != len(xmllist):
                QMessageBox.information(self,u'Wrong!',u'lens of img and xml do not equal.')
                return 
            else:
                magnification,OK=QInputDialog.getInt(self,'Integer input dialog','input agument magnification(1~4):',value=4)
                if OK:
                    if magnification<1 or magnification>4:
                        magnification=4
                    img_Temp=[]
                    xml_Temp=[]
                    for i in range(magnification):
                        img_Temp.extend(imglist)
                        xml_Temp.extend(xmllist)
                    one_step=int(len(img_Temp)/4)
                    imglist1=img_Temp[0:one_step];xmllist1=xml_Temp[0:one_step]
                    imglist2=img_Temp[one_step:2*one_step];xmllist2=xml_Temp[one_step:2*one_step]
                    imglist3=img_Temp[2*one_step:3*one_step];xmllist3=xml_Temp[2*one_step:3*one_step]
                    imglist4=img_Temp[3*one_step:];xmllist4=xml_Temp[3*one_step:]
                    progress = QProgressDialog(self)
                    progress.setWindowTitle(u"Waiting")  
                    progress.setLabelText(u"Processing now,Please wait...")
                    progress.setCancelButtonText(u"Cancle data agument")
                    progress.setMinimumDuration(1)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setRange(0,100) 
                    
                    self.agument_A(imglist1,xmllist1,progress)
                    self.agument_B(imglist2,xmllist2,progress)
                    self.agument_C(imglist3,xmllist3,progress)
                    self.agument_D(imglist4,xmllist4,progress)
                    imglist = natsort.natsorted(os.listdir(self.img_folder_path))
                    xmllist = natsort.natsorted(os.listdir(self.xml_folder_path))
                    self.exam_agument(xmllist,progress)
                    
                    progress.setValue(100)
                    QMessageBox.information(self,"Done","data agument scuesseed！")
            
        except Exception as e:
            QMessageBox.information(self,u'Sorry!',u'something is wrong. ({})'.format(e))
            
    def agument_A(self,imglist,xmllist,progress):
        print('agmt:',len(imglist))
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                new_img1=cv2.flip(img,1,dst=None)
                shift_X=np.random.randint(-0.15*size[1], 0.15*size[1])
                shift_Y=np.random.randint(-0.15*size[0], 0.15*size[0])
                shift_info.append([shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(new_img1, M, (new_img1.shape[1], new_img1.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-8,8,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtA_'+item
                cv2.imwrite(save_path,new_img)

        for i in range(len(xmllist)):
            progress.setValue(17+3*(i/(len(xmllist))))
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtA_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtA_'+item[0:-4]+'.jpg'
                for child in root:
                    if child.tag=='object':
                        for gchild in child:
                            if gchild.tag=='bndbox':
                                temp=gchild[0].text
                                gchild[0].text=str(size[1]-int(gchild[2].text)+shift_info[i][0])
                                gchild[1].text=str(int(gchild[1].text)+shift_info[i][1])
                                gchild[2].text=str(size[1]-int(temp)+shift_info[i][0])
                                gchild[3].text=str(int(gchild[3].text)+shift_info[i][1])
                tree.write(self.xml_folder_path+'/agmtA_'+item)
        print('agument_A done!')

    def agument_B(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(20+17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                result=99*np.ones(img.shape)
                k=random.uniform(0.5,0.7) #根据实际需求更改范围，小于1为缩小，大于1为放大
                small = cv2.resize(img, (0,0), fx = k, fy = k, interpolation = cv2.INTER_AREA)
                result[0:small.shape[0],0:small.shape[1],:]=small
                shift_X=np.random.randint(-0.1*size[1], 0.3*size[1])
                shift_Y=np.random.randint(-0.1*size[0], 0.3*size[0])
                shift_info.append([k,shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) 
                shifted = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-8,8,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtB_'+item
                cv2.imwrite(save_path,new_img)

        for i in range(len(xmllist)):
            progress.setValue(37+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                xmlPath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                tree = ET.ElementTree(file=xmlPath)
                root=tree.getroot()
                root.find('filename').text='agmtB_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtB_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(shift_info[i][0]*ymin+shift_info[i][2]))
                    child.find('bndbox').find('ymax').text=str(int(shift_info[i][0]*ymax+shift_info[i][2]))
                    child.find('bndbox').find('xmin').text=str(int(shift_info[i][0]*xmin+shift_info[i][1]))
                    child.find('bndbox').find('xmax').text=str(int(shift_info[i][0]*xmax+shift_info[i][1]))
                tree.write(self.xml_folder_path+'/agmtB_'+item)
        print('agument_B done!')

    def agument_C(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(40+17*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                a=int(2*random.randint(1,3)+1)
                b=random.uniform(11,21)
                blur = cv2.GaussianBlur(img,(a,a),b)
                shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
                shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
                shift_info.append([shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(blur, M, (blur.shape[1], blur.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-5,5,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtC_'+item
                cv2.imwrite(save_path,new_img)

        for i in range(len(xmllist)):
            progress.setValue(57+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtC_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtC_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
                    child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
                    child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
                    child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
                tree.write(self.xml_folder_path+'/agmtC_'+item)
        print('agument_C done!')

    def agument_D(self,imglist,xmllist,progress):
        shift_info=[]
        for i in range(len(imglist)):
            progress.setValue(60+27*(i/(len(imglist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=imglist[i]
            if item.endswith('.jpg'):
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item)
                img = cv2.imread(imgPath)
                size=img.shape
                k=random.uniform(0.6,1.3)
                gama=exposure.adjust_gamma(img,k)
                shift_X=np.random.randint(-0.1*size[1], 0.1*size[1])
                shift_Y=np.random.randint(-0.1*size[0], 0.1*size[0])
                shift_info.append([shift_X,shift_Y])
                M = np.float32([[1, 0, shift_X], [0, 1, shift_Y]]) #13
                shifted = cv2.warpAffine(gama, M, (gama.shape[1], gama.shape[0]),borderValue=(99,99,99))
                noise=np.random.randint(-5,5,size=[size[0],size[1],3])
                new_img=shifted+noise
                save_path=self.img_folder_path+'/agmtD_'+item
                cv2.imwrite(save_path,new_img)
        for i in range(len(xmllist)):
            progress.setValue(87+3*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            item=xmllist[i]
            if item.endswith('.xml'):
                filePath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                imgPath=os.path.join(os.path.abspath(self.img_folder_path), item[0:-4])+'.jpg'
                img = cv2.imread(imgPath)
                size=img.shape
                tree = ET.ElementTree(file=filePath)
                root=tree.getroot()
                root.find('filename').text='agmtD_'+item[0:-4]+'.jpg'
                root.find('path').text=self.img_folder_path.replace('\\','/')+'/agmtD_'+item[0:-4]+'.jpg'
                for child in root.findall('object'):
                    ymin=int(child.find('bndbox').find('ymin').text)
                    ymax=int(child.find('bndbox').find('ymax').text)
                    xmin=int(child.find('bndbox').find('xmin').text)
                    xmax=int(child.find('bndbox').find('xmax').text)
                    child.find('bndbox').find('ymin').text=str(int(ymin+shift_info[i][1]))
                    child.find('bndbox').find('ymax').text=str(int(ymax+shift_info[i][1]))
                    child.find('bndbox').find('xmin').text=str(int(xmin+shift_info[i][0]))
                    child.find('bndbox').find('xmax').text=str(int(xmax+shift_info[i][0]))
                tree.write(self.xml_folder_path+'/agmtD_'+item)
        print('agument_D done!')
        
    def IOU(self,rectA, rectB):  
        W = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
        H = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])
        if W <= 0 or H <= 0:
            return 0;
        SA = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
        SB = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
        cross = W * H
        return cross/(SA + SB - cross)

    def exam_bndbox_is_lawful(self,window=[0,0,100,100],bndbox=[10,10,20,30],min_IOU=0.002,max_IOU=1):
        '''
        判断bndbox是否在图片范围window内，若全部在图片范围外，删掉box；对于有部分在图片内的，将box图片范围外的部分裁剪掉，
        同时若裁剪后IOU过小，同样删除box
        '''
        if min_IOU<self.IOU(window,bndbox)<=max_IOU:
            is_lawful=True
            for i in range(2):
                if bndbox[i]<window[i]:
                    bndbox[i]=window[i]
            for j in range(2,4):
                if bndbox[j]>window[j]:
                    bndbox[j]=window[j]
        else:
            is_lawful=False
        return is_lawful,bndbox
        
    def exam_agument(self,xmllist,progress):
        for i in range(len(xmllist)):
            progress.setValue(90+9*(i/(len(xmllist)))) 
            if progress.wasCanceled():
                QMessageBox.warning(self,"Attention","agument failed, please check floder！") 
                break
            
            item=xmllist[i]
            if item.endswith('.xml'):
                xmlPath=os.path.join(os.path.abspath(self.xml_folder_path), item)
                tree = ET.ElementTree(file=xmlPath)
                root=tree.getroot()

                a=int(root.find('size').find('width').text)
                b=int(root.find('size').find('height').text)
                window=[0,0,a,b]

                for child in root.findall('object'):
                    a=int(child.find('bndbox').find('xmin').text)
                    b=int(child.find('bndbox').find('ymin').text)
                    c=int(child.find('bndbox').find('xmax').text)
                    d=int(child.find('bndbox').find('ymax').text)
                    bndbox=[min(a,c),min(b,d),max(a,c),max(b,d)]
                    is_lawful,new_bndbox=self.exam_bndbox_is_lawful(window=window,bndbox=bndbox)
                    if not is_lawful:
                        root.remove(child)
                    else:
                        child.find('bndbox').find('xmin').text=str(new_bndbox[0])
                        child.find('bndbox').find('ymin').text=str(new_bndbox[1])
                        child.find('bndbox').find('xmax').text=str(new_bndbox[2])
                        child.find('bndbox').find('ymax').text=str(new_bndbox[3])
                    
                tree.write(xmlPath)
        print('exam_agument done!')

def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'),
                     argv[3] if len(argv) >= 4 else None)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())
