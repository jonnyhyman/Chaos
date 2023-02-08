"""
    Written by Jonny Hyman, 2020
        www.jonnyhyman.com
        www.github.com/jonnyhyman

    MIT License

    This code shows three ways of understanding the logistic
    map : cobweb plot, time series, and bifurcation plot

"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout,
                             QLabel, QSizePolicy, QSlider, QSpacerItem,
                             QVBoxLayout, QWidget)

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from numba import jit, prange, njit
import numpy as np

from pathlib import Path
import sys

from time import time
import types

colors =  {
            'lightest':"#eeeeee",
            'lighter':"#e5e5e5",
            'light':"#effffb",
            'himid':"#50d890",
            'midmid':"#1089ff",
            'lomid':"#4f98ca",
            'dark' :"#272727",
            'darker' :"#23374d",
}

numlabelsize = 22
txtlabelsize = 20

# STYLING
numfont = QtGui.QFont("Avenir Next") # requires macos
numfont.setPixelSize(numlabelsize)

txtfont = QtGui.QFont("Avenir Next") # requires macos
txtfont.setPixelSize(txtlabelsize)

brushes = { k: pg.mkBrush(c) for k, c in colors.items() }

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', colors['dark'])
pg.setConfigOption('foreground', colors['light'])

QPushButton_style = f"""
QPushButton{{
	color: {colors['light']};
	background-color: transparent;
	border: 1px solid #4589b2;
	padding: 5px;

}}

QPushButton::hover{{
	background-color: rgba(255,255,255,.2);
}}

QPushButton::pressed{{
	border: 1px solid {colors['himid']};
	background-color: rgba(0,0,0,.3);
}}"""

QLabel_style = f"""
QLabel{{
    color: {colors['light']};
}}
"""

QCheckBox_style = f"""
QCheckBox{{
    background-color: {colors['darker']};
    color: {colors['light']};
    padding:5px;
}}
"""

def custom_axis_item_resizeEvent(self, ev=None):
    """ custom implementation of AxisItem.resizeEvent to control `nudge`

        this overwrites the instance method for `AxisItem`
    """

    #s = self.size()

    ## Set the position of the label
    nudge = 15

    br = self.label.boundingRect()
    p = QtCore.QPointF(0, 0)
    if self.orientation == 'left':
        p.setY(int(self.size().height()/2 + br.width()/2))
        p.setX(-nudge)
    elif self.orientation == 'right':
        p.setY(int(self.size().height()/2 + br.width()/2))
        p.setX(int(self.size().width()-br.height()+nudge))
    elif self.orientation == 'top':
        p.setY(-nudge)
        p.setX(int(self.size().width()/2. - br.width()/2.))
    elif self.orientation == 'bottom':
        p.setX(int(self.size().width()/2. - br.width()/2.))
        p.setY(int(self.size().height()-br.height()+nudge))
    self.label.setPos(p)
    self.picture = None

@njit
def linear_interp(x, in_min, in_max, out_min, out_max, lim=True, dec=None):
    y = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    if dec is not None:
        y = round(y, dec)

    if not lim:
        return y

    else:
        y = max(out_min, y)
        y = min(out_max, y)

        return y

@jit(cache=True, nopython=True) # pragma: no cover
def logistic_map(pop, rate):
    """
    From pynamical `https://github.com/gboeing/pynamical`
    Define the equation for the logistic map.

    Arguments
    ---------
    pop: float
        current population value at time t
    rate: float
        growth rate parameter values

    Returns
    -------
    float
        scalar result of logistic map at time t+1
    """

    return pop * rate * (1 - pop)



def get_function_points(model, r, n, start, end):
    """
    From pynamical
    Calculate model results for n population values evenly spaced between start and end values.

    Arguments
    ---------
    model: function
        defining an iterated map to simulate
    r: float
        growth rate parameter value to pass to the map
    n: int
        number of iterations to run
    start: float
        lower limit of the function range
    end: float
        upper limit of the function range

    Returns
    -------
    tuple
        x_vals, y_vals
    """

    x_vals = np.linspace(start, end, n)
    y_vals = [model(x, r) for x in x_vals]
    return x_vals, y_vals

def get_cobweb_points(model, r, x, n):
    """
    From pynamical `https://github.com/gboeing/pynamical`
    Calculate the vertices of cobweb lines for a cobweb plot.

    Steps in the calculation:
    1) Let x = 0.5
    2) Start on the x-axis at the point (x, 0).
    3) Draw a vertical line to the red function curve: this point has the coordinates (x, f(x)).
    4) Draw a horizontal line from this point to the gray diagonal line: this point has the coordinates (f(x), f(x)).
    5) Draw a vertical line from this point to the red function curve: this point has the coordinates (f(x), f(f(x))).
    6) Repeat steps 4 and 5 recursively one hundred times.

    Arguments
    ---------
    model: function
        defining an iterated map to simulate
    r: float
        growth rate parameter value to pass to the map
    x: float
        starting population value
    n: int
        number of iterations to run

    Returns
    -------
    tuple
        cobweb_x_vals, cobweb_y_vals
    """

    cobweb_points = [(x, 0)]
    for _ in range(n):
        y1 = model(x, r)
        cobweb_points.append((x, y1))
        cobweb_points.append((y1, y1))
        y2 = model(y1, r)
        cobweb_points.append((y1, y2))
        x = y1
    return zip(*cobweb_points)

def cobweb_plot(plt, idx=-1,
                model=logistic_map, r=0, cobweb_x=0.5,

                function_n=1000,

                cobweb_n=100, num_discard=0,
                title='', filename='', show=True, save=True,
                start=0, end=1, figsize=(6,6), diagonal_linewidth=1.35,
                cobweb_linewidth=1, function_linewidth=1.5,
                folder='images', dpi=300, bbox_inches='tight', pad=0.1):

    plt.clear()

    initial_pop = float(cobweb_x)

    stride_idx = (idx-1)*3

    func_x_vals, func_y_vals = get_function_points(
                        model=model, r=r, n=function_n, start=start, end=end)
    cobweb_x_vals, cobweb_y_vals = get_cobweb_points(
                        model=model, r=r, x=cobweb_x, n=cobweb_n)

    cobweb_x_vals = cobweb_x_vals[:stride_idx]
    cobweb_y_vals = cobweb_y_vals[:stride_idx]

    plt.setTitle(f"Cobweb Plot")

    diagonal_line = plt.plot((0,1), (0,1))
    diagonal_line.setPen(width=diagonal_linewidth)

    function_line = pg.PlotDataItem(x=func_x_vals, y=func_y_vals)
    function_line.setPen(color=colors['lomid'], width=3.0)
    function_line = plt.addItem(function_line)

    sizes = 1/np.linspace(.1,1,len(cobweb_x_vals))
    sizes = np.maximum(5, sizes)

    cobweb_line = plt.plot(cobweb_x_vals, cobweb_y_vals, symbol='o', symbolSize=sizes)
    cobweb_line.setPen(color=colors['lomid'], width=cobweb_linewidth)
    cobweb_line.setSymbolPen(color=(1,1,1,0), width=0.0)
    cobweb_line.setSymbolBrush(color=colors['himid'])

    xaxis = plt.getAxis("bottom")
    #xaxis.setTickSpacing(4,4)

    xaxis.tickFont = numfont
    xaxis.setStyle(tickTextOffset = 20)

    yaxis = plt.getAxis("left")
    #yaxis.setTickSpacing(.2, .2)

    yaxis.resizeEvent = types.MethodType(custom_axis_item_resizeEvent, yaxis)

    yaxis.tickFont = numfont
    yaxis.setStyle(tickTextOffset = 10, tickLength=10)

    xaxis.label.setFont(txtfont)
    yaxis.label.setFont(txtfont)
    plt.titleLabel.item.setFont(txtfont)

    return np.concatenate(([initial_pop,], np.array(cobweb_y_vals[1::3])))

def series_plot(plt, y_vals, idx=100, r=0, xall=False):

    plt.clear()
    plt.setTitle("Population vs Time, growth rate: {:4.2f}".format(r))

    t = np.arange(len(y_vals))
    y = y_vals[:idx]
    t = t[:idx]

    if len(t) < 20:
        plt.setXRange(*[0,20])
    else:
        if not xall:
            plt.setXRange(*[len(t)-20, len(t)])
        else:
            plt.setXRange(*[0, 30])

    plt.setYRange(*[0,1])
    plt.showGrid(x=True, y=True)

    s = 20/len(t)
    s = max(10,s)

    line = plt.plot(x=t, y=y)
    line.setPen(color=colors['lomid'], width=3.0)

    scat = pg.ScatterPlotItem(x=t, y=y, size=s, name='series')
    scat.setPen(color=(1,1,1,0), width=0.0)
    scat.setBrush(color=colors['himid'])
    plt.addItem(scat)

    xaxis = plt.getAxis("bottom")
    #xaxis.setTickSpacing(4,4)

    xaxis.tickFont = numfont
    xaxis.setStyle(tickTextOffset = 20)

    yaxis = plt.getAxis("left")
    #yaxis.setTickSpacing(.2, .2)

    yaxis.resizeEvent = types.MethodType(custom_axis_item_resizeEvent, yaxis)

    yaxis.tickFont = numfont
    yaxis.setStyle(tickTextOffset = 10, tickLength=10)

    xaxis.label.setFont(txtfont)
    yaxis.label.setFont(txtfont)
    plt.titleLabel.item.setFont(txtfont)

def bifurc_plot(plt, y_vals, r=0, ipop=0.5, discard=64):

    if (y_vals.shape[0]) < discard:
        return

    ys = y_vals[discard:]
    xs = np.repeat(r,len(ys))
    s = 1
    new_s = 5

    if len(plt.items) == 0:
        # __init__
        bifurcation = pg.ScatterPlotItem(x=xs, y=ys, size=new_s, antialias=True, name='bifurc')
        bifurcation.setBrush(color=colors['himid'], alpha=0.5, width=1)
        bifurcation.setPen(color=(0,0,0,0), width=0, alpha=0.5)
        plt.addItem(bifurcation)

        plt.titleLabel.item.setFont(txtfont)

        xaxis = plt.getAxis("bottom")
        xaxis.setTickSpacing(1, .5)

        xaxis.tickFont = numfont
        xaxis.setStyle(tickTextOffset = 20)

        yaxis = plt.getAxis("left")
        #yaxis.setTickSpacing(.2, .2)

        yaxis.tickFont = numfont
        yaxis.setStyle(tickTextOffset = 10, tickLength=10)

        xaxis.label.setFont(txtfont)
        yaxis.label.setFont(txtfont)

    else:
        # __update__
        bifurcation = plt.items[0]

        n_new = len(xs)

        xs = np.concatenate((xs, bifurcation.data['x']))
        ys = np.concatenate((ys, bifurcation.data['y']))

        ss = np.repeat(1,len(xs))
        ss[0:n_new] = new_s

        bs = [brushes['lomid'] for b in range(len(xs))]
        bs[0:n_new] = [brushes['himid'] for _ in range(n_new)]

        bifurcation.setData(xs,ys, size=ss, brush=bs)

    plt.setXRange(*[0,4])
    plt.setYRange(*[0,1])

class Controls(QWidget):
    def __init__(self, variable='', parent=None):
        super(Controls, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)

        self.l1 = QVBoxLayout()
        self.l1.setAlignment(Qt.AlignTop)

        self.cobweb_box = QtWidgets.QCheckBox('🕸️ Plot', parent=self)
        self.l1.addWidget(self.cobweb_box)
        self.cobweb_box.setChecked(0)

        self.series_box = QtWidgets.QCheckBox("📈 Plot", parent=self)
        self.l1.addWidget(self.series_box)
        self.series_box.setChecked(1)

        self.bifurc_box = QtWidgets.QCheckBox("❄️ Plot", parent=self)
        self.l1.addWidget(self.bifurc_box)
        self.bifurc_box.setChecked(0)

        self.verticalLayout.addLayout(self.l1)

        self.l2 =  QVBoxLayout()
        self.l2.setAlignment(Qt.AlignTop)

        self.animlabel = QLabel(self)
        self.l2.addWidget(self.animlabel)
        self.animlabel.setText("Animation FPS")

        self.animrate = QtWidgets.QSlider(self)
        self.animrate.setOrientation(Qt.Horizontal)
        self.l2.addWidget(self.animrate)
        self.animrate.setFixedWidth(self.animrate.width())

        self.verticalLayout.addLayout(self.l2)

        self.l3 =  QVBoxLayout()
        self.l3.setAlignment(Qt.AlignTop)

        self.sub1 = QHBoxLayout()
        self.sub1.setAlignment(Qt.AlignLeft)
        self.rate_txt = QLabel("Rate", self)

        self.rate_box = QtWidgets.QDoubleSpinBox(self)
        self.rate_box.setSingleStep(0.01)
        self.rate_box.setDecimals(2)

        self.sub1.addWidget(self.rate_txt)
        self.sub1.addWidget(self.rate_box)

        self.l3.addLayout(self.sub1)

        self.rate = QSlider(self)
        self.rate.setOrientation(Qt.Horizontal)
        self.l3.addWidget(self.rate)

        self.verticalLayout.addLayout(self.l3)

        self.l4 =  QVBoxLayout()
        self.l4.setAlignment(Qt.AlignTop)

        self.sub2 = QHBoxLayout()
        self.sub2.setAlignment(Qt.AlignLeft)

        self.ipoptxt = QLabel(self)
        self.sub2.addWidget(self.ipoptxt)
        self.ipoptxt.setAlignment(Qt.AlignHCenter)
        self.ipoptxt.setText("Initial\nPop.")

        self.ipop_box = QtWidgets.QDoubleSpinBox()
        self.ipop_box.setSingleStep(0.01)
        self.ipop_box.setDecimals(2)
        self.ipop_box.setValue(0.5)
        self.sub2.addWidget(self.ipop_box)

        self.l4.addLayout(self.sub2)

        self.ipop = QSlider(self)
        self.ipop.setOrientation(Qt.Horizontal)
        self.l4.addWidget(self.ipop)
        self.ipop.setValue(50)

        self.verticalLayout.addLayout(self.l4)

        self.resize(self.sizeHint())

        self.setFixedWidth(self.animrate.width() + 20)

        self.rate.valueChanged.connect(lambda: self.setValues('rate_slider'))
        self.rate_box.valueChanged.connect(lambda: self.setValues('rate_box'))
        self.ipop.valueChanged.connect(lambda: self.setValues('ipop_slider'))
        self.ipop_box.valueChanged.connect(lambda: self.setValues('ipop_box'))

        self.rateval = None

        self.l5 =  QVBoxLayout()
        self.l5.setAlignment(Qt.AlignTop)

        self.animb = QtWidgets.QPushButton('Play', parent=self)
        self.l5.addWidget(self.animb)
        self.animb.setFixedWidth(self.animlabel.width())

        self.reset = QtWidgets.QPushButton('Reset', parent=self)
        self.l5.addWidget(self.reset)
        self.reset.setFixedWidth(self.animlabel.width())

        self.clear = QtWidgets.QPushButton('Clear', parent=self)
        self.clear.setCheckable(False)
        self.clear.setFixedWidth(self.animlabel.width())
        self.l5.addWidget(self.clear)

        self.verticalLayout.addLayout(self.l5)

        # STYLING
        self.clear.setStyleSheet(QPushButton_style)
        self.animb.setStyleSheet(QPushButton_style)
        self.reset.setStyleSheet(QPushButton_style)

        self.animlabel.setStyleSheet(QLabel_style)
        self.rate_txt.setStyleSheet(QLabel_style)
        self.ipoptxt.setStyleSheet(QLabel_style)

        self.cobweb_box.setStyleSheet(QCheckBox_style)
        self.series_box.setStyleSheet(QCheckBox_style)
        self.bifurc_box.setStyleSheet(QCheckBox_style)

    def resizeEvent(self, event):
        super(Controls, self).resizeEvent(event)

    def setValues(self, kind=''):

        if kind == 'rate_slider':
            self.rateval = linear_interp(self.rate.value(),
                                            0, 99, 0, 3.99, lim=1, dec=2)
            self.rate_box.blockSignals(True)
            self.rate_box.setValue(self.rateval)
            self.rate_box.blockSignals(False)

        elif kind == 'rate_box':
            self.rateval = linear_interp(self.rate_box.value(),
                                            0, 3.99, 0, 3.99, lim=1, dec=2)
            self.rate.blockSignals(True)
            self.rate.setValue( int(linear_interp(self.rateval,
                                            0, 3.99, 0, 99, lim=1, dec=2)) )
            self.rate.blockSignals(False)

        elif kind == 'ipop_slider':
            self.ipopval = linear_interp(self.ipop.value(),
                                            0, 100, 0, 1, lim=1)


            self.ipop_box.blockSignals(True)
            self.ipop_box.setValue(float(self.ipopval))
            self.ipop_box.blockSignals(False)


        elif kind == 'ipop_box':
            self.ipopval = self.ipop_box.value()

            self.ipop.blockSignals(True)
            self.ipop.setValue( int(linear_interp(self.ipopval,
                                            0, 1, 0, 100, lim=1)) )
            self.ipop.blockSignals(False)


        else:
            # defaults
            self.rate_box.setValue(1.0)
            self.ipop_box.setValue(0.5)
            self.setValues('rate_box')
            self.setValues('ipop_box')


class Widget(QWidget):
    def __init__(self, app, parent=None):
        super(Widget, self).__init__(parent=parent)

        self.setStyleSheet(f"Widget {{ background-color: {colors['dark']}; }}")

        self.app = app

        self.horizontalLayout = QHBoxLayout(self)
        self.controls = Controls(parent=self)

        self.controls.setValues()
        self.horizontalLayout.addWidget(self.controls)

        self.win = pg.GraphicsLayoutWidget()

        self.setWindowTitle("Logistic Map 🤯")
        self.horizontalLayout.addWidget(self.win)

        self.plots = [
                        self.win.addPlot(col=1, title="Cobwebb",
                                        labels={'left':"Population Next",
                                                'bottom':"Population"}),

                        self.win.addPlot(col=2, title="Population vs Time",
                                        labels={'left':"Population",
                                                'bottom':"Time"}),

                        self.win.addPlot(col=3, title="Bifurcations",
                                        labels={'left':"Equilibrium Population",
                                                'bottom':"Rates"}),
        ]

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate_plot)
        self.animate = False
        self.f = 0

        self.update_plot()

        self.controls.rate.valueChanged.connect(self.update_plot)
        self.controls.rate_box.valueChanged.connect(self.update_plot)
        self.controls.ipop.valueChanged.connect(self.update_plot)
        self.controls.cobweb_box.stateChanged.connect(self.update_plot)
        self.controls.series_box.stateChanged.connect(self.update_plot)
        self.controls.bifurc_box.stateChanged.connect(self.update_plot)
        self.controls.reset.pressed.connect(self.reset)
        self.controls.clear.pressed.connect(self.clear)
        self.controls.animb.pressed.connect(self.animate_toggle)

    def clear(self):
        print('Clear')
        for p in self.plots:
            p.clear()
        self.update_plot()

    def reset(self):
        print('Reset')
        self.animate = False
        self.timer.stop()
        self.f = 0
        self.update_plot()

    def animate_toggle(self):

        self.animate = not self.animate

        if self.animate:
            print('Animation play')
            self.update_plot()
            self.controls.animb.setText('Pause')
        else:
            print("Animation pause")
            self.timer.stop()
            self.controls.animb.setText('Play')

    def keyPressEvent(self, event):

        if event.key() == 32: # Space
            self.animate_toggle()

        elif event.key() == 16777219: # Backspace
            self.reset()

        else:
            print(f'Unknown keypress: {event.key()}, "{event.text()}"')

    def update_plot(self):

        if self.animate and self.f == 0:
            # start animate
            self.animspeed = self.controls.animrate.value()
            self.animspeed = max(1, self.animspeed)
            self.animspeed = 1000/self.animspeed # sec/frame -> frames/msec

            self.f = 0
            self.fmax = 100
            self.timer.start(int(self.animspeed))

            self.redraw_plots()

        elif self.animate and self.f != 0:

            # continue animate
            self.animspeed = self.controls.animrate.value()
            self.animspeed = max(1, self.animspeed)
            self.animspeed = 1000/self.animspeed

            self.timer.start(int(self.animspeed))

        elif not self.animate and self.f == 0:

            self.redraw_plots()

    def redraw_plots(self, index=0):

        """ redraw plots ; controlled by checkmarks """

        rate = self.controls.rateval
        ipop = self.controls.ipopval

        y_vals = cobweb_plot(self.plots[0], idx=0, r=rate, cobweb_x=ipop)

        if self.controls.cobweb_box.isChecked():
            self.plots[0].setVisible(True)
        else:
            self.plots[0].setVisible(False)

        if self.controls.series_box.isChecked():
            self.plots[1].setVisible(True)
            series_plot(self.plots[1], y_vals, r=rate, xall=True)
        else:
            self.plots[1].setVisible(False)

        if self.controls.bifurc_box.isChecked():
            self.plots[2].setVisible(True)
            bifurc_plot(self.plots[2], y_vals, r=rate, ipop=ipop)
        else:
            self.plots[2].setVisible(False)

    def animate_plot(self):

        rate = self.controls.rateval
        ipop = self.controls.ipopval

        y_vals = cobweb_plot(self.plots[0], idx=1 + self.f, r=rate, cobweb_x=ipop)

        if self.controls.cobweb_box.isChecked():
            self.plots[0].setVisible(True)
        else:
            self.plots[0].setVisible(False)

        if self.controls.series_box.isChecked():
            self.plots[1].setVisible(True)
            series_plot(self.plots[1], y_vals, idx =1 + self.f, r=rate)
        else:
            self.plots[1].setVisible(False)

        if self.controls.bifurc_box.isChecked():
            self.plots[2].setVisible(True)
            bifurc_plot(self.plots[2], y_vals, r=rate, ipop=ipop)
        else:
            self.plots[2].setVisible(False)

        self.f += 1

        if self.f >= self.fmax:
            self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget(app)
    w.show()
    sys.exit(app.exec_())
