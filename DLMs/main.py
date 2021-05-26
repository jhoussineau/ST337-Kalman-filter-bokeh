
# This file is a free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# Copyright (C) 2019 Jeremie Houssineau
#
# Support: jeremie.houssineau AT warwick.ac.uk
#

import numpy as np
from numpy.linalg import inv

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, Button, Select
from bokeh.plotting import figure, show

def noise_free(F, xdim, n, N):
    x = np.zeros((xdim, n*N))
    x[:, 0] = x0
    for k in range(1,n*N):
        x[:, k] = np.matmul(F, x[:, k-1])
    return x

def generate_data(x0, F, H, sqrtU, sqrtV, n, xdim, ydim, state_noise_seq, obs_noise_seq):
    x = np.zeros((xdim, n))
    y = np.zeros((ydim, n))
    x[:, 0] = x0
    y[:, 0] = np.matmul(H, x0) + np.matmul(sqrtV, obs_noise_seq[:, 0])
    for k in range(1,n):
        x[:, k] = np.matmul(F, x[:, k-1]) + np.matmul(sqrtU, state_noise_seq[:, k-1])
        y[:, k] = np.matmul(H, x[:, k]) + np.matmul(sqrtV, obs_noise_seq[:, k])
    return x, y


xdim = 2
ydim = 1

# Initial state
x0 = np.array([1.0, 1.0])

# Likelihood
sigma = 1.0
H = np.array([[1.0, 0.0]])
sqrtV = np.array([[sigma]])

# Evolution
lambd = 1.0
evol_noise = 0.2
# Transition matrix with one real eigenvalue
F = np.array([[lambd, 1.0], [0.0, lambd]])
sqrtU = evol_noise * np.eye(xdim) 

# Time steps
n = 100

state_noise_seq = np.random.randn(sqrtU.shape[1], n-1)
obs_noise_seq = np.random.randn(sqrtV.shape[1], n)

# Data generation
(x, y) = generate_data(x0, F, H, sqrtU, sqrtV, n, xdim, ydim, state_noise_seq, obs_noise_seq)


# Prepare display
N = 250
discr = 12 # Number of discretisation points
xllim = -25
xulim = 25
k_sel = 0
x_grid = np.linspace(xllim, xulim, N)

# Transition matrix for the interpolated noise-free model (with time step 1/discr)
lambd_pow_invN = np.power(lambd, 1./discr)
F_discr = np.array([[lambd_pow_invN, 1./(discr * lambd_pow_invN**(discr-1))], [0.0, lambd_pow_invN]])
x_cont = noise_free(F_discr, xdim, n, discr)

# Set up plot
plot = figure(plot_height=400, plot_width=800, title='First component and observations',
              tools='crosshair,pan,reset,save,wheel_zoom',
              x_axis_label='time step', y_axis_label='first component')

data = ColumnDataSource(data=dict(k=range(n), xx=x[0, :], xv=x[1, :], y=y[0, :]))
data_cont = ColumnDataSource(data=dict(k=np.linspace(0,n,n*discr), xx=x_cont[0, :], xv=x_cont[1, :]))

inter_x = plot.line('k', 'xx', source=data_cont, line_width=3, alpha=0.5, legend='noise-free state (lambda > 0)')
plot.line('k', 'xx', line_dash='dashed', color='firebrick', source=data, line_width=3, alpha=0.5, legend='state')
plot.circle('k', 'y', source=data, size=5, color='mediumseagreen', alpha=0.5, legend='observation')
 
plot.legend.location = "top_left"

plot_vel = figure(plot_height=400, plot_width=800, title='Second component',
              tools='crosshair,pan,reset,save,wheel_zoom',
              x_axis_label='time step', y_axis_label='second component')

inter_vx = plot_vel.line('k', 'xv', source=data_cont, line_width=3, alpha=0.5, legend='noise-free state (lambda > 0)')
plot_vel.line('k', 'xv', line_dash='dashed', color='firebrick', source=data, line_width=3, alpha=0.5, legend='state')

plot_vel.legend.location = "top_left"


# Set up widgets

menu = [("oneREV", "One real eigenvalue"), ("twoCEV", "Two complex eigenvalues")]
dropdown = Select(title="Model", options=menu, value='oneREV')

button = Button(label="Regenerate data")
sigma_slider = Slider(title="observation noise", value=sigma, start=0.05, end=5, step=0.05)
lambda_slider = Slider(title="lambda", value=lambd, start=-1.25, end=1.25, step=0.01)
omega_slider = Slider(title="--", value=0, start=0, end=2*np.pi, step=np.pi/12)
evol_noise_slider = Slider(title="sigma evolution noise", value=evol_noise, start=0.0, end=2.5, step=0.05)
x0_pos_slider = Slider(title="init first component", value=x0[0], start=-15.0, end=15.0, step=0.1)
x0_vel_slider = Slider(title="init second component", value=x0[1], start=-2.5, end=2.5, step=0.1)

# Function modifying the plot depending on the parameters
def update(attrname, old, new):
    
    x0[0] = x0_pos_slider.value
    x0[1] = x0_vel_slider.value

    evol_noise = evol_noise_slider.value
    lambd = lambda_slider.value
    # The interpolated noise-free model is only given for lambd >= 0 0 (otherwise square root is complex)
    if lambd < 0:
        lambd_pow_invN = 0.
        inter_x.visible = False
        inter_vx.visible = False
    else:
        lambd_pow_invN = np.power(lambd, 1./discr)
        inter_x.visible = True
        inter_vx.visible = True

    if dropdown.value == 'oneREV':
        F = np.array([[lambd, 1.0], [0.0, lambd]])
        if lambd <= 0:
            F_discr = np.zeros((2, 2))
        else:
            F_discr = np.array([[lambd_pow_invN, 1./(discr * lambd_pow_invN**(discr-1))], [0.0, lambd_pow_invN]])
    else:       
        omega = omega_slider.value
        co = np.cos(omega)
        so = np.sin(omega)
        F = lambd * np.array([[co, so],[-so, co]])
        if lambd <= 0:
            F_discr = np.zeros((2, 2))
        else:
            co = np.cos(omega / discr)
            so = np.sin(omega / discr)
            F_discr = lambd_pow_invN * np.array([[co, so],[-so, co]])

    sqrtU = evol_noise * np.eye(xdim)
    
    sigma = sigma_slider.value
    sqrtV = np.array([[sigma]])

    (x, y) = generate_data(x0, F, H, sqrtU, sqrtV, n, xdim, ydim, state_noise_seq, obs_noise_seq)
    x_cont = noise_free(F_discr, xdim, n, discr)
    data.data = dict(k=range(n), xx=x[0, :], xv=x[1, :], y=y[0, :])
    data_cont.data = dict(k=np.linspace(0,n,n*discr), xx=x_cont[0, :], xv=x_cont[1, :])

def update_model(attrname, old, new):
    if dropdown.value == 'oneREV':
        omega_slider.title = '--'
    else:
        omega_slider.title = 'omega'
    update('value', 1, 1)

dropdown.on_change('value', update_model)

# Function regenerating the observations
def reg():
    global state_noise_seq, obs_noise_seq
    state_noise_seq = np.random.randn(sqrtU.shape[1], n-1)
    obs_noise_seq = np.random.randn(sqrtV.shape[1], n) 
    update('value', 1, 1)

button.on_click(reg)

sigma_slider.on_change('value', update)
lambda_slider.on_change('value', update)
omega_slider.on_change('value', update)
evol_noise_slider.on_change('value', update)
x0_pos_slider.on_change('value', update)
x0_vel_slider.on_change('value', update)

# Set up layouts and add to document
inputs = widgetbox(dropdown, button, sigma_slider, lambda_slider, omega_slider, evol_noise_slider, x0_pos_slider, x0_vel_slider)

curdoc().add_root(gridplot([plot, inputs, plot_vel], ncols=2))
curdoc().title = "HMM: one real eigenvalue"

