
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
from bokeh.models.widgets import Slider, Button
from bokeh.plotting import figure, show

def normal_pdf(x, mu, var):
    return np.exp(-np.power(x-mu, 2)/(2*var)) / np.sqrt(2 * np.pi * var)

def prediction(x, F, sqrtU):
    return np.matmul(F, x) + np.matmul(sqrtU, np.random.randn(sqrtU.shape[1]))

def observation(x, H, sqrtV):
    return np.matmul(H, x) + np.matmul(sqrtV, np.random.randn(sqrtV.shape[1]))

def Kalman_filter(y, m0, P0, F, H, U, V, n, xdim, delta):
    mu = np.zeros((xdim, n))
    Pu = np.zeros((xdim, xdim, n))
    mp = np.zeros((xdim, n))
    Pp = np.zeros((xdim, xdim, n))
    mp[:, 0] = m0
    Pp[:, :, 0] = P0
    for k in range(n):
        if k < n - delta:
            # Update
            z = y[:, k] - np.matmul(H, mp[:, k])
            S = np.matmul(np.matmul(H, Pp[:, :, k]), H.transpose()) + V
            K = np.matmul(np.matmul(Pp[:, :, k], H.transpose()), inv(S))
            mu[:, k] = mp[:, k] + np.matmul(K, z)
            Pu[:, :, k] = np.matmul(np.eye(xdim) - np.matmul(K, H), Pp[:, :, k])
        else:
            mu[:, k] = mp[:, k]
            Pu[:, :, k] = Pp[:, :, k]
        
        if k < n-1:
            # Prediction
            mp[:, k+1] = np.matmul(F, mu[:, k])
            Pp[:, :, k+1] = np.matmul(np.matmul(F, Pu[:, :, k]), F.transpose()) + U
    return mp, Pp, mu, Pu

def generate_data(F, H, sqrtU, sqrtV, n, xdim, ydim):
    x = np.zeros((xdim, n))
    y = np.zeros((ydim, n))
    x[:, 0] = x0
    y[:, 0] = observation(x0, H, sqrtV)
    for k in range(1,n):
        x[:, k] = prediction(x[:, k-1], F, sqrtU)
        y[:, k] = observation(x[:, k], H, sqrtV)
    return x, y


xdim = 2
ydim = 1

# Initial state
x0 = np.array([0.0, 0.5])

# Likelihood
sigma = 1.0
H = np.array([[1.0, 0.0]])
sqrtV = np.array([[sigma]])
V = np.matmul(sqrtV, sqrtV.transpose())

# Prior
m0 = np.array([0.0, 0.0])
P0 = np.array([[1.0, 0.0], [0.0, 0.25]])

# Evolution
dt = 1.0
true_acc_noise = 0.1
true_sqrtU = true_acc_noise * np.array([[dt**2/2],[dt]])

acc_noise = 0.05
F = np.array([[1.0, dt], [0.0, 1.0]])
sqrtU = acc_noise * np.array([[dt**2/2],[dt]])
U = np.matmul(sqrtU, sqrtU.transpose())

# Time steps
n = 10
delta = 0

# Data generation
(x, y) = generate_data(F, H, true_sqrtU, sqrtV, n, xdim, ydim)

# Kalman filter
(mp, Pp, mu, Pu) = Kalman_filter(y, m0, P0, F, H, U, V, n, xdim, delta)

# Number of points to plot the p.d.f.s (reduce this number of speed)
N = 250

# Prepare display
xllim = -7.5
xulim = 12.5
k_sel = 0
x_grid = np.linspace(xllim, xulim, N)

# Set up plot
plot = figure(plot_height=400, plot_width=800, title='Kalman filter (position)',
              tools='crosshair,pan,reset,save,wheel_zoom',
              x_range=[-.5, n], y_range=[xllim, xulim],
              x_axis_label='time step', y_axis_label='position')

data = ColumnDataSource(data=dict(k=range(n),xx=x[0, :], xv=x[1, :], y=y[0, :]))
est_data = ColumnDataSource(data=dict(k=range(n), mx=mu[0, :], mv=mu[1, :]))

plot.line('k', 'xx', source=data, line_width=3, line_alpha=0.5, legend_label='true')
plot.circle('k', 'y', source=data, size=5, color='mediumseagreen', alpha=0.5, legend_label='observation')
plot.circle('k', 'mx', source=est_data, size=5, color='firebrick', alpha=0.5, legend_label='estimated')

pdf_data = []
pdf_plot_prior = []
pdf_plot_post = []
for k in range(n):
    prior_pdf = normal_pdf(x_grid, mp[0, k], Pp[0, 0, k])
    post_pdf = normal_pdf(x_grid, mu[0, k], Pu[0, 0, k])
    pdf_data.append(ColumnDataSource(data=dict(x_prior=k+prior_pdf, x_post=k+post_pdf, y=x_grid)))
    if k == k_sel:
        vis = True
    else:
        vis = False

    pdf_plot_prior.append(plot.line('x_prior', 'y', legend_label='predicted', source=pdf_data[k], line_width=3,
            line_alpha=0.6))
    pdf_plot_post.append(plot.line('x_post', 'y', legend_label='posterior', source=pdf_data[k], line_width=3,
            line_alpha=0.6, line_dash='dashed'))
    pdf_plot_prior[k].visible = vis
    pdf_plot_post[k].visible = vis

plot.legend.location = "top_left"

# show(plot)

plot_vel = figure(plot_height=400, plot_width=800, title='Kalman filter (velocity)',
              tools='crosshair,pan,reset,save,wheel_zoom',
              x_range=[-.5, n], y_range=[-2.0, 2.0],
              x_axis_label='time step', y_axis_label='velocity')


plot_vel.line('k', 'xv', source=data, line_width=3, line_alpha=0.5, legend_label='true')
plot_vel.circle('k', 'mv', source=est_data, size=5, color='firebrick', alpha=0.5, legend_label='estimated')

plot_vel.legend.location = "top_left"

pdf_data_vel = []
pdf_plot_prior_vel = []
pdf_plot_post_vel = []
for k in range(n):
    prior_pdf_vel = normal_pdf(x_grid, mp[1, k], Pp[1, 1, k])
    post_pdf_vel = normal_pdf(x_grid, mu[1, k], Pu[1, 1, k])
    pdf_data_vel.append(ColumnDataSource(data=dict(x_prior=k+prior_pdf_vel, x_post=k+post_pdf_vel, y=x_grid)))
    if k == k_sel:
        vis = True
    else:
        vis = False

    pdf_plot_prior_vel.append(plot_vel.line('x_prior', 'y', legend_label='predicted', source=pdf_data_vel[k],
            line_width=3, line_alpha=0.6))
    pdf_plot_post_vel.append(plot_vel.line('x_post', 'y', legend_label='posterior', source=pdf_data_vel[k],
            line_width=3, line_alpha=0.6, line_dash='dashed'))
    pdf_plot_prior_vel[k].visible = vis
    pdf_plot_post_vel[k].visible = vis


# Set up widgets
button = Button(label="Regenerate data")
k_slider = Slider(title="k", value=0, start=0, end=n-1, step=1)
delta_slider = Slider(title="delta (forecasting from k-delta)", value=0, start=0, end=n, step=1)
acc_noise_slider = Slider(title="sigma acceleration noise", value=acc_noise, start=0.01, end=0.1, step=0.01)
m0_pos_slider = Slider(title="prior position", value=m0[0], start=-5.0, end=5.0, step=0.1)
sig0_pos_slider = Slider(title="prior sigma position", value=np.sqrt(P0[0, 0]), start=0.05, end=5.0, step=0.05)
sig0_vel_slider = Slider(title="prior sigma velocity", value=np.sqrt(P0[1, 1]), start=0.05, end=2.0, step=0.05)

def update_time(attrname, old, new):
    pdf_plot_prior[old].visible = False
    pdf_plot_post[old].visible = False
    pdf_plot_prior[new].visible = True
    pdf_plot_post[new].visible = True
    pdf_plot_prior_vel[old].visible = False
    pdf_plot_post_vel[old].visible = False
    pdf_plot_prior_vel[new].visible = True
    pdf_plot_post_vel[new].visible = True

k_slider.on_change('value', update_time)

def update(attrname, old, new):
    acc_noise = acc_noise_slider.value
    sqrtU = acc_noise * np.array([[dt**2/2],[dt]])
    U = np.matmul(sqrtU, sqrtU.transpose())

    delta = delta_slider.value
    m0[0] = m0_pos_slider.value
    P0[0, 0] = sig0_pos_slider.value**2
    P0[1, 1] = sig0_vel_slider.value**2
    (mp, Pp, mu, Pu) = Kalman_filter(y, m0, P0, F, H, U, V, n, xdim, delta)
    est_data.data = dict(k=range(n), mx=mu[0, :], mv=mu[1, :])

    k_sel = k_slider.value
    for k in range(n):
        prior_pdf = normal_pdf(x_grid, mp[0, k], Pp[0, 0, k])
        post_pdf = normal_pdf(x_grid, mu[0, k], Pu[0, 0, k])
        pdf_data[k].data = dict(x_prior=k+prior_pdf, x_post=k+post_pdf, y=x_grid)
        if k == k_sel:
            vis = True
        else:
            vis = False
        pdf_plot_prior[k].visible = vis
        pdf_plot_post[k].visible = vis
        
        prior_pdf_vel = normal_pdf(x_grid, mp[1, k], Pp[1, 1, k])
        post_pdf_vel = normal_pdf(x_grid, mu[1, k], Pu[1, 1, k])
        pdf_data_vel[k].data = dict(x_prior=k+prior_pdf_vel, x_post=k+post_pdf_vel, y=x_grid)
        if k == k_sel:
            vis = True
        else:
            vis = False
        pdf_plot_prior_vel[k].visible = vis
        pdf_plot_post_vel[k].visible = vis

def reg():
    global x, y
    (x, y) = generate_data(F, H, true_sqrtU, sqrtV, n, xdim, ydim)
    data.data = dict(k=range(n),xx=x[0, :], xv=x[1, :], y=y[0, :])
    update('value', 1, 1)

button.on_click(reg)

delta_slider.on_change('value', update)
acc_noise_slider.on_change('value', update)
m0_pos_slider.on_change('value', update)
sig0_pos_slider.on_change('value', update)
sig0_vel_slider.on_change('value', update)

# Set up layouts and add to document
inputs = widgetbox(button, k_slider, delta_slider, acc_noise_slider,
        m0_pos_slider, sig0_pos_slider, sig0_vel_slider)

curdoc().add_root(gridplot([plot, inputs, plot_vel], ncols=2))
curdoc().title = "Kalman filtering for a nearly constant velocity model"

