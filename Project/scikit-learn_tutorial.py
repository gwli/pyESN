# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 16:45:48 2017

@author: vili
"""

def test_linear_model():
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

test_linear_model()