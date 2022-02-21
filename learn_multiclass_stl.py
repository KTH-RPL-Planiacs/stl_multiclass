import sys, getopt, os
import operator as operatorclass
import pickle
import dill
import itertools
from collections import Counter, deque
import math
from psomax import pso
from evaluation_metrics import hamming_loss
from evaluation_metrics import example_based_accuracy
import numpy as np
import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
from sklearn.metrics import confusion_matrix


#plotting a confusion heatmap matrix
def plot_cm(y_true, y_pred, class_names):
    y_true = [str(list(i)) for i in y_true]
    y_pred = [str(list(i)) for i in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9)) 
    ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap=sns.diverging_palette(220, 20, n=7),
            ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # ta-da!
    # plt.savefig('cm_userstudy.pdf', bbox_inches='tight') # ta-da!



def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
    


class STLFormula:
    """
    Class for representing an STL Formula.
    """
    
    
    class TrueF:
        """
        Class representing the True boolean constant
        """
        def __init__(self):
            self.robustness = lambda s, t : float('inf')
            self.sat = True
            self.horizon = 0
            
        def __str__(self):
            return "\\top"


    class FalseF:
        """
        Class representing the False boolean constant
        """
        def __init__(self):
            self.robustness = lambda s, t : float('-inf')
            self.sat = False
            self.horizon = 0
            
        def __str__(self):
            return "\\bot"


    class Predicate:
        """
        Class representing a Predicate, s.t. f(s) \sim \mu
        The constructor takes 4 arguments:
            * dimension: string/name of the dimension
            * operator: operator (geq, lt...)
            * mu: \mu
            * pi_index_signal: in the signal, which index corresponds to the predicate's dimension
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
            * sat: a function returning whether \rho(s,(f(s) \sim \mu),t) > 0
            * horizon: 0
        """
        def __init__(self,dimension,operator,mu,pi_index_signal):
            self.pi_index_signal = pi_index_signal
            self.dimension = dimension
            self.operator = operator
            self.mu = mu
            if operator == operatorclass.gt or operator == operatorclass.ge:
                self.robustness = lambda s, t : s[t][pi_index_signal] - mu
                self.sat = lambda s, t : s[t][pi_index_signal] - mu > 0
            else:
                self.robustness = lambda s, t : -s[t][pi_index_signal] + mu
                self.sat = lambda s, t : -s[t][pi_index_signal] + mu > 0
            
            self.horizon = 0
        
        def __str__(self):
            return self.dimension+operators_iv[self.operator]+str(self.mu)


    class STPredicate2D:
        """
        Class representing a Spatio-Temporal 2D Predicate of the form (\alpha < x < \beta  \wedge \gamma < y < \delta)
        The constructor takes 6 arguments:
            * index_signal_dimension_x: dimension index for x-dimension (typically 0)
            * index_signal_dimension_y: dimension index for y-dimension (typically 1)
            * alpha: \alpha
            * beta: \beta
            * gamma: \gamma
            * delta: \delta
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,(f(s) \sim \mu),t) & = \begin{cases} \mu-f(s_t) & \sim=\le \\ f(s_t)-\mu & \sim=\ge \end{cases}
            * sat: a function returning whether \rho > 0
            * horizon: 0
        """
        
        
        
        def __init__(self,index_signal_dimension_x,index_signal_dimension_y,alpha,beta,gamma,delta):
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
            self.delta = delta
            
            # class NotWellFormed_STPredicate2D(Exception):
                # """When \alpha > \beta or \gamma > \delta"""
                # print("("+str(self.alpha)+" < x < "+str(self.beta)+" \wedge "+str(self.gamma)+" < y < "+str(self.delta)+") is not a well formed STPredicate2D")
                # pass
            
            # if alpha > beta or gamma > delta:
                # raise NotWellFormed_STPredicate2D
            
            #encoding \alpha < x
            alpha_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_x] - alpha
            alpha_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_x] - alpha > 0
            #encoding x < \beta
            beta_gt_x_robustness  = lambda s, t : -s[t][index_signal_dimension_x] + beta
            beta_gt_x_sat         = lambda s, t : -s[t][index_signal_dimension_x] + beta > 0
            #encoding \gamma < y
            gamma_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_y] - gamma
            gamma_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_y] - gamma > 0
            #encoding y < \delta
            delta_gt_x_robustness = lambda s, t : -s[t][index_signal_dimension_y] + delta
            delta_gt_x_sat        = lambda s, t : -s[t][index_signal_dimension_y] + delta > 0
            
            self.horizon = 0
            
            self.robustness = lambda s, t : min([alpha_lt_x_robustness(s,t),beta_gt_x_robustness(s,t),gamma_lt_x_robustness(s,t),delta_gt_x_robustness(s,t)])
            self.sat        = lambda s, t : all([alpha_lt_x_sat(s,t),beta_gt_x_sat(s,t),gamma_lt_x_sat(s,t),delta_gt_x_sat(s,t)])
        
        
        def __str__(self):
            return "("+str(round(self.alpha,3))+" < x < "+str(round(self.beta,3))+" \wedge "+str(round(self.gamma,3))+" < y < "+str(round(self.delta,3))+")"



    class Always_STPredicate2D:
        """
        Class representing a Spatio-Temporal 2D Predicate of the form \Box_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta)
        The constructor takes 8 arguments:
            * index_signal_dimension_x: dimension index for x-dimension (typically 0)
            * index_signal_dimension_y: dimension index for y-dimension (typically 1)
            * alpha: \alpha
            * beta: \beta
            * gamma: \gamma
            * delta: \delta
            * t1: t1
            * t2: t2
        The class contains additional attributes:
            * robustness
            * sat: a function returning whether \rho > 0
            * horizon
        """    
        
        def __init__(self,index_signal_dimension_x,index_signal_dimension_y,alpha,beta,gamma,delta,t1,t2):
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
            self.delta = delta
            self.t1 = t1
            self.t2 = t2
            self.index_signal_dimension_x = index_signal_dimension_x
            self.index_signal_dimension_y = index_signal_dimension_y
            
            #encoding \alpha < x
            alpha_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_x] - alpha
            alpha_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_x] - alpha > 0
            #encoding x < \beta
            beta_gt_x_robustness  = lambda s, t : -s[t][index_signal_dimension_x] + beta
            beta_gt_x_sat         = lambda s, t : -s[t][index_signal_dimension_x] + beta > 0
            #encoding \gamma < y
            gamma_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_y] - gamma
            gamma_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_y] - gamma > 0
            #encoding y < \delta
            delta_gt_x_robustness = lambda s, t : -s[t][index_signal_dimension_y] + delta
            delta_gt_x_sat        = lambda s, t : -s[t][index_signal_dimension_y] + delta > 0
            
            #enconding the conjunction of 4 subpredicates
            self.conj_robustness = lambda s, t : min([alpha_lt_x_robustness(s,t),beta_gt_x_robustness(s,t),gamma_lt_x_robustness(s,t),delta_gt_x_robustness(s,t)])
            self.conj_sat        = lambda s, t : all([alpha_lt_x_sat(s,t),beta_gt_x_sat(s,t),gamma_lt_x_sat(s,t),delta_gt_x_sat(s,t)])
        
            #encoding the whole always stl2d predicate
            self.robustness = lambda s, t : min([ self.conj_robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.sat        = lambda s, t : all([ self.conj_sat(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2
        
        def __str__(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(round(self.alpha,3))+" < x < "+str(round(self.beta,3))+" \wedge "+str(round(self.gamma,3))+" < y < "+str(round(self.delta,3))+")"


        def __and__(self, other):
            """
                Function returning the intersection of 2 STL formulae of the form of \Box_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return None
            x1 = max(min(self.alpha, self.beta), min(other.alpha, other.beta))
            y1 = max(min(self.gamma, self.delta), min(other.gamma, other.delta))
            t1 = max(min(self.t1, self.t2), min(other.t1, other.t2))
            x2 = min(max(self.alpha, self.beta), max(other.alpha, other.beta))
            y2 = min(max(self.gamma, self.delta), max(other.gamma, other.delta))
            t2 = min(max(self.t1, self.t2), max(other.t1, other.t2))
            if x1 < x2 and y1 < y2 and t1 < t2:
                return type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, y1, x2, y2, t1, t2)
            return None

        def __sub__(self,other):
            """
                Function returning the difference between 2 STL formulae of the form of \Box_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return self
            if not self & other:
                yield self
                return
            xs = {self.alpha, self.beta}
            ys = {self.gamma, self.delta}
            ts = {self.t1, self.t2}
            if self.alpha < other.alpha < self.beta: xs.add(other.alpha)
            if self.alpha < other.beta < self.beta: xs.add(other.beta)
            if self.gamma < other.gamma < self.delta: ys.add(other.gamma)
            if self.gamma < other.delta < self.delta: ys.add(other.delta)
            if self.t1 < other.t1 < self.t2: ts.add(other.t1)
            if self.t1 < other.t2 < self.t2: ts.add(other.t2)
            for (x1, x2), (y1, y2), (t1, t2) in itertools.product(pairwise(sorted(xs)), pairwise(sorted(ys)), pairwise(sorted(ts))):
                rect = type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, x2, y1, y2, t1, t2)
                if rect != self & other:
                    yield rect
        
        def __iter__(self):
            yield self.alpha
            yield self.beta
            yield self.gamma
            yield self.delta
            yield self.t1
            yield self.t2
        
        def __eq__(self, other):
            return type(self) is type(other) and tuple(self) == tuple(other)
        
        def __ne__(self, other):
            return not (self == other)


    class Eventually_STPredicate2D:      
        """
        Class representing a Spatio-Temporal 2D Predicate of the form \diamondsuit_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta)
        The constructor takes 8 arguments:
            * index_signal_dimension_x: dimension index for x-dimension (typically 0)
            * index_signal_dimension_y: dimension index for y-dimension (typically 1)
            * alpha: \alpha
            * beta: \beta
            * gamma: \gamma
            * delta: \delta
            * t1: t1
            * t2: t2
        The class contains additional attributes:
            * robustness
            * sat: a function returning whether \rho > 0
            * horizon
        """      
        
        def __init__(self,index_signal_dimension_x,index_signal_dimension_y,alpha,beta,gamma,delta,t1,t2):
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
            self.delta = delta
            self.t1 = t1
            self.t2 = t2
            self.index_signal_dimension_x = index_signal_dimension_x
            self.index_signal_dimension_y = index_signal_dimension_y
            
            #encoding \alpha < x
            alpha_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_x] - alpha
            alpha_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_x] - alpha > 0
            #encoding x < \beta
            beta_gt_x_robustness  = lambda s, t : -s[t][index_signal_dimension_x] + beta
            beta_gt_x_sat         = lambda s, t : -s[t][index_signal_dimension_x] + beta > 0
            #encoding \gamma < y
            gamma_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_y] - gamma
            gamma_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_y] - gamma > 0
            #encoding y < \delta
            delta_gt_x_robustness = lambda s, t : -s[t][index_signal_dimension_y] + delta
            delta_gt_x_sat        = lambda s, t : -s[t][index_signal_dimension_y] + delta > 0
            
            #enconding the conjunction of 4 subpredicates
            self.conj_robustness = lambda s, t : min([alpha_lt_x_robustness(s,t),beta_gt_x_robustness(s,t),gamma_lt_x_robustness(s,t),delta_gt_x_robustness(s,t)])
            self.conj_sat        = lambda s, t : all([alpha_lt_x_sat(s,t),beta_gt_x_sat(s,t),gamma_lt_x_sat(s,t),delta_gt_x_sat(s,t)])
        
            #encoding the whole eventually stl2d predicate
            self.robustness = lambda s, t : max([ self.conj_robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.sat        = lambda s, t : any([ self.conj_sat(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2
        
        def __str__(self):
            return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(round(self.alpha,3))+" < x < "+str(round(self.beta,3))+" \wedge "+str(round(self.gamma,3))+" < y < "+str(round(self.delta,3))+")"
        
        def __and__(self, other):
            """
                Function returning the intersection of 2 STL formulae of the form of \diamondsuit_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return None
            x1 = max(min(self.alpha, self.beta), min(other.alpha, other.beta))
            y1 = max(min(self.gamma, self.delta), min(other.gamma, other.delta))
            t1 = max(min(self.t1, self.t2), min(other.t1, other.t2))
            x2 = min(max(self.alpha, self.beta), max(other.alpha, other.beta))
            y2 = min(max(self.gamma, self.delta), max(other.gamma, other.delta))
            t2 = min(max(self.t1, self.t2), max(other.t1, other.t2))
            if x1 < x2 and y1 < y2 and t1 < t2:
                return type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, y1, x2, y2, t1, t2)
            return None

        def __sub__(self,other):
            """
                Function returning the difference between 2 STL formulae of the form of \diamondsuit_{[t1,t2]}(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return self
            if not self & other:
                yield self
                return
            xs = {self.alpha, self.beta}
            ys = {self.gamma, self.delta}
            ts = {self.t1, self.t2}
            if self.alpha < other.alpha < self.beta: xs.add(other.alpha)
            if self.alpha < other.beta < self.beta: xs.add(other.beta)
            if self.gamma < other.gamma < self.delta: ys.add(other.gamma)
            if self.gamma < other.delta < self.delta: ys.add(other.delta)
            if self.t1 < other.t1 < self.t2: ts.add(other.t1)
            if self.t1 < other.t2 < self.t2: ts.add(other.t2)
            for (x1, x2), (y1, y2), (t1, t2) in itertools.product(pairwise(sorted(xs)), pairwise(sorted(ys)), pairwise(sorted(ts))):
                rect = type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, x2, y1, y2, t1, t2)
                if rect != self & other:
                    yield rect
        
        def __iter__(self):
            yield self.alpha
            yield self.beta
            yield self.gamma
            yield self.delta
            yield self.t1
            yield self.t2
        
        def __eq__(self, other):
            return type(self) is type(other) and tuple(self) == tuple(other)
        
        def __ne__(self, other):
            return not (self == other)


    class AlwaysNot_STPredicate2D:      
        """
        Class representing a Spatio-Temporal 2D Predicate of the form \Box_{[t1,t2]}\neg(\alpha < x < \beta  \wedge \gamma < y < \delta)
        The constructor takes 8 arguments:
            * index_signal_dimension_x: dimension index for x-dimension (typically 0)
            * index_signal_dimension_y: dimension index for y-dimension (typically 1)
            * alpha: \alpha
            * beta: \beta
            * gamma: \gamma
            * delta: \delta
            * t1: t1
            * t2: t2
        The class contains additional attributes:
            * robustness
            * sat: a function returning whether \rho > 0
            * horizon
        """      
        
        def __init__(self,index_signal_dimension_x,index_signal_dimension_y,alpha,beta,gamma,delta,t1,t2):
            self.index_signal_dimension_x = index_signal_dimension_x
            self.index_signal_dimension_y = index_signal_dimension_y
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
            self.delta = delta
            self.t1 = t1
            self.t2 = t2
            
            #encoding \alpha < x
            alpha_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_x] - alpha
            alpha_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_x] - alpha > 0
            #encoding x < \beta
            beta_gt_x_robustness  = lambda s, t : -s[t][index_signal_dimension_x] + beta
            beta_gt_x_sat         = lambda s, t : -s[t][index_signal_dimension_x] + beta > 0
            #encoding \gamma < y
            gamma_lt_x_robustness = lambda s, t : s[t][index_signal_dimension_y] - gamma
            gamma_lt_x_sat        = lambda s, t : s[t][index_signal_dimension_y] - gamma > 0
            #encoding y < \delta
            delta_gt_x_robustness = lambda s, t : -s[t][index_signal_dimension_y] + delta
            delta_gt_x_sat        = lambda s, t : -s[t][index_signal_dimension_y] + delta > 0
            
            #enconding the conjunction of 4 subpredicates
            self.conj_robustness = lambda s, t : -min([alpha_lt_x_robustness(s,t),beta_gt_x_robustness(s,t),gamma_lt_x_robustness(s,t),delta_gt_x_robustness(s,t)])
            self.conj_sat        = lambda s, t : not all([alpha_lt_x_sat(s,t),beta_gt_x_sat(s,t),gamma_lt_x_sat(s,t),delta_gt_x_sat(s,t)])
        
            #encoding the whole eventually stl2d predicate
            self.robustness = lambda s, t : max([ self.conj_robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.sat        = lambda s, t : any([ self.conj_sat(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2
        
        def __str__(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]} \lnot ("+str(round(self.alpha,3))+" < x < "+str(round(self.beta,3))+" \wedge "+str(round(self.gamma,3))+" < y < "+str(round(self.delta,3))+")"


        def __and__(self, other):
            """
                Function returning the intersection of 2 STL formulae of the form of \Box_{[t1,t2]}\neg(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return None
            x1 = max(min(self.alpha, self.beta), min(other.alpha, other.beta))
            y1 = max(min(self.gamma, self.delta), min(other.gamma, other.delta))
            t1 = max(min(self.t1, self.t2), min(other.t1, other.t2))
            x2 = min(max(self.alpha, self.beta), max(other.alpha, other.beta))
            y2 = min(max(self.gamma, self.delta), max(other.gamma, other.delta))
            t2 = min(max(self.t1, self.t2), max(other.t1, other.t2))
            if x1 < x2 and y1 < y2 and t1 < t2:
                return type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, y1, x2, y2, t1, t2)
            return None

        def __sub__(self,other):
            """
                Function returning the difference between 2 STL formulae of the form of \Box_{[t1,t2]}\neg(\alpha < x < \beta  \wedge \gamma < y < \delta).
                If the 2 formulae are not of the same form, or if the STL formulae do not intersect, it returns None.
            """
            if type(self) is not type(other):
                return self
            if not self & other:
                yield self
                return
            xs = {self.alpha, self.beta}
            ys = {self.gamma, self.delta}
            ts = {self.t1, self.t2}
            if self.alpha < other.alpha < self.beta: xs.add(other.alpha)
            if self.alpha < other.beta < self.beta: xs.add(other.beta)
            if self.gamma < other.gamma < self.delta: ys.add(other.gamma)
            if self.gamma < other.delta < self.delta: ys.add(other.delta)
            if self.t1 < other.t1 < self.t2: ts.add(other.t1)
            if self.t1 < other.t2 < self.t2: ts.add(other.t2)
            for (x1, x2), (y1, y2), (t1, t2) in itertools.product(pairwise(sorted(xs)), pairwise(sorted(ys)), pairwise(sorted(ts))):
                rect = type(self)(self.index_signal_dimension_x,self.index_signal_dimension_y, x1, x2, y1, y2, t1, t2)
                if rect != self & other:
                    yield rect
        
        
        def __iter__(self):
            yield self.alpha
            yield self.beta
            yield self.gamma
            yield self.delta
            yield self.t1
            yield self.t2
        
        def __eq__(self, other):
            return type(self) is type(other) and tuple(self) == tuple(other)
        
        def __ne__(self, other):
            return not (self == other)


    class Conjunction: 
        """
        Class representing the Conjunction operator, s.t. \phi_1 \wedge \phi_2 \wedge \ldots \wedge \phi_n.
        The constructor takes 1 arguments:
            * lst_conj: a list of STL formulae in the conjunction
        The class contains 1 additional attributes:
            * sat: a function \sigma(t_i) \models \phi_1 \land \phi_2 \land  \ldots \land \phi_n \Leftrightarrow (\sigma(t_i) \models \phi_1 ) \land (\sigma(t_i) \models \phi_2) \land \ldots \land (\sigma(t_i) \models \phi_n )
        """
        def __init__(self,lst_conj):
            self.lst_conj   = lst_conj
            self.sat        = lambda s, t : all([formula.sat(s,t) for formula in self.lst_conj])
            self.robustness = lambda s, t : min([formula.robustness(s,t) for formula in self.lst_conj])
            self.horizon    = max([formula.horizon(s,t) for formula in self.lst_conj])
        
        def __str__(self):
            s = "("
            for conj in self.lst_conj:
                s += str(conj) + " \wedge "
            return s[:-8]+")"


    class Negation: 
        """
        Class representing the Negation operator, s.t. \neg \phi.
        The constructor takes 1 argument:
            * formula 1: \phi
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\neg \phi,t) = - \rho(s,\phi,t)
            * horizon: \left\|\phi\right\|=\left\|\neg \phi\right\|
        """
        def __init__(self,formula):
            self.formula = formula
            self.robustness = lambda s, t : -formula.robustness(s,t)
            self.sat = lambda s, t : not formula.sat(s,t)
            self.horizon = formula.horizon
        
        def __str__(self):
            return "\lnot ("+str(self.formula)+")"


    class Disjunction: 
        """
        Class representing the Disjunction operator, s.t. \phi_1 \vee \phi_2.
        The constructor takes 2 arguments:
            * formula 1: \phi_1
            * formula 2: \phi_2
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\phi_1 \lor \phi_2,t) = \max(\rho(s,\phi_1,t),\rho(s,\phi_2,t) )
            * horizon: \left\|\phi_1 \lor \phi_2\right\|= \max\{\left\|\phi_1\right\|, \left\|\phi_2\right\|\}
        """
        def __init__(self,lst_disj,list_probas):
            self.lst_disj    = lst_disj
            self.list_probas = list_probas
            self.sat         = lambda s, t : any([formula.sat(s,t) for formula in self.lst_disj])
            self.robustness  = lambda s, t : max([formula.robustness(s,t) for formula in self.lst_disj])
            self.horizon    = max([formula.horizon(s,t) for formula in self.lst_conj])
        
        def __str__(self):
            s = "("
            for disj,prob in zip(self.lst_disj,self.list_probas):
                s += "(" + str(disj) + ")_{" + str(prob) + "} \\vee "
            return s[:-6]+")"


    class Always: 
        """
        Class representing the Always operator, s.t. \mathcal{G}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{G}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\min~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{G}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t : min([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.sat        = lambda s, t : all([ formula.sat(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{G}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"


    class Eventually: 
        """
        Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * formula: a formula \phi
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 2 additional attributes:
            * robustness: a function \rho(s,\mathcal{F}_{[t1,t2]}~ \phi,t) = underset{t' \in t+[t1,t2]}\max~  \rho(s,\phi,t').
            * horizon: \left\|\mathcal{F}_{[t1, t2]} \phi\right\|=t2+ \left\|\phi\right\|
        """
        def __init__(self,formula,t1,t2):
            self.formula = formula
            self.t1 = t1
            self.t2 = t2
            self.robustness = lambda s, t :  max([ formula.robustness(s,k) for k in range(t+t1, t+t2+1)])
            self.sat        = lambda s, t :  any([ formula.sat(s,k) for k in range(t+t1, t+t2+1)])
            self.horizon = t2 + formula.horizon
        
        def __str__(self):
            return "\mathcal{F}_{["+str(self.t1)+","+str(self.t2)+"]}("+str(self.formula)+")"






class DTLearn:
    """
        Class representing the Multi-label Multi-Class Decision Tree learning algorithm of an STL Formula.
        Constructor takes as input:
            * dict_trajectories: python dict identifying trajectoryID as key, and trajectories as value. A trajectory is a list of values [x_val,y_val], where each list of value represents the value of a trajectory at a given discrete time.
            * dict_trajectories_classes: python dict identifying trajectoryID as key, and list of classes as value.
            * list_classes: list of the different classes trajectories can belong to.
            * min_x: lowerbound of trajectories' values on the x-axis (or lowerbound search of specifications on the x-axis).
            * min_y: lowerbound of trajectories' values on the y-axis (or lowerbound search of specifications on the y-axis).
            * min_h: lowerbound of target specifications' horizon.
            * max_x: upperbound of trajectories' values on the x-axis (or upperbound search of specifications on the x-axis).
            * max_y: upperbound of trajectories' values on the y-axis (or upperbound search of specifications on the y-axis).
            * max_h: upperbound of target specifications' horizon.
            * max_depth (optional): termination criterion -- the maximum depth of the decision tree (default set to 5).
            * stl_diff (optional): if uses the STL difference based prunning of the decision tree nodes (default set to True). Enables to render more concise STL formulae.
            * verbose (optional): print details on the execution of the algorithm (default set to False).
    """    


    #Incremental ID for each new node
    ID = 1
    
    
    def __init__(self, 
                 dict_trajectories, 
				 dict_trajectories_classes, 
				 list_classes,
				 min_x, 
				 min_y, 
				 min_h, 
				 max_x, 
				 max_y, 
				 max_h, 
				 max_depth=5,
				 stl_diff=True,
				 verbose=False):
        self.max_depth = max_depth
        self.min_x     = min_x
        self.min_y     = min_y
        self.min_h     = min_h
        self.max_x     = max_x
        self.max_y     = max_y
        self.max_h     = max_h
        self.stl_diff  = stl_diff
        self.verbose   = verbose
        self.dict_trajectories         = dict_trajectories
        self.dict_trajectories_classes = dict_trajectories_classes
        self.list_classes              = list_classes
    
    
    class Node:
        """
            Class representing a non-terminal node of the decision tree.
            Constructor takes as input:
                * stl: the STL Formula used to locally separate the data in this node
                * left: left child node
                * right: right child node
                * elements: the elements to classify in the node
                * depth: the depth of the node (used for termination criteria)
            Attributes:
                * identifier: a unique identifier for the node
        """
        def __init__(self,stl,left,right,depth):
            self.stl      = stl
            self.elements = []
            self.left     = left
            self.right    = right
            self.depth    = depth
            self.identifier = DTLearn.ID
            DTLearn.ID += 1
        
        def __str__(self):
            return self.identifier
    
    
    class Leaf:
        """
            Class representing a terminal node of the decision tree.
            Constructor takes as input:
                * label: the label of the leaf
            Attributes:
                * elements: signals in the dataset being classified in this leaf
        """
        def __init__(self,label,elements):
            self.label = label
            self.elements = elements


    def allSameClass(self,set_trajectories):
        curr_class = self.dict_trajectories_classes[set_trajectories[0]]
        for other_trajectory in set_trajectories:
            if curr_class != self.dict_trajectories_classes[other_trajectory]:
                return False
        return True


    def partition(self,S,phi_bst):
        S_T = []
        S_F = []
        for s in S:
            if not phi_bst.sat(self.dict_trajectories[s],0):
                S_F.append(s)
            else:
                S_T.append(s)
        return S_T, S_F 


    def log(self, x):
        try:
            return math.log(x,2)
        except ValueError:
            return 0
    
    
    def multi_label_entropy(self,set_trajectories):
        num_classes = {class_i:0 for class_i in self.list_classes}
        for trajectory in set_trajectories:
            for class_i in list(self.dict_trajectories_classes[trajectory]):
                num_classes[class_i] += 1
        freq_classes = {class_i:num_classes[class_i]/len(set_trajectories) for class_i in self.list_classes}
        ent = 0
        for num in freq_classes:
            ent += (freq_classes[num]*self.log(freq_classes[num])) + ( (1-freq_classes[num]) * self.log(1-freq_classes[num]) )
        return ent*-1


    def gain(self,set_trajectories,stl_formula):
        impurityBeforeSplit = self.multi_label_entropy(set_trajectories)
        
        S_T, S_F = self.partition(set_trajectories,stl_formula)
        impurityAfterSplit = 0
        try:
            impurityAfterSplit += (len(S_T)/len(set_trajectories))*self.multi_label_entropy(S_T)
            impurityAfterSplit += (len(S_F)/len(set_trajectories))*self.multi_label_entropy(S_F)
        except ZeroDivisionError:
            return 0
        
        return impurityBeforeSplit - impurityAfterSplit
    
    
    def recursiveGenerateTree(self, set_trajectories, depth=0, path=[]):
        
        #Checks the stop criteria: either the trajectories are all labeled the same way, or the max tree depth is reacher
        if self.allSameClass(set_trajectories):
            if verbose:
                print("return leaf \n")
            return DTLearn.Leaf(self.dict_trajectories_classes[set_trajectories[0]],set_trajectories)

        if depth>self.max_depth:
            if verbose:
                print("return leaf \n")
            sub_dict = {key: self.dict_trajectories_classes[key] for key in set_trajectories}
            return DTLearn.Leaf(Counter(sub_dict.values()).most_common(1)[0][0],set_trajectories)
            
        #Define the lower and upper bounds for alpha, beta, gamma, delta, t1 and t2, respectively
        lb = [self.min_x,self.min_x,self.min_y,self.min_y,self.min_h, self.min_h+1]
        ub = [self.max_x,self.max_x,self.max_y,self.max_y,self.max_h-1, self.max_h]

        #Parameters to optimize
        alpha = (self.max_x-self.min_x)/2
        beta  = (self.max_x-self.min_x)/2
        gamma = (self.max_y-self.min_y)/2
        delta = (self.max_y-self.min_y)/2
        t1    = self.min_h
        t2    = self.max_h/2

        # Define the objective for each primitive (to be maximized)
        # \Box_{[t1,t2]} \neg (\alpha < x < \beta  \wedge \gamma < y < \delta)
        def weight_p1(x):
            alpha,beta,gamma,delta,t1,t2 = x
            # if alpha > beta or gamma > delta or t1>t2 or t2-t1>20 or beta-alpha>3 or delta-gamma>3:
            if alpha > beta or gamma > delta or t1>t2 or t2-t1>10:
                return float("inf")
            return -self.gain(set_trajectories,STLFormula.AlwaysNot_STPredicate2D(0,1,alpha,beta,gamma,delta,int(round(t1)),int(round(t2))))
        
        # \Box_{[t1,t2]} (\alpha < x < \beta  \wedge \gamma < y < \delta)
        def weight_p2(x):
            alpha,beta,gamma,delta,t1,t2 = x
            # if alpha > beta or gamma > delta or t1>t2 or t2-t1>20 or beta-alpha>3 or delta-gamma>3:
            if alpha > beta or gamma > delta or t1>t2 or t2-t1>10:
                return float("inf")
            return -self.gain(set_trajectories,STLFormula.Always_STPredicate2D(0,1,alpha,beta,gamma,delta,int(round(t1)),int(round(t2))))
        
        # \diamondsuit_{[t1,t2]} (\alpha < x < \beta  \wedge \gamma < y < \delta)
        def weight_p3(x):
            alpha,beta,gamma,delta,t1,t2 = x
            # if alpha > beta or gamma > delta or t1>t2 or t2-t1>20 or beta-alpha>3 or delta-gamma>3:
            if alpha > beta or gamma > delta or t1>t2 or t2-t1>10:
                return float("inf")
            return -self.gain(set_trajectories,STLFormula.Eventually_STPredicate2D(0,1,alpha,beta,gamma,delta,int(round(t1)),int(round(t2))))
        
        #Optimize each primitive parameter using particle swarm
        if verbose:
            print("Optimizing Box_{[t1,t2]} neg (alpha < x < beta  wedge gamma < y < delta)")
        xopt_primitive_1, fopt_primitive_1 = pso(weight_p1, lb, ub, debug=True,maxiter=200,swarmsize=1000, max_no_improvement=20)
        if verbose:
            print("Optimizing Box_{[t1,t2]} (alpha < x < beta  wedge gamma < y < delta)")
        xopt_primitive_2, fopt_primitive_2 = pso(weight_p2, lb, ub, debug=True,maxiter=200,swarmsize=1000, max_no_improvement=20)
        if verbose:
            print("Optimizing diamondsuit_{[t1,t2]} (alpha < x < beta  wedge gamma < y < delta)")
        xopt_primitive_3, fopt_primitive_3 = pso(weight_p3, lb, ub, debug=True,maxiter=200,swarmsize=1000, max_no_improvement=20)

        #Instatiate each valued primitive and find the best
        p1 = STLFormula.AlwaysNot_STPredicate2D(0,1,xopt_primitive_1[0],xopt_primitive_1[1],xopt_primitive_1[2],xopt_primitive_1[3],int(round(xopt_primitive_1[4])),int(round(xopt_primitive_1[5])))
        p2 = STLFormula.Always_STPredicate2D(0,1,xopt_primitive_2[0],xopt_primitive_2[1],xopt_primitive_2[2],xopt_primitive_2[3],int(round(xopt_primitive_2[4])),int(round(xopt_primitive_2[5])))
        p3 = STLFormula.Eventually_STPredicate2D(0,1,xopt_primitive_3[0],xopt_primitive_3[1],xopt_primitive_3[2],xopt_primitive_3[3],int(round(xopt_primitive_3[4])),int(round(xopt_primitive_3[5])))

        #If using the STL-difference method    
        #Compute the difference between found STL formulae and STL formulae in the path
        if stl_diff:
            diff_p1_path = [p1]
            for stl_path in path:
                diff_p1_path = diff_stl_formulae(diff_p1_path, stl_path)
            diff_p2_path = [p2]
            for stl_path in path:
                diff_p2_path = diff_stl_formulae(diff_p2_path, stl_path)
            diff_p3_path = [p3]
            for stl_path in path:
                diff_p3_path = diff_stl_formulae(diff_p3_path, stl_path)
            
            dict_id_gain = {}
            dict_id_stl  = {}
            id_d = 0
            for d in diff_p1_path:
                dict_id_stl[id_d]  = d
                dict_id_gain[id_d] = -self.gain(set_trajectories,d)
                id_d += 1
            for d in diff_p2_path:
                dict_id_stl[id_d]  = d
                dict_id_gain[id_d] = -self.gain(set_trajectories,d)
                id_d += 1
            for d in diff_p3_path:
                dict_id_stl[id_d]  = d
                dict_id_gain[id_d] = -self.gain(set_trajectories,d)
                id_d += 1
            
            if min(dict_id_gain.values()) >= 0:
                if verbose:
                    print("return leaf \n")
                sub_dict = {key: self.dict_trajectories_classes[key] for key in set_trajectories}
                return DTLearn.Leaf(Counter(sub_dict.values()).most_common(1)[0][0],set_trajectories)
            
            #Among the STL formulae in the difference between the 3 best primitives and the STL formulae in the path of the decision tree, choose the best 
            phi_best = dict_id_stl[random.choice(list(filter(lambda x: dict_id_gain[x]==min(dict_id_gain.values()), dict_id_gain)))]
            if verbose:
                print("found",phi_best,min(dict_id_gain.values()))

            path.append(phi_best)
        
        #If not using the STL-difference method
        else:    
            if fopt_primitive_1 >= 0.0 and fopt_primitive_2 >= 0.0 and fopt_primitive_3 >= 0.0:
                if verbose:
                    print("return leaf \n")
                sub_dict = {key: self.dict_trajectories_classes[key] for key in set_trajectories}
                return DTLearn.Leaf(Counter(sub_dict.values()).most_common(1)[0][0],set_trajectories)
            elif fopt_primitive_2 <= fopt_primitive_1 and fopt_primitive_2 <= fopt_primitive_3:
                phi_best = p2
                if verbose:
                    print("found",phi_best,fopt_primitive_2,"\n")
            elif fopt_primitive_1 <= fopt_primitive_2 and fopt_primitive_1 <= fopt_primitive_3:
                phi_best = p1
                if verbose:
                    print("found",phi_best,fopt_primitive_1,"\n")
            elif fopt_primitive_3 <= fopt_primitive_1 and fopt_primitive_3 <= fopt_primitive_2:
                phi_best = p3
                if verbose:
                    print("found",phi_best,fopt_primitive_3,"\n")
            else:
                if verbose:
                    print("return leaf \n")
                sub_dict = {key: self.dict_trajectories_classes[key] for key in set_trajectories}
                return DTLearn.Leaf(Counter(sub_dict.values()).most_common(1)[0][0],set_trajectories)
            
        set_trajectories_positive, set_trajectories_negative = self.partition(set_trajectories,phi_best)
        
        if not set_trajectories_positive or not set_trajectories_negative:
            if verbose:
                print("return leaf \n")
            sub_dict = {key: self.dict_trajectories_classes[key] for key in set_trajectories}
            return DTLearn.Leaf(Counter(sub_dict.values()).most_common(1)[0][0],set_trajectories)
        
        left_node  = self.recursiveGenerateTree(set_trajectories_positive, depth+1, path)
        right_node = self.recursiveGenerateTree(set_trajectories_negative, depth+1, path)
        
        #if left node is a leaf, and right node is a leaf WITH THE SAME majority class, we return a leaf with that majority class instead of a node with 2 leaves with that majority class.
        if left_node.__class__.__name__ == "Leaf" and right_node.__class__.__name__ == "Leaf":
            if left_node.label == right_node.label:
                return DTLearn.Leaf(left_node.label,set_trajectories)

        return DTLearn.Node(phi_best,left_node,right_node,depth)
    
    
    
    
    def recursiveParseTree(trajectory, node):
        if node.__class__.__name__ == "Leaf":
            return node.label
        if node.stl.sat(trajectory,0):
            return DTLearn.recursiveParseTree(trajectory, node.left)
        else:
            return DTLearn.recursiveParseTree(trajectory, node.right)
    
    
    
    def to_string(node,indent=''):
        if node.__class__.__name__ == "Leaf":
            print(indent,node.label)
        else:
            print(indent,node.stl)
            DTLearn.to_string(node.left,indent+'\t')
            DTLearn.to_string(node.right,indent+'\t')
    






# Recursive function to find paths from the root node to every leaf node
def printRootToLeafPaths(node, path, leading_to_class,PATHS):
 
    # if a leaf node is found, print the path
    if node.__class__.__name__ == "Leaf":
        path.append(None)
        if leading_to_class in node.label:
            # print(list(path))
            PATHS.append(list(path)[:-1])
    # recur for the left and right subtree
    else:
        path.append(node.stl)
        printRootToLeafPaths(node.left, path, leading_to_class,PATHS)
        path.pop()
        path.append(STLFormula.Negation(node.stl))
        printRootToLeafPaths(node.right, path, leading_to_class,PATHS)
 
    # backtrack: remove the current node after the left, and right subtree are done
    path.pop()



def printRootToLeafPath(root, leading_to_class):
    PATHS = []
    # list to store root-to-leaf path
    path = deque()
    printRootToLeafPaths(root, path, leading_to_class,PATHS)
    # for p in PATHS:
        # print(p)
    return PATHS



def get_STL_formulae(tree, stl_class):
    paths_in_dt = printRootToLeafPath(root, leading_to_class)

    elements_in_disjunction = []
    for p in paths_in_dt:
        elements_in_disjunction.append(STLFormula.Conjunction(p))
    #TODO: process probabilities of each branch in the tree!!!!!!!
    return STLFormula.Conjunction(elements_in_disjunction,[])



def evaluate(tree, dict_trajectories_classes, dict_trajectories, list_classes):
    y_true = []
    y_pred = []
    
    for trajectory in dict_trajectories_classes:
        true = []
        for i in range(0,len(list_classes)):
            if list_classes[i] in dict_trajectories_classes[trajectory]:
                true.append(1)
            else:
                true.append(0)
        y_true.append(true)
        
        pred = []
        for i in range(0,len(list_classes)):
            if list_classes[i] in DTLearn.recursiveParseTree(dict_trajectories[trajectory], tree):
                pred.append(1)
            else:
                pred.append(0)
        y_pred.append(pred)  
    
    return y_true, y_pred


#both are lists
def diff_stl_formulae(stlformulae, other):
    diff = []
    for stlformula in stlformulae:
        diff.extend(stlformula - other)
    return diff



if __name__ == '__main__':

    dict_trajectories_classes = pickle.load(open("user_study_data/dict_trajectories_classes.pkl", "rb" ))
    dict_trajectories = pickle.load(open("user_study_data/dict_trajectories.pkl", "rb" ))
    list_classes = pickle.load(open("user_study_data/list_classes.pkl", "rb" ))
    
    max_depth, min_x, min_y, min_h, max_x, max_y, max_h = 4, -7, -10, 30, 7, 5, 90
    
    stl_diff = True
    verbose = True
    outputmodel = "user_study_data/output_models/stl_dt_diffs_h5.dill"
    
    # print(list_classes)
    # print(Counter(dict_trajectories_classes.values()))
    # print(Counter(dict_trajectories_classes.values()).most_common(1)[0][0])
    
    
    #Parse CLI parameters and replace default parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"a:b:c:d:e:f:g:h:i:j:k:l:m:n:",["dicttrajectories=","dicttrajectoriesclasses=","listclasses=","minx","miny=","minh=","maxx=","maxy=","maxh=","maxdepth=","stldiff=","verbose=","cm=","outputmodel="])
    except getopt.GetoptError:
        print('some options not filled, will proceed with default parameters for these')
        
    dictbool = {'0':False, '1':True}
    for opt, arg in opts:
        if opt in ("-a", "--dicttrajectories"):
            inputfile = arg
            dict_trajectories_classes = pickle.load( open(inputfile, "rb" ) )
        elif opt in ("-b", "--dicttrajectoriesclasses"):
            inputfile = arg
            dict_trajectories = pickle.load( open(inputfile, "rb" ) )
        elif opt in ("-c", "--listclasses"):
            inputfile = arg
            list_classes = pickle.load( open(inputfile, "rb" ) )
        elif opt in ("-d", "--minx"):
            min_x = float(arg)
        elif opt in ("-e", "--miny"):
            min_y = float(arg)
        elif opt in ("-f", "--minh"):
            min_h = float(arg)
        elif opt in ("-g", "--maxx"):
            max_x = float(arg)
        elif opt in ("-h", "--maxy"):
            max_y = float(arg)
        elif opt in ("-i", "--maxh"):
            max_h = float(arg)
        elif opt in ("-j", "--maxdepth"):
            max_depth = float(arg)
        elif opt in ("-k", "--stldiff"):
            stl_diff = dictbool[arg]
        elif opt in ("-l", "--verbose"):
            verbose = dictbool[arg]
        elif opt in ("-m", "--cm"):
            cm = dictbool[arg]
        elif opt in ("-n", "--outputmodel"):
            outputmodel = dictbool[arg]
    

    
    
    dtlearn = DTLearn(dict_trajectories, dict_trajectories_classes, list_classes, min_x, min_y, min_h, max_x, max_y, max_h, max_depth, stl_diff, verbose)
    tree = dtlearn.recursiveGenerateTree(list(dict_trajectories))
    
    if outputmodel:
        with open(outputmodel, "wb") as pfile:
            dill.dump(tree, pfile)
    
    # tree = dill.load(open("user_study_data/output_models/stl_dt_diffs_h5.dill", "rb" ))
    
    if verbose:
        DTLearn.to_string(tree)

        for c in list_classes:
            print("\n \nspecification for class ",c)
            printRootToLeafPath(tree, c)
        

    y_true, y_pred = evaluate(tree, dict_trajectories_classes, dict_trajectories, list_classes)
    
    print(hamming_loss(np.array(y_true), np.array(y_pred)),example_based_accuracy(np.array(y_true), np.array(y_pred)))
    
    class_names = np.unique([str(list(i)) for i in list(np.concatenate((y_true, y_pred), axis=0))])
    
    if cm:
        plot_cm(y_true, y_pred, class_names)
    
    # dict_classes_trajectory_paper_name = {'[0, 1, 0, 0]':"$\{f\}$",'[0, 1, 1, 1]':"$\{f,w,h\}$",'[0, 1, 1, 0]':"$\{f,w\}$",'[0, 1, 0, 1]':"$\{f,h\}$",'[0, 0, 1, 0]':"$\{w\}$",'[0, 0, 1, 1]':"$\{w,h\}$",'[0, 0, 0, 1]':"$\{h\}$",'[1, 0, 0, 0]':"$\{c\}$"}
    # paper_class_names = [dict_classes_trajectory_paper_name[c] for c in class_names]
    # plot_cm(y_true, y_pred, paper_class_names)
