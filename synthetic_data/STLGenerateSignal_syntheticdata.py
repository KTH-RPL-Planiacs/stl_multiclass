from STL import STLFormula
import operator as operatorclass
import pulp as plp
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import random
import pickle

#CONSTANTS
M = 100000
M_up = 100000
M_low = 0.000001

#HARDCODED
#TODO: manage more dimensions
NB_DIMENSIONS = 2



def generate_signal_milp_quantitative(phi,start,rand_area,U,epsilon,OPTIMIZE_ROBUSTNESS):
    """
        Function generating a signal satisfying an STL Formula.
        Takes as input:
            * phi: an STL Formula
            * start: a vector of the form [x0,y0] for the starting point coordinates
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * U: a basic control policy standing for how much can move in 1 time stamp, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
            * epsilon: basic control policy parameter
            * OPTIMIZE_ROBUSTNESS: a flag whether the robustness of the generated signal w.r.t. phi has to be maximized or not
        The encoding details of the MILP optimization problem follows the quantitative enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """    
    dict_vars = {}
  
    #objective, maximize robustness
    rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Continuous')
    dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)] = rvar
            
    #Initialize model
    if OPTIMIZE_ROBUSTNESS:
        opt_model = plp.LpProblem("MIP Model", plp.LpMaximize)
        opt_model += rvar
    else:
        opt_model = plp.LpProblem("MIP Model")
    

    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(NB_DIMENSIONS)),rand_area[0],rand_area[1],plp.LpContinuous)

    #the start is specified
    opt_model += s[0][0] == start[0]
    opt_model += s[0][1] == start[1]
    
    #basic control policy, i.e. how much can move in 1 time stamp
    #\forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-epsilon,U+epsilon)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-epsilon,U+epsilon)
        
        
    #recursive function
    def model_phi(phi,t,opt_model):
        if isinstance(phi, STLFormula.Predicate):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu == rvar
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu == rvar
            
        elif isinstance(phi, STLFormula.TrueF):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            opt_model += rvar >= M            
            
        elif isinstance(phi, STLFormula.FalseF):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            opt_model += rvar <= -M
            
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)
            
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi(phi.first_formula,t,opt_model)
            model_phi(phi.second_formula,t,opt_model)
            
            try:
                pvar1 = dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.first_formula))+'_t_'+str(t)] = pvar1       
            try:
                pvar2 = dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.second_formula))+'_t_'+str(t)] = pvar2
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            
            opt_model += pvar1+pvar2 == 1 #(3)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] #(4)
            opt_model += rvar >= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] #(4)
            opt_model += dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] - (1 - pvar1)*M <= rvar <= dict_vars['r_'+str(id(phi.first_formula))+'_t_'+str(t)] + (1 - pvar1)*M #(5)
            opt_model += dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] - (1 - pvar2)*M <= rvar <= dict_vars['r_'+str(id(phi.second_formula))+'_t_'+str(t)] + (1 - pvar2)*M #(5)

        elif isinstance(phi,STLFormula.Always):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                     
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Eventually):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi(phi.formula,t_i,opt_model)
                
                try:
                    pvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                    
                opt_model += rvar >= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] #(4)
                opt_model += dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] - (1 - pvar_i)*M <= rvar <= dict_vars['r_'+str(id(phi.formula))+'_t_'+str(t_i)] + (1 - pvar_i)*M #(5)
            opt_model += plp.lpSum([dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)]) == 1 #(3)
            
        elif isinstance(phi,STLFormula.Negation):
            try:
                rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                rvar = plp.LpVariable('r_'+str(id(phi))+'_t_'+str(t),cat='Continuous')
                dict_vars['r_'+str(id(phi))+'_t_'+str(t)] = rvar
            model_phi(phi.formula,t,opt_model)
            try:
                rvar_i = dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                rvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['p_'+str(id(phi.formula))+'_t_'+str(t)] = rvar_i
            opt_model += rvar == -rvar_i
    
    
    model_phi(phi,phi.horizon,opt_model)
    rvar = dict_vars['r_'+str(id(phi))+'_t_'+str(phi.horizon)]
    opt_model += rvar >= 0 
    
    opt_model.solve(plp.GUROBI_CMD(msg=False))

    if s[0][0].varValue == None:
        raise Exception("")
    
    return [[s[j][i].varValue for i in range(NB_DIMENSIONS)] for j in range(phi.horizon+1)]
    
    
    


def generate_signal_milp_boolean(phi,start,rand_area,U,epsilon):
    """
        Function generating a signal satisfying an STL Formula.
        Takes as input:
            * phi: an STL Formula
            * start: a vector of the form [x0,y0] for the starting point coordinates
            * rand_area: the domain on which signals are generated. rand_area = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
            * U: a basic control policy standing for how much can move in 1 time stamp, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
            * epsilon: basic control policy parameter
        The encoding details of the MILP optimization problem follows the boolean enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
    """    
    dict_vars = {}
  
    #satisfaction of phi
    zvar1 = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(phi.horizon),cat='Binary')
    dict_vars['z1_'+str(id(phi))+'_t_'+str(phi.horizon)] = zvar1

    opt_model = plp.LpProblem("MIP Model")
  
            
    #We want to optimize a signal. The lower and upperbounds are specified by the random area.
    s = plp.LpVariable.dicts("s",(range(phi.horizon+1),range(NB_DIMENSIONS)),rand_area[0],rand_area[1],plp.LpContinuous)
       
    #the start is specified
    opt_model += s[0][0] == start[0]
    opt_model += s[0][1] == start[1]
    
    #control policy
    for t in range(0,phi.horizon):
        opt_model += s[t+1][0]-s[t][0] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][0]-s[t][0]) <= random.uniform(U-epsilon,U+epsilon)
        opt_model += s[t+1][1]-s[t][1] <= random.uniform(U-epsilon,U+epsilon)
        opt_model += -(s[t+1][1]-s[t][1]) <= random.uniform(U-epsilon,U+epsilon)   
       
    #recursive function
    def model_phi1(phi,t,opt_model):
        if isinstance(phi, STLFormula.TrueF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1
        elif isinstance(phi, STLFormula.FalseF):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 0
        if isinstance(phi, STLFormula.Predicate):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                opt_model += s[t][phi.pi_index_signal] - phi.mu <= M_up*zvar-M_low
                opt_model += -(s[t][phi.pi_index_signal] - phi.mu) <= M_up*(1-zvar)-M_low
            else:
                opt_model += -s[t][phi.pi_index_signal] + phi.mu <= M_up*zvar-M_low
                opt_model += -(-s[t][phi.pi_index_signal] + phi.mu) <= M_up*(1-zvar)-M_low
        elif isinstance(phi, STLFormula.Negation):
            model_phi1(phi.formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t)] = zvar1 
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar == 1-zvar1
        elif isinstance(phi, STLFormula.Conjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar <= zvar1
            opt_model += zvar <= zvar2
            opt_model += zvar >= 1-2+zvar1+zvar2
        elif isinstance(phi, STLFormula.Disjunction):
            model_phi1(phi.first_formula,t,opt_model)
            model_phi1(phi.second_formula,t,opt_model)
            try:
                zvar1 = dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)]
            except KeyError:
                zvar1 = plp.LpVariable('z1_'+str(id(phi.first_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.first_formula))+'_t_'+str(t)] = zvar1       
            try:
                zvar2 = dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)]
            except KeyError:
                zvar2 = plp.LpVariable('z1_'+str(id(phi.second_formula))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi.second_formula))+'_t_'+str(t)] = zvar2
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            opt_model += zvar >= zvar1
            opt_model += zvar >= zvar2
            opt_model += zvar <= zvar1+zvar2
        elif isinstance(phi,STLFormula.Always):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                opt_model += zvar <= zvar_i
            opt_model += zvar >= 1 - (phi.t2+1-phi.t1) + plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
        elif isinstance(phi,STLFormula.Eventually):
            try:
                zvar = dict_vars['z1_'+str(id(phi))+'_t_'+str(t)]
            except KeyError:
                zvar = plp.LpVariable('z1_'+str(id(phi))+'_t_'+str(t),cat='Binary')
                dict_vars['z1_'+str(id(phi))+'_t_'+str(t)] = zvar
            for t_i in range(phi.t1,phi.t2+1):
                model_phi1(phi.formula,t_i,opt_model)
                try:
                    zvar_i = dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)]
                except KeyError:
                    zvar_i = plp.LpVariable('z1_'+str(id(phi.formula))+'_t_'+str(t_i),cat='Binary')
                    dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] = pvar_i
                opt_model += zvar >= zvar_i
            opt_model += zvar <= plp.lpSum([dict_vars['z1_'+str(id(phi.formula))+'_t_'+str(t_i)] for t_i in range(phi.t1,phi.t2+1)])
    
    model_phi1(phi,phi.horizon,opt_model)
    
    opt_model += zvar1 == 1
        
    opt_model.solve(plp.GUROBI_CMD(msg=False))
    
    if s[0][0].varValue == None:
        raise Exception("")
    
    return [[s[j][i].varValue for i in range(NB_DIMENSIONS)] for j in range(phi.horizon+1)]

    
    
    
    
if __name__ == '__main__':

    #CONSTANTS
    INDEX_X = 0
    INDEX_Y = 1

    #Definition of STL Formulae
    
    #Phi1
    predicate_x_gt_min2 = STLFormula.Predicate('x',operatorclass.gt,-2.25,INDEX_X)
    predicate_x_le_2 = STLFormula.Predicate('x',operatorclass.le,2.25,INDEX_X)
    predicate_y_gt_min1 = STLFormula.Predicate('y',operatorclass.gt,-1.25,INDEX_Y)
    predicate_y_le_1 = STLFormula.Predicate('y',operatorclass.le,1.25,INDEX_Y)
    phi1 = STLFormula.toNegationNormalForm(STLFormula.Always(STLFormula.Negation(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_min2,predicate_x_le_2),STLFormula.Conjunction(predicate_y_gt_min1,predicate_y_le_1))),0,100),False)
    
    #Phi2
    predicate_x_gt_min6 = STLFormula.Predicate('x',operatorclass.gt,-6,INDEX_X)
    predicate_x_le_min2 = STLFormula.Predicate('x',operatorclass.le,-2,INDEX_X)
    predicate_y_gt_min4 = STLFormula.Predicate('y',operatorclass.gt,-4,INDEX_Y)
    predicate_y_le_min3 = STLFormula.Predicate('y',operatorclass.le,-3,INDEX_Y)
    phi2 = STLFormula.Always(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_min6,predicate_x_le_min2),STLFormula.Conjunction(predicate_y_gt_min4,predicate_y_le_min3)),10,15)
    
    #Phi3
    predicate_x_gt_6 = STLFormula.Predicate('x',operatorclass.gt,6,INDEX_X)
    predicate_x_le_8 = STLFormula.Predicate('x',operatorclass.le,8,INDEX_X)
    predicate_y_gt_min6 = STLFormula.Predicate('y',operatorclass.gt,-6,INDEX_Y)
    predicate_y_le_min2 = STLFormula.Predicate('y',operatorclass.le,-2,INDEX_Y)
    phi3 = STLFormula.Always(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_6,predicate_x_le_8),STLFormula.Conjunction(predicate_y_gt_min6,predicate_y_le_min2)),25,30)
    
    #Phi4
    predicate_x_gt_min6 = STLFormula.Predicate('x',operatorclass.gt,-6,INDEX_X)
    predicate_x_le_min4 = STLFormula.Predicate('x',operatorclass.le,-4,INDEX_X)
    predicate_y_gt_3 = STLFormula.Predicate('y',operatorclass.gt,3,INDEX_Y)
    predicate_y_le_4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    phi4 = STLFormula.Eventually(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_min6,predicate_x_le_min4),STLFormula.Conjunction(predicate_y_gt_3,predicate_y_le_4)),40,80)
    
    #Phi5
    predicate_x_gt_1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_x_le_4 = STLFormula.Predicate('x',operatorclass.le,4,INDEX_X)
    predicate_y_gt_2 = STLFormula.Predicate('y',operatorclass.gt,2,INDEX_Y)
    predicate_y_le_4 = STLFormula.Predicate('y',operatorclass.le,4,INDEX_Y)
    phi5 = STLFormula.Eventually(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_1,predicate_x_le_4),STLFormula.Conjunction(predicate_y_gt_2,predicate_y_le_4)),50,70)
    
    #Phi6
    predicate_x_gt_min3 = STLFormula.Predicate('x',operatorclass.gt,-3,INDEX_X)
    predicate_x_le_min1 = STLFormula.Predicate('x',operatorclass.le,-1,INDEX_X)
    predicate_y_gt_5 = STLFormula.Predicate('y',operatorclass.gt,5,INDEX_Y)
    predicate_y_le_8 = STLFormula.Predicate('y',operatorclass.le,8,INDEX_Y)
    phi6 = STLFormula.Always(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_min3,predicate_x_le_min1),STLFormula.Conjunction(predicate_y_gt_5,predicate_y_le_8)),85,90)
    
    #Phi7
    predicate_x_gt_1 = STLFormula.Predicate('x',operatorclass.gt,1,INDEX_X)
    predicate_x_le_3 = STLFormula.Predicate('x',operatorclass.le,3,INDEX_X)
    predicate_y_gt_5 = STLFormula.Predicate('y',operatorclass.gt,5,INDEX_Y)
    predicate_y_le_8 = STLFormula.Predicate('y',operatorclass.le,8,INDEX_Y)
    phi7 = STLFormula.Always(STLFormula.Conjunction(STLFormula.Conjunction(predicate_x_gt_1,predicate_x_le_3),STLFormula.Conjunction(predicate_y_gt_5,predicate_y_le_8)),95,100)

    
    #The different classes
    c1       = STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi2),phi6)
    c2       = STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi3),STLFormula.Conjunction(phi4,phi7))
    c3       = STLFormula.Conjunction(STLFormula.Conjunction(phi2,phi5),phi7)
    c1_c2    = STLFormula.Conjunction(STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi2),phi3),STLFormula.Conjunction(STLFormula.Conjunction(phi4,phi6),phi7))
    c1_c3    = STLFormula.Conjunction(STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi2),phi5),STLFormula.Conjunction(phi6,phi7))
    c2_c3    = STLFormula.Conjunction(STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi2),STLFormula.Conjunction(phi3,phi4)),STLFormula.Conjunction(phi5,phi7))
    c1_c2_c3 = STLFormula.Conjunction(STLFormula.Conjunction(STLFormula.Conjunction(phi1,phi2),STLFormula.Conjunction(phi3,phi4)),STLFormula.Conjunction(STLFormula.Conjunction(phi5,phi6),phi7))

    #parameters
    start=[0, -7]
    rand_area=[-7.1, 8]
    U = 0.4
    epsilon = 0.1
    
    #generation of 3 trajectories (quantitative no maximization, quantitative with maximization, boolean)
    # trajectory1 = generate_signal_milp_quantitative(c1,start,rand_area,U,epsilon,False)
    # trajectory2 = generate_signal_milp_quantitative(c1,start,rand_area,U,epsilon,True)
    # trajectory1 = generate_signal_milp_boolean(c1,start,rand_area,U,epsilon)
    
    
    list_classes = [1,2,3]

    TRAJECTORY_ID = 0
    dict_trajectories = {}
    dict_trajectories_classes = {}

    
    trajectories_c1 = []
    trajectories_c2 = []
    trajectories_c3 = []
    trajectories_c1_c2 = []
    trajectories_c1_c3 = []
    trajectories_c2_c3 = []
    trajectories_c1_c2_c3 = []
    
    #class c_1
    classes = [1]
    while not len(trajectories_c1) >= 100:
        try:
            trajectory = generate_signal_milp_quantitative(c1,start,rand_area,U,epsilon,True)
            trajectories_c1.append(trajectory)
            print(len(trajectories_c1))
        except Exception:
            pass
    for trajectory in trajectories_c1:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_2
    classes = [2]
    while not len(trajectories_c2) >= 100:
        try:
            trajectory = generate_signal_milp_quantitative(c2,start,rand_area,U,epsilon,True)
            trajectories_c2.append(trajectory)
            print(len(trajectories_c2))
        except Exception:
            pass
    for trajectory in trajectories_c2:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_3
    classes = [3]
    while not len(trajectories_c3) >= 100:
        try:
            trajectory = generate_signal_milp_quantitative(c3,start,rand_area,U,epsilon,True)
            trajectories_c3.append(trajectory)
            print(len(trajectories_c3))
        except Exception:
            pass
    for trajectory in trajectories_c3:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_1_c_2
    classes = [1,2]
    while not len(trajectories_c1_c2) >= 50:
        try:
            trajectory = generate_signal_milp_boolean(c1_c2,start,rand_area,0.8,epsilon)
            trajectories_c1_c2.append(trajectory)
            print(len(trajectories_c1_c2))
        except Exception:
            pass
    for trajectory in trajectories_c1_c2:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_1_c_3
    classes = [1,3]
    while not len(trajectories_c1_c3) >= 50:
        try:
            trajectory = generate_signal_milp_boolean(c1_c3,start,rand_area,0.7,epsilon)
            trajectories_c1_c3.append(trajectory)
            print(len(trajectories_c1_c3))
        except Exception:
            pass
    for trajectory in trajectories_c1_c3:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_2_c_3
    classes = [2,3]
    while not len(trajectories_c2_c3) >= 50:
        try:
            trajectory = generate_signal_milp_quantitative(c2_c3,start,rand_area,0.8,epsilon,True)
            trajectories_c2_c3.append(trajectory)
            print(len(trajectories_c2_c3))
        except Exception:
            pass
    for trajectory in trajectories_c2_c3:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    #class c_1_c_2_c_3
    classes = [1,2,3]
    while not len(trajectories_c1_c2_c3) >= 50:
        try:
            trajectory = generate_signal_milp_boolean(c1_c2_c3,start,rand_area,0.8,epsilon)
            trajectories_c1_c2_c3.append(trajectory)
            print(len(trajectories_c1_c2_c3))
        except Exception:
            pass
    for trajectory in trajectories_c1_c2_c3:
        dict_trajectories[TRAJECTORY_ID] = trajectory
        dict_trajectories_classes[TRAJECTORY_ID] = tuple(sorted(classes))
        TRAJECTORY_ID += 1
    
    
    dict_classes_trajectory = {}
    for value_to_process in dict_trajectories_classes.values():
        dict_classes_trajectory[value_to_process] = [key  for (key, value) in dict_trajectories_classes.items() if value == value_to_process]

    with open("processed_classes/dict_trajectories_classes.pkl", "wb") as pfile:
        pickle.dump(dict_trajectories_classes, pfile)
    with open("processed_classes/dict_trajectories.pkl", "wb") as pfile:
        pickle.dump(dict_trajectories, pfile)
    with open("processed_classes/dict_classes_trajectory.pkl", "wb") as pfile:
        pickle.dump(dict_classes_trajectory, pfile)
    with open("processed_classes/list_classes.pkl", "wb") as pfile:
        pickle.dump(list_classes, pfile)
    
    exit()
    

    #Plot
    plt.clf()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks(list(range(-10,11,2)))
    ax.set_yticks(list(range(-10,11,2)))
    fig.tight_layout()
    codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
    
    def label(xy, text):
        y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
        plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)
    
    path_phi2 = [
        (-6., -4.), # left, bottom
        (-6., -3.), # left, top
        (-2., -3.), # right, top
        (-2., -4.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_2 = Path(path_phi2, codes)
    patch4_2 = patches.PathPatch(path4_2, facecolor='darkgreen',lw=0)
    ax.add_patch(patch4_2)
    plt.text(-8, -4.6,'$\phi_2=\Box_{[10,15]}$')
    
    path_phi3 = [
        (6., -6.), # left, bottom
        (6., -2.), # left, top
        (8., -2.), # right, top
        (8., -6.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_3 = Path(path_phi3, codes)
    patch4_3 = patches.PathPatch(path4_3, facecolor='darkgreen',lw=0)
    ax.add_patch(patch4_3)
    plt.text(6, -6.6,'$\phi_3=\Box_{[25,30]}$')
    
    path_phi4 = [
        (-6., 3.), # left, bottom
        (-6., 4.), # left, top
        (-4., 4.), # right, top
        (-4., 3.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_4 = Path(path_phi4, codes)
    patch4_4 = patches.PathPatch(path4_4, facecolor='palegreen',lw=0)
    ax.add_patch(patch4_4)
    plt.text(-10, 3.25,'$\phi_4=\diamondsuit_{[40,80]}$')
    
    path_phi5 = [
        (1., 2.), # left, bottom
        (1., 4.), # left, top
        (4., 4.), # right, top
        (4., 2.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_5 = Path(path_phi5, codes)
    patch4_5 = patches.PathPatch(path4_5, facecolor='palegreen',lw=0)
    ax.add_patch(patch4_5)
    plt.text(4.1, 2.1,'$\phi_5=\diamondsuit_{[50,70]}$')
    
    path_phi6 = [
        (-3., 5.), # left, bottom
        (-3., 8.), # left, top
        (-1., 8.), # right, top
        (-1., 5.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_6 = Path(path_phi6, codes)
    patch4_6 = patches.PathPatch(path4_6, facecolor='darkgreen',lw=0)
    ax.add_patch(patch4_6)
    plt.text(-6.5, 8.2,'$\phi_6=\Box_{[85,90]}$')
    
    path_phi7 = [
        (1., 5.), # left, bottom
        (1., 8.), # left, top
        (3., 8.), # right, top
        (3., 5.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_7 = Path(path_phi7, codes)
    patch4_7 = patches.PathPatch(path4_7, facecolor='darkgreen',lw=0)
    ax.add_patch(patch4_7)
    plt.text(3, 8.2,'$\phi_7=\Box_{[95,100]}$')
    
    path_phi1 = [
        (-2., -1.), # left, bottom
        (-2., 1.), # left, top
        (2., 1.), # right, top
        (2., -1.), # right, bottom
        (0., 0.), # ignored
        ]
    path4_1 = Path(path_phi1, codes)
    patch4_1 = patches.PathPatch(path4_1, facecolor='mistyrose',lw=0)
    ax.add_patch(patch4_1)
    plt.text(-1.95, -0.25,'$\phi_1=\Box_{[0,100]}\\neg$')
    
    ax.plot([0], [-7], '-r', marker='X')
    
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plt.axis([-10.2, 10.2, -10.2, 10.2])
    plt.grid(True)

    overlapping = 0.5

    # ax.plot([x for (x, y) in trajectory1], [y for (x, y) in trajectory1], '-g', marker='o', label=r'quantitave $\rho='+str(round(c1.robustness(trajectory1,0),3))+'$')
    ax.plot([x for (x, y) in trajectory1], [y for (x, y) in trajectory1], '-g', label=r'$\sigma_1 \in c_1$', alpha=overlapping)
    plt.grid(True)

    ax.plot([x for (x, y) in trajectory2],[y for (x, y) in trajectory2], '-b', label=r'$\sigma_2 \in c_2$', alpha=overlapping)
    plt.grid(True)     
    
    ax.plot([x for (x, y) in trajectory3],[y for (x, y) in trajectory3], '-r', label=r'$\sigma_3 \in c_3$', alpha=overlapping)
    plt.grid(True)            
    
    ax.plot([x for (x, y) in trajectory4],[y for (x, y) in trajectory4], '-c', label=r'$\sigma_4 \in \{c_1,c_2\}$', alpha=overlapping)
    plt.grid(True)            
    
    ax.plot([x for (x, y) in trajectory5],[y for (x, y) in trajectory5], '-m', label=r'$\sigma_5 \in \{c_1,c_3\}$', alpha=overlapping)
    plt.grid(True)            
    
    ax.plot([x for (x, y) in trajectory6],[y for (x, y) in trajectory6], '-y', label=r'$\sigma_6 \in \{c_2,c_3\}$', alpha=overlapping)
    plt.grid(True)            
    
    ax.plot([x for (x, y) in trajectory7],[y for (x, y) in trajectory7], '-k', label=r'$\sigma_7 \in \{c_1,c_2,c_3\}$', alpha=overlapping)
    plt.grid(True)                    

    print("len t1",len(trajectory1))
    print(trajectory1)
    print("len t2",len(trajectory2))
    print(trajectory2)
    print("len t3",len(trajectory3))
    print(trajectory3)
    print("len t4",len(trajectory4))
    print(trajectory4)
    print("len t5",len(trajectory5))
    print(trajectory5)
    print("len t6",len(trajectory6))
    print(trajectory6)
    print("len t7",len(trajectory7))
    print(trajectory7)

    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=3, shadow=True)
    plt.savefig('synthetic_data.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

