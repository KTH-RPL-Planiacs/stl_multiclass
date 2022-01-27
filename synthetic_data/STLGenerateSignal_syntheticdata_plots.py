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


#HARDCODED
#TODO: manage more dimensions
NB_DIMENSIONS = 2

    
    
if __name__ == '__main__':

    #CONSTANTS
    INDEX_X = 0
    INDEX_Y = 1

    dict_classes_trajectory_paper_name = {(1,):"\{c_1\}",(2,):"\{c_2\}",(3,):"\{c_3\}",(1, 2):"\{c_1,c_2\}",(1, 3):"\{c_1,c_3\}",(2, 3):"\{c_2,c_3\}",(1, 2, 3):"\{c_1,c_2,c_3\}"}
    dict_trajectories_classes = pickle.load(open("processed_classes/dict_trajectories_classes.pkl", "rb" ))
    dict_trajectories = pickle.load(open("processed_classes/dict_trajectories.pkl", "rb" ))
    dict_classes_trajectory = pickle.load(open("processed_classes/dict_classes_trajectory.pkl", "rb" ))
    list_classes = pickle.load(open("processed_classes/list_classes.pkl", "rb" ))
    
    
    print(list_classes)
    print(dict_classes_trajectory)
    

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
    
    lines = []
    for trajectory in dict_classes_trajectory[(1,)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-g', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c1.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(2,)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-b', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c2.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(3,)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-r', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c3.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(1, 2)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-c', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c1_c2.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(1, 3)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-m', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c1_c3.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(2, 3)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-y', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c2_c3.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()
    
    lines = []
    for trajectory in dict_classes_trajectory[(1, 2, 3)]:
        line_2 = ax.plot([x for (x, y) in dict_trajectories[trajectory]], [y for (x, y) in dict_trajectories[trajectory]], '-k', alpha=overlapping)
        lines.append(line_2)
    plt.savefig('synthetic_data_c1_c2_c3.pdf', bbox_inches='tight')    
    for line in lines:
        l = line.pop(0)
        l.remove()

    exit()

