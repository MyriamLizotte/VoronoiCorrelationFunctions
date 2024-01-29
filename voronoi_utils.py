import numpy as np
import seaborn as sns
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib import cm, colormaps
from scipy.spatial import ConvexHull

from Poisson_process import Poisson_point_process, Poisson_point_process_cross, calculate_area_rectangle


def Voronoi_cells(points, show=True):
    vor = Voronoi(points, qhull_options='Qbb Qc Qx')
    
    if show:
        fig = voronoi_plot_2d(vor)
#         plt.xlim([-1000, 2000])
#         plt.ylim([-1000, 2000])
        plt.show()
    
#     if [] in vor.regions:
        
#         vor.regions.remove([])
    
    return vor

def make_adj_matrix(vor):
    adj = np.zeros((len(vor.regions),len(vor.regions)))

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
    #         print(vor.regions[i])
    #         print(vor.regions[j])
            seti = set(vor.regions[i])
            if -1 in seti: seti.remove(-1)
            setj = set(vor.regions[j])
            if -1 in setj: setj.remove(-1)
            adj[i,j] = len(seti.intersection(setj)) >= 1 


    return adj

def compute_shortest_path_dist(adj):

    shortest_path_dists = scipy.sparse.csgraph.dijkstra(adj, directed=False)
    
    return shortest_path_dists
    # shortest_path_dists[0,:]
    
    
    
def compute_u_s_all_points(points):

    vor = Voronoi_cells(points, show=False)
    adj = make_adj_matrix(vor)
    shortest_path_dists = compute_shortest_path_dist(adj)

    u_s = {}
    for k in range(int(np.max(shortest_path_dists))):
        u_s[k] = get_k_neighbors_all_points(shortest_path_dists, k)

    return u_s
    
    
def compute_u_s(points):
    u_s_all_points = compute_u_s_all_points
    
    u_s = {}
    for k in u_s_all_points.keys():
        u_s[k] = np.mean(u_s_all_points[k])

    return u_s

def compute_u_s_cross_all_points(points, c1mask, c2mask=None):

    vor = Voronoi_cells(points, show=False)
    adj = make_adj_matrix(vor)
    shortest_path_dists = compute_shortest_path_dist(adj)
    # visualize_voronoi_source(shortest_path_dists, vor, source = 0)

    u_s_cross = {}
    # u_s_21 = {}
    for k in range(int(np.max(shortest_path_dists))):
        if k >= 1:
            u_s_cross[k] = get_k_neighbors_cross_all_points(shortest_path_dists, k, c1mask, c2mask)
    #         u_s_21[k] = avg_k_neighbors_cross(shortest_path_dists, k, c1mask)

    return u_s_cross

    
    
def compute_u_s_cross(points, c1mask, c2mask=None):
    u_s_all_points = compute_u_s_cross_all_points
    
    u_s_cross = {}
    for k in u_s_cross_all_points.keys():
        u_s_cross[k] = np.mean(u_s_cross_all_points[k])

    return u_s_cross


def compute_u_id(density, xMin, xMax, yMin, yMax, N=100):

    dict_all_iters = run_poisson(density, xMin, xMax, yMin, yMax, N=N)
#     hist_k(dict_all_iters)
    u_id = avg_k(dict_all_iters)
    
    return u_id, dict_all_iters
    
    
    
def avg_k_neighbors(shortest_path_dists, k):
    counts = np.count_nonzero(shortest_path_dists == k, axis=1)
    return np.mean(counts)


def get_k_neighbors_all_points(shortest_path_dists, k):
    counts = np.count_nonzero(shortest_path_dists == k, axis=1)
    return counts


def avg_k_neighbors_cross(shortest_path_dists, k, c1mask, c2mask=None):

    if c2mask is None: 
        c2mask = np.logical_not(c1mask)

    # should be symmetric    
        
    A = shortest_path_dists[c1mask,:] [:, c2mask]
    counts = np.count_nonzero(A == k, axis=1)
    
    return np.mean(counts)


def get_k_neighbors_cross_all_points(shortest_path_dists, k, c1mask, c2mask=None):

    if c2mask is None: 
        c2mask = np.logical_not(c1mask)

    # should be symmetric    
        
    A = shortest_path_dists[c1mask,:] [:, c2mask]
    counts = np.count_nonzero(A == k, axis=1)
    
    return counts


def visualize_voronoi_source(shortest_path_dists, vor, xMin=0, xMax=1, yMin=0, yMax=1, source = 0):

    cmap = cm.viridis_r(np.linspace(0,1, int(np.max(shortest_path_dists))))
    # cmap = plt.get_cmap('viridis', int(np.max(shortest_path_dists) ) )
    # print(type(cmap))

    # cmap = cm.get_cmap('Spectral')
    # cmap = colormaps.get_cmap('Spectral')

    # print(shortest_path_dist[source, i])

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes()
    # sm = plt.cm.ScalarMappable(cmap=cmap)

    for i in range(len(vor.regions)):

        seti = set(vor.regions[i])
        if -1 in seti: seti.remove(-1)
    #     print(seti)
        vs = vor.vertices[list(seti)]

        hull = ConvexHull(vs)
        new_vs= vs[hull.vertices]
    #     print(new_vs)

    #     if shortest_path_dists[source, i] < 5 :
    #         if (vs > 0).all() and (vs < 2000).all():
        color = cmap[int(shortest_path_dists[source, i])]
        p = plt.fill(new_vs[:,0], new_vs[:,1], color = color, edgecolor = 'grey', linewidth = 1)

    #     fig.colorbar(p, cax = ax)
    #     fig.colorbar(sm, cax = ax)

    plt.scatter(vor.points[:,0], vor.points[:,1], color='w', marker='.', s=5) # show data points

    plt.xlim([xMin, xMax])
    plt.ylim([yMin, yMax])

#   plt.scatter(vs[:,0], vs[:,1], color = 'k', s=1) # show vertices of voronoi cells

    source_pt = vor.points[np.asarray(vor.point_region == source).nonzero()][0]
    
    idx_source =  vor.points[np.asarray(vor.point_region == source).nonzero()][0]
        
    plt.scatter(source_pt[[0]], source_pt[[1]], color = 'r', s=100, marker='x')

#     plt.show()
    




def poisson_process_average_k(density, xMin, xMax, yMin, yMax, show=False):
    
    xx, yy = Poisson_point_process(density, xMin, xMax, yMin, yMax, show = show)
    poisson_points = np.concatenate((xx,yy),axis=1)

    
    poisson_vor = Voronoi_cells(poisson_points, show=show)
    poisson_adj = make_adj_matrix(poisson_vor)
    poisson_shortest_path_dists = compute_shortest_path_dist(poisson_adj)
    # visualize_voronoi_source(shortest_path_dists, vor, source = 0)

    avgs_k = {}

    for k in range(int(np.max(poisson_shortest_path_dists))):
        avg = avg_k_neighbors(poisson_shortest_path_dists, k)
        avgs_k[k] = avg
#     print(avgs_k)
    return avgs_k


def run_poisson(density, xMin, xMax, yMin, yMax, N=1000, show = False):
    dict_all_iters = {} 

    for key in range(20):
        dict_all_iters[key] = []

    for i in range(N):
        avgs_k = poisson_process_average_k(density, xMin, xMax, yMin, yMax, show = show)

        for key, value in avgs_k.items():
    #         print(key)
    #         print(value)

            dict_all_iters[key].append(value) 

    return dict_all_iters


def hist_k(dict_all_iters):
#     columns = 10
#     rows = int(len(dict_all_iters.keys())/columns)
#     fig, ax_array = plt.subplots(rows, columns,squeeze=False)
#     for i,ax_row in enumerate(ax_array):
#         for j,axes in enumerate(ax_row):
#             axes.set_title('{},{}'.format(i,j))
#             axes.set_yticklabels([])
#             axes.set_xticklabels([])
#     #         axes.plot(you_data_goes_here,'r-')
#     plt.show()
    
    
    
#     list_k = dict_all_iters[k]
#     ax = plt.subplot(2,10, 5)
#     ax.hist(list_k)
#     plt.title(k)
#     plt.show()
        
    fig = plt.figure(figsize=(20, 7))  
    avgs = {}
    for k in dict_all_iters.keys():
        ax = fig.add_subplot(2,10, k+1)
        list_k = dict_all_iters[k]
        ax.hist(list_k)
        plt.title(k)
        
        pvg_k = np.array(list_k).mean()
        avgs[k]=avg_k
#     plt.show()
        


def avg_k(dict_all_iters):
    avgs = {}
    for k in dict_all_iters.keys():
        list_k = dict_all_iters[k]       
        avg_k = np.array(list_k).mean()
        avgs[k]=avg_k        
    return avgs


## functions for cross Voronoi pcf poisson

def poisson_process_average_k_cross(N_c1, N_total, density, xMin, xMax, yMin, yMax, show=False):

    xx, yy, c1mask = Poisson_point_process_cross(N_c1, N_total, density, xMin, xMax, yMin, yMax, show = show)
    poisson_points = np.concatenate((xx,yy),axis=1)
    
    poisson_vor = Voronoi_cells(poisson_points, show=show)
    poisson_adj = make_adj_matrix(poisson_vor)
    poisson_shortest_path_dists = compute_shortest_path_dist(poisson_adj)
    # visualize_voronoi_source(shortest_path_dists, vor, source = 0)


    avgs_k = {}

    for k in range(int(np.max(poisson_shortest_path_dists))):
        avg = avg_k_neighbors_cross(poisson_shortest_path_dists, k, c1mask)
        avgs_k[k] = avg

    return avgs_k

def run_poisson_cross(N_c1, N_total, density, xMin, xMax, yMin, yMax, max_k= 20, N=1000, show=False):
    dict_all_iters = {} 

    for key in range(max_k):
        dict_all_iters[key] = []

    for i in range(N):
        avgs_k = poisson_process_average_k_cross(N_c1, N_total, density, xMin, xMax, yMin, yMax, show = show)

        for key, value in avgs_k.items():
            dict_all_iters[key].append(value) 

    return dict_all_iters
