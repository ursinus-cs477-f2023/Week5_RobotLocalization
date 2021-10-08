"""
Programmer: Chris Tralie

Purpose: To simulate a 2D environment and a robot inside of it with
an omnidirectional range scanner
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import skimage
import skimage.io


def clamp(arrs, boolexpr):
    """
    Clamp numpy arrays to a boolean expression
    Parameters
    ----------
    arrs: Length-k list of [ndarray(N)]
        List of arrays
    boolexpr: ndarray(N, dtype=np.boolean)
        Boolean expression
    Returns
    -------
    Length-k list of [ndarray(M)], where M is
    the number of true evaluations in boolexpr
    """
    return [arr[boolexpr] for arr in arrs]

def ray_intersect_curve(p0, v, X):
    """
    Parameters
    ----------
    p0: ndarray(2)
        Initial point on ray
    v: ndarray(2)
        Direction of ray
    X: ndarray(N, 2)
        Points on the curve
    
    Returns {'t': float,
                    The distance of the closest point of intersection, or 
                    np.inf if there was no intersection
            'p': ndarray(2)
                    Point of intersection
            }
    """
    # vec{a} + s*vec{u} refers to line segments on curve
    # vec{b} + t*vec{v} refers to ray
    u = X[1::, :] - X[0:-1, :]
    u = np.concatenate((u, [u[-1, :]]), axis=0) # Continue the last slope
    # Use Cramer's rule
    # uy*vx - ux*vy
    denom = u[:, 1]*v[0] - u[:, 0]*v[1] 
    # vx*(by - ay) - vy*(bx - ax)
    num_s = v[0]*(p0[1]-X[:, 1]) - v[1]*(p0[0]-X[:, 0])
    # ux*(by - ay) - uy*(bx - ax)
    num_t = u[:, 0]*(p0[1]-X[:, 1]) - u[:, 1]*(p0[0]-X[:, 0])
    
    idx = np.arange(X.shape[0])
    [num_s, num_t, denom, idx] = clamp([num_s, num_t, denom, idx], np.abs(denom) > 0)
    s = num_s/denom
    t = num_t/denom
    # Intersection is within the bounds of the segments
    [s, t, idx] = clamp([s, t, idx], (s >= 0)*(s <= 1))
    tmin = np.inf
    p = np.zeros(2)
    if s.size > 0:
        [s, t, idx] = clamp([s, t, idx], t >= 0)
        if t.size > 0:
            minidx = np.argmin(t)
            tmin = t[minidx]
            s = s[minidx]
            t = t[minidx]
            idx = idx[minidx]
            p = p0 + t*v
    return {'t':tmin, 'p':p}


class Environment(object):
    """
    Attributes
    ----------
    img: ndarray(M, N)
        The image for the maze
    
    """
    def __init__(self, path):
        """
        Setup an environment based on a file
        
        Parameters
        ----------
        path: string    
            Path to environment file, where occupied cells are white
            and blocked cells are black
        """
        img = skimage.io.imread(path)
        img = np.sum(img, axis=2)
        self.img = img
        self.contours = find_contours(img, 0.5)
        self.setup_positions()
    
    def setup_positions(self):
        y, x = np.meshgrid(np.arange(self.img.shape[0]), np.arange(self.img.shape[1]))
        x = x[self.img > 0]
        y = y[self.img > 0]
        self.X = np.array([x, y]).T
        N = x.size
        pos2idx = {(x[i], y[i]):i for i in range(x.size)}
        neighbors = [[i] for i in range(N)]
        for i in range(N):
            xi = x[i]
            yi = y[i]
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighb = (xi+dx, yi+dy)
                if neighb in pos2idx:
                    neighbors[i].append(pos2idx[neighb])
        self.pos2idx = pos2idx
        self.neighbors = neighbors

    
    def plot(self):
        plt.imshow(self.img.T, cmap='gray')
        for X in self.contours:
            plt.plot(X[:, 0], X[:, 1])
        plt.gca().invert_yaxis()


    def get_range_scan(self, pos, res, alpha=0):
        """
        Simulate a range scan
        Parameters
        ----------
        pos: ndarray(2)
            The x/y position of the camera
        res: int
            The number of samples to take
        alpha: float
            Noise amount
        
        Returns
        -------
        X: ndarray(N, 2)
            An 2D point cloud from the point of view of the camera
        XWorld: ndarray(N, 2)
            An 2D point cloud in world coordinates
        """
        scan = np.zeros(res)
        for i, theta in enumerate(np.linspace(0, 2*np.pi, res)):
            V = np.array([np.cos(theta), np.sin(theta)])
            # Find the closest point of intersection
            t = np.inf
            for curve in self.contours:
                res = ray_intersect_curve(pos, V, curve)
                if res['t'] < t:
                    t = res['t']
            if np.isfinite(t):
                scan[i] = max(0, t*(1+alpha*np.random.randn()))
        return scan

    def plot_range_scan(self, x, scan, max_range=None):
        """
        Plot a range scan

        Parameters
        ----------
        x: ndarray(2)
            The x/y position of the camera
        scan: ndarray(N)
            The scan
        max_range: float
            Maximum range of scan (region to which to shrink plot)
        """
        N = len(scan)
        t = np.linspace(0, 2*np.pi, N)
        
        plt.subplot(121)
        self.plot()
        plt.title("Ground Truth Position")
        plt.scatter([x[0]], [x[1]])
        plt.plot([x[0], x[0]+2], [x[1], x[1]], 'C0')
        plt.plot([x[0], x[0]], [x[1], x[1]+2], 'C1')
        plt.subplot(122)
        plt.scatter([0], [0])
        plt.plot(scan*np.cos(t), scan*np.sin(t))
        if max_range:
            plt.xlim([-max_range, max_range])
            plt.ylim([-max_range, max_range])
        plt.title("Range Scan")
        
    
    def simulate_trajectory(self, xs):
        """
        Simulate a trajectory which is a shortest path between a set
        of points on the map

        Parameters
        ----------
        x: list of [float, float]
            Location of points

        """
        from collections import deque
        X = []
        for i in range(len(xs)-1):
            ## BFS
            x0 = tuple(xs[i])
            if not x0 in self.pos2idx:
                print("Error: ", x0, "is not a position on the grid")
                return
            x1 = tuple(xs[i+1])
            if not x1 in self.pos2idx:
                print("Error: ", x1, "is not a position on the grid")
                return
            i1 = self.pos2idx[x0]
            i2 = self.pos2idx[x1]
            visited = set([])
            queue = deque()
            queue.append((i1, None))
            found = False
            state = None
            while len(queue) > 0 and not found:
                (i, prev) = queue.popleft()
                if i == i2:
                    found = True
                    state = (i, prev)
                else:
                    for n in self.neighbors[i]:
                        if not n in visited:
                            visited.add(n)
                            queue.append((n, (i, prev)))
            Xi = []
            while state[-1]:
                Xi.append(self.X[state[0], :])
                state = state[-1]
            Xi.reverse()
            X += Xi
        return np.array(X)