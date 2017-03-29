"""
A collections of plots
"""
import matplotlib.pyplot as plt


def plot_ramachandran(pose, scatter=True, ax=None):
    """
    Ramachandran plot

    Parameters
    ----------
    pose : protein object
    scatter : boolean
    if True return a scatter plot, if False a hexbin plot
    default True
    ax : axes
        Matplotlib axes. Defaults to None.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots()

    phi = []
    psi = []
    for i in range(len(pose)):
        phi.append(pose.phi(i))
        psi.append(pose.psi(i))

    if scatter:
        ax.scatter(phi, psi)
    else:
        gridsize = 100
        #gridsize = int(2*len(phi)**(1/3))
        # if gridsize < 36:
        #    gridsize = 36
        ax.hexbin(phi, psi, gridsize=gridsize, cmap=plt.cm.summer, mincnt=1)

    ax.set_xlabel('$\phi$', fontsize=16)
    ax.set_ylabel('$\psi$', fontsize=16, rotation=0)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)

    return ax
