

def plot(*objects):
    ax = None
    for obj in objects:
        ax = obj.plot(ax=ax)
