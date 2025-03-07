from itertools import product

#Some example domains
def Lshape(N, width, height):
    """
    i indexes over x's
    j indexes over y's
    """
    assert width < N and height < N 
    bottom = [(0,j) for j in range(N)]
    left = [(i,0) for i in range(N)]
    top = [(i,N-1) for i in range(width)]
    right_int = [(width,j) for j in range(height, N)]
    top_int = [(i,height) for i in range(width,N)]
    right = [(N-1, j) for j in range(height)]

    total_bndry = bottom+left+top+right_int+top_int+right

    interior = [(i,j)for i in range(1,N-1) for j in range(1,height)]
    interior += [(i,j) for i in range(1,width) for j in range(height,N-1)]

    all_points = [(i,j) for i in range(N) for j in range(N)]
    domain_bndry = interior + total_bndry
    outside = [x for x in all_points if x not in domain_bndry]
    return interior, total_bndry, outside

def LshapeFn(N, width, idxs):
    assert width < N
    x = idxs[0]
    y = idxs[1]
    #Points on the interior
    if N-1>x> width and 0 < y< width:
        return 0
    elif 0<x< width and N-1 > y > width:
        return 0
    elif 0 < x < width and 0 < y < width:
        return 0
    elif x == width and 0 < y < width:
        return 0
    elif y == width and 0 < x < width:
        return 0
    #Points on boundary
    elif x == width and N-1 > y >= width:
        return 1
    elif y == width and N-1 > x >= width:
        return 1
    elif y == 0:
        return 1
    elif x == 0:
        return 1
    elif x == N-1 and 0 < y<= width:
        return 1
    elif y == N-1 and 0 < x<= width:
        return 1
    #Points outside
    else:
        return 2


def squareHole(N,width):
    assert 2 <= width < N
    points = np.arange(0,N,1)
    grid = list(product(points,points))
    #Place near center
    x0 = N//2-1
    y0 = N//2-1
    r = width//2
    hole = [x for x in grid if abs(x[0]-x0) <= r and abs(x[1]-y0)<=r ] #includes bndry
    hole_bndry = [x for x in hole if abs(x[0]-x0)==r or abs(x[1]-y0)==r]
    print(hole_bndry)
    hole_inside = [x for x in hole if (x[0],x[1]) not in hole_bndry ]
    interior = [x for x in grid if x not in hole]

    return interior, hole_bndry, hole_inside


