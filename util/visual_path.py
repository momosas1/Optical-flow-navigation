import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def draw(pointlist):
    for i in range(len(pointlist) - 1):
        # infos is of structure [{'eye_pos':, 'eye_quat':, 'episode':}, {}]
        first = pointlist[i]
        second = pointlist[i + 1]

        plt.plot([first[0], second[0]], [first[1], second[1]], 'k')
    # draw start point as yellow circle and end point as yellow star
    start_point = pointlist[0]
    end_point = pointlist[len(pointlist) - 1]
    plt.plot(start_point[0], start_point[1], color='r', marker='o')
    plt.plot(end_point[0], end_point[1], color='r', marker='*')
    # plt.show()
    print('start_point: {}'.format(start_point))
    print('end_point: {}'.format(end_point))
    ## draw target on image
    '''
    if args.targetX is not None and args.targetY is not None:
    	print('draw target to image ...')
    	plt.plot(args.targetX, args.targetY, color='y', marker='*')'''
    ## save image to the path
    plt.title('{}'.format("path"))
    plt.axis("equal")

    x = MultipleLocator(1)
    y = MultipleLocator(1)
    ax = plt.gca()
    ax.set_aspect(1)
    ax.xaxis.set_major_locator(x)
    ax.yaxis.set_major_locator(y)
    plt.xlim(3,-3)
    plt.ylim(-5,0)
    plt.savefig("path.png")
    plt.show()