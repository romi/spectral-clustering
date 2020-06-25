import numpy as np

def angle(v1, v2, degree):
    # v1 is your first vector
    # v2 is your second vector
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    norm1 = np.sqrt(np.square(v1[0]) + np.square(v1[1]) + np.square(v1[2]))
    norm2 = np.sqrt(np.square(v2[0]) + np.square(v2[1]) + np.square(v2[2]))
    angle = np.arccos(dot / (norm1 * norm2))
    if degree is True:
        angle = np.degrees(angle)
    return angle


if __name__ == '__main__':

    v1 = np.array([1., 0.5, 0.])
    v2 = np.array([0.3,0.3,0.3])
    a = angle(v1,v2,degree=True)

