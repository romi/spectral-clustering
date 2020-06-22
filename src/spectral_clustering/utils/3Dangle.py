import numpy as np

def angle(v1, v2, degree):
    # v1 is your first vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if angle > np.pi:
        angle2 = 2 * np.pi - angle
        angle = angle2
    if degree is True:
        angle = np.degrees(angle)
    return angle



