import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_EAR(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear


def calculate_MAR(mouth_points):
    A = euclidean_distance(mouth_points[2], mouth_points[6])
    C = euclidean_distance(mouth_points[0], mouth_points[4])

    mar = A / C
    return mar