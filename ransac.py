import numpy as np
import random

def estimate_rigid_transformation(X, Y):
        Xmean = X.mean(axis=1).reshape((2, 1))
        Ymean = Y.mean(axis=1).reshape((2, 1))
        XY = np.dot(X - Xmean, (Y - Ymean).T)
        u, s, vt = np.linalg.svd(XY)
        R = np.matrix(np.dot(vt.T, u.T))
        T = Ymean - np.dot(R, Xmean)
        return R, T

def ransac(pts1, pts2, matches, num_to_sample=4, inlier_pct=10):
    iterations = max(100, int(np.log(1 - 0.99) / np.log(1.0 - (inlier_pct / 100.0) ** 4)))

    num_matches = len(matches)

    # put coords in i,j (= y,x in OpenCV notation) order
    pts1 = np.array([pts1[m.queryIdx].pt for m in matches]).T[::-1, :]
    pts2 = np.array([pts2[m.trainIdx].pt for m in matches]).T[::-1, :]
    numpts = pts1.shape[1]

    besterr = abs(pts1).max() ** 2

    for iter in range(iterations):
        indices = np.random.choice(numpts, num_to_sample, replace=False)
        sub1 = pts1[:, indices]
        sub2 = pts2[:, indices]

        R, T = estimate_rigid_transformation(sub1, sub2)
        dists = ((np.dot(R, pts1) - pts2 + T).A ** 2).sum(axis=0)
        err = np.percentile(dists, inlier_pct)
        if err < besterr:
            besterr = err
            bestR = R
            bestT = T
            bestdists = dists
            if err < 1:
                break

    return np.sqrt(besterr), bestR, bestT, np.sqrt(bestdists), iterations
