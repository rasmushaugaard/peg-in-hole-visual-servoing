from typing import Callable, Union
import bisect

import cv2
import numpy as np

COLORS = {
    'r': (255, 0, 0, 255),
    'g': (0, 255, 0, 255),
    'b': (0, 0, 255, 255),
    'k': (0, 0, 0, 255),
    'w': (255, 255, 255, 255),
}


def draw_points(img, points, c: Union[str, tuple] = 'r'):
    if isinstance(c, str):
        c = COLORS[c]
    for i, p in enumerate(points):
        cv2.drawMarker(img, tuple(p[::-1]), c, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)


def gui_select_vector(window_title: str, grab_frame_cb: Callable[[], np.ndarray], first_point=None,
                      roi=None, arrow=False, destroy_window_on_end=True) -> np.ndarray:
    points = [] if first_point is None else [tuple(first_point)]
    mouse_pos = [None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and len(points) == 1:
            mouse_pos[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, mouse_callback)
    left, upper, right, lower = roi or (0, 0, None, None)
    while True:
        cv2.waitKey(16)
        img = grab_frame_cb()[upper:lower, left:right].copy()
        if mouse_pos[0]:
            cv2.arrowedLine(img, points[0], mouse_pos[0], (255, 0, 0), 2, tipLength=.2 if arrow else 0.)
        cv2.imshow(window_title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if len(points) > 1:
            break
    if destroy_window_on_end:
        cv2.destroyWindow(window_title)
    return np.array(points) + (left, upper)


def bisect_closest(a, x):
    idx = bisect.bisect(a, x)
    if idx == len(a):
        return idx - 1
    if idx > 0:
        if x - a[idx - 1] < a[idx] - x:
            idx = idx - 1
    return idx


def closest_point_to_lines(line_points: np.ndarray, line_directions: np.ndarray,
                           direction_is_unit=False):
    assert line_points.shape == line_directions.shape
    *pre_shape, N, d = line_points.shape

    if not direction_is_unit:
        line_directions = line_directions / np.linalg.norm(line_directions, axis=-1, keepdims=True)

    A = np.empty((*pre_shape, d, d))
    for k in range(d):
        uk = np.zeros(d)
        uk[k] = 1.
        A[..., k, :] = (line_directions[..., k:k + 1] * line_directions - uk).sum(axis=-2)
    b = line_directions * np.sum(line_points * line_directions, axis=-1, keepdims=True) - line_points
    b = b.sum(axis=-2)
    return np.linalg.solve(A, b)


def _gui_selector_test():
    print(gui_select_vector(
        'hey',
        lambda: np.zeros((500, 500, 3), dtype=np.uint8),
        roi=(300, 0, 500, 200),
        arrow=True,
    ))


def _bisect_closest_test():
    for x, idx in [(-0.1, 0), (0.49, 0), (1.51, 2), (10, 2)]:
        idx_ = bisect_closest([0, 1, 2], x)
        assert idx_ == idx, '{}, expexted {}, but got {}'.format(x, idx, idx_)


def _closest_point_to_lines_test():
    import matplotlib.pyplot as plt
    colors = 'rgbcmyk'[:3]
    n, N, d = len(colors), 3, 2
    line_points, line_directions = np.random.uniform(-1, 1, (2, n, N, d))

    pts = closest_point_to_lines(line_points, line_directions)

    for line_points_, line_directions_, p, c in zip(line_points, line_directions, pts, colors):
        for line_point, line_direction in zip(line_points_, line_directions_):
            x0, y0 = line_point[:2]
            dx, dy = line_direction[:2] * 100
            plt.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], c=c)

        plt.scatter([p[0]], [p[1]], c=c, zorder=3)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect(1)
    plt.show()


if __name__ == '__main__':
    _gui_selector_test()
    _bisect_closest_test()
    _closest_point_to_lines_test()
