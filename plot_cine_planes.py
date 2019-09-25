import pathlib
import re
import sys
import webbrowser

import cv2 as cv
import matplotlib as mpl

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import plotly
import pydicom as dicom
from scipy import ndimage
from scipy.interpolate import interpn
import sympy
from sympy import Point, Line, Segment, Plane, Point3D

plotly.offline.init_notebook_mode(connected=True)


class RotatableAxes:
    def __init__(self, fig: mpl.figure.Figure, axes: mpl.axes.Axes,
                 rect_angle: list, rect_reset: list):
        self.fig = fig
        # Suppose that there exists an image in the axes
        self.axes = axes
        self.renderer = self.axes.figure.canvas.get_renderer()
        self.axes_img_instance = self.axes.get_images()[0]
        self.original_axes_img = self.axes_img_instance.make_image(self.renderer, unsampled=True)[0]

        self.axes_for_angle_slider = self.fig.add_axes(rect_angle)
        self.axes_for_reset_button = self.fig.add_axes(rect_reset)
        self.angle_slider = Slider(self.axes_for_angle_slider, 'Angle(Degree)', 0.0,
                                   359.0, valinit=0.0, valstep=0.1)
        self.angle_slider.on_changed(self.update_img)
        self.reset_button = Button(self.axes_for_reset_button, 'Reset')
        self.reset_button.on_clicked(self.reset)

    def connect(self) -> None:
        # connect to all the events we need
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def disconnect(self) -> None:
        # disconnect all the stored connection ids
        self.fig.canvas.mpl_disconnect(self.onclick)

    def update_la_img(self):
        self.axes_img_instance = self.axes.get_images()[0]
        self.original_axes_img = self.axes_img_instance.make_image(self.renderer, unsampled=True)[0]
        self.angle_slider.reset()

    def onclick(self, event: mpl.backend_bases.Event) -> None:
        if self.axes == event.inaxes:
            cur_img = self.axes_img_instance.make_image(self.renderer, unsampled=True)[0]
            if event.button == mpl.backend_bases.MouseButton.LEFT and event.inaxes is not None:
                rotated_image = ndimage.rotate(cur_img, 90.0, reshape=False)
                self.axes_img_instance.set_data(rotated_image)
            elif event.button == mpl.backend_bases.MouseButton.RIGHT and event.inaxes is not None:
                flipped_img = cur_img[:, ::-1]
                self.axes_img_instance.set_data(flipped_img)
            self.axes.figure.canvas.draw()
            self.axes.figure.canvas.flush_events()

    def update_img(self, new_angle: float) -> None:
        axes_images_list = self.axes.get_images()
        rotated_img = ndimage.rotate(self.original_axes_img, new_angle, reshape=False)
        axes_images_list[0].set_data(rotated_img)
        self.axes.figure.canvas.update()
        self.axes.figure.canvas.flush_events()

    def reset(self, event: mpl.backend_bases.Event):
        self.angle_slider.reset()


class FileSliderFig:
    def __init__(self, la_img_list: list, sa_img_list: list,
                 est_la_img_list: list, intersection_points_list: list,
                 la_titles_list: list, sa_titles_list: list, est_la_titles_list: list,
                 la_rect: list, sa_rect: list):
        fig, axes_list = plt.subplots(1, 3, tight_layout=True)
        self.fig = fig
        self.la_axes = axes_list[0]
        self.sa_axes = axes_list[1]
        self.est_la_axes = axes_list[2]
        self.la_img_list = la_img_list
        self.sa_img_list = sa_img_list
        self.est_la_img_list = est_la_img_list
        self.intersection_points_list = intersection_points_list
        self.la_titles_list = la_titles_list
        self.sa_titles_list = sa_titles_list
        self.est_la_titles_list = est_la_titles_list
        self.n_la_images = len(self.la_img_list)
        self.n_sa_images = len(self.sa_img_list)
        self.axes_for_la_file_slider = self.fig.add_axes(la_rect)
        self.axes_for_sa_file_slider = self.fig.add_axes(sa_rect)
        self.la_file_slider = Slider(self.axes_for_la_file_slider, 'LA INDEX',
                                     1.0, self.n_la_images, valinit=1.0, valstep=1.0)
        self.sa_file_slider = Slider(self.axes_for_sa_file_slider, 'SA INDEX',
                                     1.0, self.n_sa_images, valinit=1.0, valstep=1.0)
        self.la_file_slider.on_changed(self.update_la_file)
        self.sa_file_slider.on_changed(self.update_sa_file)
        self.rot_axes = None

    def imshow(self):
        self.la_axes.imshow(self.la_img_list[0], cmap='gray')
        self.sa_axes.imshow(self.sa_img_list[0], cmap='gray')
        p1, p2 = self.intersection_points_list[0][0]
        self.sa_axes.plot((p1[0], p2[0]), (p1[1], p2[1]), 'r--')

        update_axes_range(self.sa_img_list[0], self.sa_axes)

        self.est_la_axes.imshow(self.est_la_img_list[0][0], cmap='gray')
        self.la_axes.title.set_text(self.la_titles_list[0])
        self.sa_axes.title.set_text(self.sa_titles_list[0])
        self.est_la_axes.title.set_text(self.est_la_titles_list[0])
        self.rot_axes = RotatableAxes(self.fig, self.la_axes,
                                 [0.25, 0.06, 0.5, 0.03], [0.72, 0.01, 0.03, 0.03])
        self.rot_axes.connect()

    def show(self):
        self.fig.canvas.manager.window.showMaximized()
        plt.show()

    def update_la_file(self, new_la_slider_val: float) -> None:
        new_la_idx = int(new_la_slider_val - 1.0)
        cur_sa_idx = int(self.sa_file_slider.val - 1.0)

        la_axes_images_list = self.la_axes.get_images()
        la_axes_images_list[0].set_data(self.la_img_list[new_la_idx])
        self.la_axes.set_title(self.la_titles_list[new_la_idx])
        self.la_axes.figure.canvas.draw()
        self.la_axes.figure.canvas.flush_events()
        self.rot_axes.update_la_img()

        sa_axes_lines_list = self.sa_axes.get_lines()
        p1, p2 = self.intersection_points_list[new_la_idx][cur_sa_idx]
        sa_axes_lines_list[0].set_data((p1[0], p2[0]), (p1[1], p2[1]))
        self.sa_axes.figure.canvas.draw()
        self.sa_axes.figure.canvas.flush_events()

        est_la_axes_images_list = self.est_la_axes.get_images()
        est_la_axes_images_list[0].set_data(self.est_la_img_list[new_la_idx][cur_sa_idx])
        self.est_la_axes.set_title(self.est_la_titles_list[new_la_idx])
        self.est_la_axes.figure.canvas.draw()
        self.est_la_axes.figure.canvas.flush_events()

    def update_sa_file(self, new_sa_slider_val: float) -> None:
        cur_la_idx = int(self.la_file_slider.val - 1.0)
        new_sa_idx = int(new_sa_slider_val - 1.0)

        sa_axes_images_list = self.sa_axes.get_images()
        sa_axes_images_list[0].set_data(self.sa_img_list[new_sa_idx])
        self.sa_axes.set_title(self.sa_titles_list[new_sa_idx])

        sa_axes_lines_list = self.sa_axes.get_lines()
        p1, p2 = self.intersection_points_list[cur_la_idx][new_sa_idx]
        sa_axes_lines_list[0].set_data((p1[0], p2[0]), (p1[1], p2[1]))
        sa_plane_range = get_plane_xy_range(self.sa_img_list[new_sa_idx])

        xlim, ylim = sa_plane_range[:, 1], sa_plane_range[:, 0]
        ylim = ylim[::-1]
        self.sa_axes.set(xlim=xlim, ylim=ylim)
        self.sa_axes.figure.canvas.draw()
        self.sa_axes.figure.canvas.flush_events()

        est_la_axes_images_list = self.est_la_axes.get_images()
        est_la_axes_images_list[0].set_data(self.est_la_img_list[cur_la_idx][new_sa_idx])
        self.est_la_axes.figure.canvas.draw()
        self.est_la_axes.figure.canvas.flush_events()


def update_axes_range(img, axes):
    img_range = get_plane_xy_range(img)
    xlim, ylim = img_range[:, 1], img_range[:, 0]
    ylim = ylim[::-1]
    axes.set(xlim=xlim, ylim=ylim)


def get_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray, surfacecolor: np.ndarray,
              colorscale='Greys', showscale: bool = False, reversescale: bool = True) -> plotly.graph_objs.Surface:
    return plotly.graph_objs.Surface(x=x, y=y, z=z, surfacecolor=surfacecolor, cauto=True,
                                     colorscale=colorscale, showscale=showscale, reversescale=reversescale)


def get_trans_mat3D(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    position = get_pos(dcm)
    pixel_spacing = dcm.PixelSpacing
    c_res = pixel_spacing[1]
    r_res = pixel_spacing[0]
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    row_cos_vec, col_cos_vec = orientation[:3], orientation[3:]
    trans_mat = np.array([[row_cos_vec[0] * c_res, col_cos_vec[0] * r_res, 0.0, position[0]],
                          [row_cos_vec[1] * c_res, col_cos_vec[1] * r_res, 0.0, position[1]],
                          [row_cos_vec[2] * c_res, col_cos_vec[2] * r_res, 0.0, position[2]],
                          [0.0, 0.0, 0.0, 1.0]])
    return trans_mat


def get_trans_mat2D(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    pixel_spacing = dcm.PixelSpacing
    c_res = pixel_spacing[1]
    r_res = pixel_spacing[0]
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    row_cos_vec, col_cos_vec = orientation[:3], orientation[3:]
    trans_mat2D = np.array([[row_cos_vec[0] * c_res, col_cos_vec[0] * r_res],
                            [row_cos_vec[1] * c_res, col_cos_vec[1] * r_res]])
    return trans_mat2D


def get_trans_constant(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    position = get_pos(dcm)
    trans_mat = np.array([position[0], position[1]])
    return trans_mat


def thru_plane_position(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    """Gets spatial coordinate of image origin whose axis
    is perpendicular to image plane.
    """
    orientation = tuple((float(o) for o in dcm.ImageOrientationPatient))
    position = tuple((float(p) for p in dcm.ImagePositionPatient))
    row_vec, col_vec = orientation[:3], orientation[3:]
    normal_vector = np.cross(row_vec, col_vec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def get_spacing_between_slices(dcm_files: dicom.dataset.FileDataset) -> np.ndarray:
    spacings = np.diff([thru_plane_position(dcm) for dcm in dcm_files])
    spacing_between_slices = np.mean(spacings)
    return spacing_between_slices


def get_pos(dcm: dicom.dataset.FileDataset) -> np.ndarray:
    return np.array([float(p) for p in dcm.ImagePositionPatient])


def sort_by_plane_number(path: pathlib.Path):
    return int((re.split(r'(\d+)', str(path)))[-4])


def get_sorted_SA_plane_names(file_names_list: list) -> list:
    dcm_files = []
    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append((dfile, fname))
    dcm_files = sorted(dcm_files, key=lambda x: thru_plane_position(x[0]))
    _, sorted_SA_file_names = zip(*dcm_files)
    return sorted_SA_file_names


def get_interpolated_img_stack(file_names_list: list) -> np.ndarray:
    dcm_files = []
    cine_img_arr = []
    n_slices = len(file_names_list)

    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append(dfile)

    for dfile in dcm_files:
        cine_img_arr.append(dfile.pixel_array.astype(np.float32))

    n_row, n_col = cine_img_arr[0].shape
    spacing_between_slices = get_spacing_between_slices(dcm_files)
    num_of_inserted_picture = int(round(spacing_between_slices / dcm_files[0].PixelSpacing[0]))

    cine_img_stack = np.dstack(cine_img_arr)
    n_extended_height = ((n_slices - 1) * num_of_inserted_picture + n_slices)

    interpolated_img_stack = []
    for i in range(n_row):
        resized_img = np.expand_dims(cv.resize(cine_img_stack[i], (n_extended_height, n_col),
                                               interpolation=cv.INTER_LINEAR), axis=0)
        interpolated_img_stack.append(resized_img)

    interpolated_img_stack = np.concatenate(interpolated_img_stack, axis=0)

    return interpolated_img_stack


def get_plotly_planes_list(file_names_list: list, n_planes: int = sys.maxsize) -> list:
    dcm_files = []
    planes_list = []
    cine_img_arr = []
    n_slices = min(len(file_names_list), n_planes)

    for fname in file_names_list:
        dfile = dicom.read_file(str(fname))
        dcm_files.append(dfile)

    for dfile in dcm_files:
        cine_img_arr.append(dfile.pixel_array.astype(np.float32))

    n_row, n_col = cine_img_arr[0].shape
    cine_img_stack = np.dstack(cine_img_arr)

    for i in range(n_slices):
        trans_mat = get_trans_mat3D(dcm_files[i])
        new_pos = np.array([[trans_mat @ np.array([k, j, 0.0, 1.0])
                             for k in range(n_col)] for j in range(n_row)])
        plane = get_plane(new_pos[:, :, 0], new_pos[:, :, 1],
                          new_pos[:, :, 2], cine_img_stack[:, :, i])
        planes_list.append(plane)

    return planes_list


def get_n_points_from_img(n: int, img: np.ndarray, cmap='gray') -> list:
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap=cmap)
    points_list = plt.ginput(n, timeout=-1)
    plt.close(fig)
    return points_list


def get_update_menus(sa_plotly_planes_list: list, sa_file_names: list) -> list:
    def get_visibility(i):
        return [x for x in [True if i == j else False for j in range(1, len(sa_plotly_planes_list) + 1)] + [True]]

    def get_plane_buttons():
        plane_buttons = [dict(label="SA" + (re.split(r'(\d+)', str(sa_file_names[i - 1])))[-4],
                              method='update', args=[{'visible': get_visibility(i)}])
                         for i in range(1, len(sa_plotly_planes_list) + 1)]
        return plane_buttons

    n_sa_planes = len(sa_plotly_planes_list)
    update_menus = list([
        dict(type="buttons",
             active=-1,
             buttons=list([
                 dict(label='LA ONLY',
                      method='update',
                      args=[{'visible': [False] * n_sa_planes + [True]}]),
                 dict(label='RESET',
                      method='update',
                      args=[{'visible': [True] * n_sa_planes + [True]}])
             ]) + get_plane_buttons()
             )
    ])
    return update_menus


def plot_planes(planes_list: list, width: int = 1000, height: int = 1000,
                title: str = 'plotly') -> None:
    layout = dict(width=width, height=height, title=title)
    fig = plotly.graph_objs.Figure(data=planes_list, layout=layout)
    plotly.offline.iplot(fig)


def plot_planes_with_buttons(sa_plotly_planes_list: list, la_plotly_planes_list: list, sa_file_names: list,
                             width: int = 1000, height: int = 1000,
                             title: str = 'plotly', filename: str = None) -> None:
    update_menus = get_update_menus(sa_plotly_planes_list, sa_file_names)
    layout = dict(width=width, height=height, title=title, updatemenus=update_menus)
    fig = plotly.graph_objs.Figure(data=sa_plotly_planes_list + la_plotly_planes_list, layout=layout)
    if filename:
        plotly.offline.plot(fig, filename=filename, auto_open=False)
    else:
        plotly.offline.iplot(fig)


def get_file_names_lists(patient_number: str, phase_number: int) -> tuple:
    la_file_names = "*_LA*_ph" + str(phase_number) + ".dcm"
    sa_file_names = "*_SA*_ph" + str(phase_number) + ".dcm"
    la_dir_path = pathlib.Path(patient_number)
    sa_dir_path = pathlib.Path(patient_number)
    la_file_names_list = la_dir_path.glob(la_file_names)
    sa_file_names_list = sa_dir_path.glob(sa_file_names)

    return la_file_names_list, sa_file_names_list


def get_est_la_plane_from_img_stack(points: list, interpolated_img_stack: np.ndarray) -> np.ndarray:
    assert (points[0] != points[1])

    def get_t_val_from_x(x_: np.ndarray) -> np.ndarray:
        return (x_ - points[0][1]) / (points[1][1] - points[0][1])

    def get_t_val_from_y(y_: np.ndarray) -> np.ndarray:
        return (y_ - points[0][0]) / (points[1][0] - points[0][0])

    def get_x_val(t_: np.ndarray) -> np.ndarray:
        return (1 - t_) * points[0][1] + t_ * points[1][1]

    def get_y_val(t_: np.ndarray) -> np.ndarray:
        return (1 - t_) * points[0][0] + t_ * points[1][0]

    epsilon = 1.0
    n_x, n_y, n_z = interpolated_img_stack.shape
    x = np.linspace(0, n_x - 1, n_x)
    y = np.linspace(0, n_y - 1, n_y)
    z = np.linspace(0, n_z - 1, n_z)

    if abs(points[1][1] - points[0][1]) < epsilon:
        t_range_for_x = [-sys.float_info.max, sys.float_info.max]
    else:
        t_range_for_x = [min(get_t_val_from_x(0), get_t_val_from_x(n_x - 1)),
                         max(get_t_val_from_x(0), get_t_val_from_x(n_x - 1))]

    if abs(points[1][0] - points[0][0]) < epsilon:
        t_range_for_y = [-sys.float_info.max, sys.float_info.max]
    else:
        t_range_for_y = [min(get_t_val_from_y(0), get_t_val_from_y(n_y - 1)),
                         max(get_t_val_from_y(0), get_t_val_from_y(n_y - 1))]

    t_range = [max(t_range_for_x[0], t_range_for_y[0]),
               min(t_range_for_x[1], t_range_for_y[1])]

    # By Pythagorean theorem
    n_t = int(np.sqrt(np.square(get_x_val(t_range[0]) - get_x_val(t_range[1])) +
                      np.square(get_y_val(t_range[0]) - get_y_val(t_range[1]))))

    if get_x_val(t_range[0]) > get_x_val(t_range[1]):
        t = np.linspace(t_range[1], t_range[0], n_t)
    else:
        t = np.linspace(t_range[0], t_range[1], n_t)

    new_z = np.linspace(0, n_z - 1, n_z)
    t, new_z = np.meshgrid(t, new_z)
    new_x = get_x_val(t)
    new_y = get_y_val(t)

    la_plane = interpn((x, y, z), interpolated_img_stack,
                       np.dstack((new_x, new_y, new_z)))

    return la_plane


# Suppose that the intersection of two planes is a line
def smart_crop2D(img: np.ndarray, threshold=0.0) -> np.ndarray:
    non_empty_columns = np.where(img.max(axis=0) > threshold)[0]
    non_empty_rows = np.where(img.max(axis=1) > threshold)[0]
    crop_box = (np.min(non_empty_rows), np.max(non_empty_rows),
                np.min(non_empty_columns), np.max(non_empty_columns))
    new_img = np.copy(img[crop_box[0]:crop_box[1] + 1, crop_box[2]:crop_box[3] + 1, ...])
    return new_img


def get_intersection_line3D(lhs_plane: plotly.graph_objs.Surface,
                            rhs_plane: plotly.graph_objs.Surface) -> sympy.Line3D:
    lhs_points = Point3D(lhs_plane.x[0, 0], lhs_plane.y[0, 0], lhs_plane.z[0, 0]), \
                 Point3D(lhs_plane.x[-1, -1], lhs_plane.y[-1, -1], lhs_plane.z[-1, -1]), \
                 Point3D(lhs_plane.x[0, -1], lhs_plane.y[0, -1], lhs_plane.z[0, -1])
    rhs_points = Point3D(rhs_plane.x[0, 0], rhs_plane.y[0, 0], rhs_plane.z[0, 0]), \
                 Point3D(rhs_plane.x[-1, -1], rhs_plane.y[-1, -1], rhs_plane.z[-1, -1]), \
                 Point3D(rhs_plane.x[0, -1], rhs_plane.y[0, -1], rhs_plane.z[0, -1])

    lhs_plane = Plane(*lhs_points)
    rhs_plane = Plane(*rhs_points)

    return lhs_plane.intersection(rhs_plane)


def get_dicom_file(file_name: pathlib.Path) -> dicom.dataset.FileDataset:
    dicom_file = dicom.read_file(str(file_name))
    return dicom_file


# The intersection of two planes is a line
def get_intersection_points2D(intersection_points: list,
                              trans_mat2D: np.ndarray,
                              trans_constant: np.ndarray) -> tuple:
    lhs_point, rhs_point = intersection_points
    new_lhs_point = Point(lhs_point.x, lhs_point.y)
    new_rhs_point = Point(rhs_point.x, rhs_point.y)
    new_lhs_point = new_lhs_point.translate(-trans_constant[0], -trans_constant[1])
    new_rhs_point = new_rhs_point.translate(-trans_constant[0], -trans_constant[1])
    x1 = np.linalg.solve(trans_mat2D,
                         np.array([float(new_lhs_point.x), float(new_lhs_point.y)]))
    x2 = np.linalg.solve(trans_mat2D,
                         np.array([float(new_rhs_point.x), float(new_rhs_point.y)]))

    x = np.array([x1[0], x2[0]])
    y = np.array([x1[1], x2[1]])
    return x, y


def get_plane_xy_range(plane: np.ndarray) -> np.ndarray:
    x_max, y_max = plane.shape
    x_min, y_min = 0.0, 0.0
    return np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float32)


def get_intersection_points2D_with_img(intersection_points: list,
                                       plane_range: np.ndarray) -> tuple:
    x, y = intersection_points
    p1 = Point(x[0], y[0])
    p2 = Point(x[1], y[1])
    intersection_line = Line(p1, p2)

    points1 = Point(plane_range[0, 1], plane_range[0, 0]), Point(plane_range[0, 1], plane_range[1, 0])
    points2 = Point(plane_range[0, 1], plane_range[1, 0]), Point(plane_range[1, 1], plane_range[1, 0])
    points3 = Point(plane_range[1, 1], plane_range[1, 0]), Point(plane_range[1, 1], plane_range[0, 0])
    points4 = Point(plane_range[1, 1], plane_range[0, 0]), Point(plane_range[0, 1], plane_range[0, 0])

    line1 = Segment(*points1)
    line2 = Segment(*points2)
    line3 = Segment(*points3)
    line4 = Segment(*points4)

    result = tuple(filter(lambda x: x != [], intersection_line.intersection(line1) + intersection_line.intersection(
        line2) + intersection_line.intersection(line3) + intersection_line.intersection(line4)))

    return (float(result[0].x), float(result[0].y)), (float(result[1].x), float(result[1].y))


def is_invertible(x: np.ndarray) -> bool:
    return x.shape[0] == x.shape[1] and np.linalg.matrix_rank(x) == x.shape[0]


def main() -> None:
    N_PATIENT, N_PHASE = "DET0000101", 0
    la_file_names_list, sa_file_names_list = get_file_names_lists(N_PATIENT, N_PHASE)
    la_file_names_list = sorted(la_file_names_list, key=sort_by_plane_number)
    sa_file_names_list = get_sorted_SA_plane_names(sa_file_names_list)

    N_LA_PLANES = len(la_file_names_list)
    N_SA_PLANES = len(sa_file_names_list)

    interpolated_img_stack = get_interpolated_img_stack(sa_file_names_list)
    sa_plotly_planes_list = get_plotly_planes_list(sa_file_names_list)

    la_img_list = []
    sa_img_list = [sa_plotly_planes_list[i].surfacecolor for i in range(N_SA_PLANES)]
    estimated_la_img_list = []
    intersection_points_list = []

    la_titles_list = [f"{N_PATIENT}_original_la{str(i + 1)}_ph{str(N_PHASE)}" for i in range(N_LA_PLANES)]
    sa_num_list = [int((re.split(r'(\d+)', str(sa_file_names_list[i])))[-4]) for i in range(N_SA_PLANES)]
    sa_titles_list = [f"{N_PATIENT}_original_sa{sa_num}_ph{str(N_PHASE)}" for sa_num in sa_num_list]
    est_la_titles_list = [f"estimated_la{str(i + 1)}_ph{str(N_PHASE)}" for i in range(N_LA_PLANES)]

    for i, la_file_name in enumerate(la_file_names_list):
        la_plotly_planes_list = get_plotly_planes_list([la_file_name], 1)

        la_dicom_file = get_dicom_file(la_file_name)
        la_trans_mat2D = get_trans_mat2D(la_dicom_file)
        la_trans_constant = get_trans_constant(la_dicom_file)
        la_plane_range = get_plane_xy_range(la_plotly_planes_list[0].surfacecolor)

        estimated_la_img_list.append([])
        intersection_points_list.append([])

        for j, sa_file_name in enumerate(sa_file_names_list):
            intersection_line3D = get_intersection_line3D(sa_plotly_planes_list[j],
                                                          la_plotly_planes_list[0])
            intersection_points3D = intersection_line3D[0].points

            sa_dicom_file = get_dicom_file(sa_file_name)
            sa_trans_mat2D = get_trans_mat2D(sa_dicom_file)
            sa_trans_constant = get_trans_constant(sa_dicom_file)
            sa_plane_range = get_plane_xy_range(sa_plotly_planes_list[j].surfacecolor)
            sa_intersection_points2D = get_intersection_points2D(intersection_points3D,
                                                                 sa_trans_mat2D, sa_trans_constant)
            sa_intersection_points2D_with_img = get_intersection_points2D_with_img(sa_intersection_points2D,
                                                                                   sa_plane_range)
            sa_p1, sa_p2 = sa_intersection_points2D_with_img
            intersection_points_list[i].append([sa_p1, sa_p2])

            if j == 0:
                la_img = la_plotly_planes_list[0].surfacecolor
                if is_invertible(la_trans_mat2D):
                    la_intersection_points2D = get_intersection_points2D(intersection_points3D,
                                                                         la_trans_mat2D, la_trans_constant)
                    la_intersection_points2D_with_img = get_intersection_points2D_with_img(la_intersection_points2D,
                                                                                           la_plane_range)
                    la_p1, la_p2 = la_intersection_points2D_with_img
                    la_img = ndimage.rotate(la_img,
                                            -np.arctan2(la_p2[0] - la_p1[0], la_p2[1] - la_p1[1]) * 180.0 / np.pi,
                                            reshape=False)
                la_img_list.append(la_img)

            estimated_la = get_est_la_plane_from_img_stack(sa_intersection_points2D_with_img,
                                                           interpolated_img_stack)
            estimated_la = estimated_la[::-1]
            estimated_la_img_list[i].append(estimated_la)

        html_file_path = pathlib.Path(f"{str(la_file_name.name).split('.')[0]}.html")
        if not html_file_path.exists():
            plot_planes_with_buttons(sa_plotly_planes_list, la_plotly_planes_list,
                                     sa_file_names_list, width=1000, height=1000,
                                     title="Plotting SA planes with a LA plane",
                                     filename=str(html_file_path))
            print(f"{str(html_file_path)} created")
        webbrowser.open_new(str(html_file_path))

    file_slider_fig = FileSliderFig(la_img_list, sa_img_list, estimated_la_img_list, intersection_points_list,
                                    la_titles_list, sa_titles_list, est_la_titles_list,
                                    [0.25, 0.95, 0.5, 0.03], [0.25, 0.905, 0.5, 0.03])
    file_slider_fig.imshow()
    file_slider_fig.show()


if __name__ == "__main__":
    main()

