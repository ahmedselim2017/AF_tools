from functools import singledispatch
from pathlib import Path
from typing import Any
import math

import pandas as pd

from Bio.PDB.Structure import Structure


def load_structure(path: Path | str) -> Structure:
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".cif":
        from Bio.PDB.MMCIFParser import MMCIFParser
        parser = MMCIFParser()
    elif path.suffix == ".pdb":
        from Bio.PDB.PDBParser import PDBParser
        parser = PDBParser()
    else:
        raise TypeError(f"Unknwon model file types:{path}")
    return parser.get_structure(path, path)


@singledispatch
def calc_radiusofgyration(data: Any) -> float:
    raise NotImplementedError(
        (f"Argument type {type(data)} for data is"
         "not implemented for calc_radiusofgyration function."))


@calc_radiusofgyration.register
def _(data: Structure) -> float:
    cm = data.center_of_mass()

    tot_mass = 0.0
    A = 0.0
    for at in data.get_atoms():
        at_mass = at.mass
        A += at_mass * ((at.get_coord() - cm)**2).sum()
        tot_mass += at_mass
    return math.sqrt(A / tot_mass)


@calc_radiusofgyration.register
def _(data: str) -> float:
    structure = load_structure(data)
    return calc_radiusofgyration(structure)


@calc_radiusofgyration.register
def _(data: pd.Series) -> float:
    return calc_radiusofgyration(data["best_model_path"])


@singledispatch
def fit_ellipse(data: Any) -> tuple:
    raise NotImplementedError((f"Argument type {type(data)} for data is",
                               "not implemented for fit_ellipse function."))


@fit_ellipse.register
def _(data: Path | str) -> tuple:
    return fit_ellipse(load_structure(data))


@fit_ellipse.register
def _(data: pd.Series) -> tuple:
    return fit_ellipse(data["best_model_path"])


@fit_ellipse.register
def _(data: Structure) -> tuple:
    from ellipse import LsqEllipse
    from skspatial.objects import Plane
    import numpy as np

    def _points2XY(points, plane):
        A = plane.normal[0]
        B = plane.normal[1]
        C = plane.normal[2]
        D = -(A * plane.point[0] + B * plane.point[1] + C * plane.point[2])

        cost = C / (np.sqrt(A**2 + B**2 + C**2))
        sint = np.sqrt((A**2 + B**2) / (A**2 + B**2 + C**2))
        u1 = B / np.sqrt(A**2 + B**2)
        u2 = -A / np.sqrt(A**2 + B**2)

        rotation_matrix = np.array((
            (cost + u1**2 * (1 - cost), u1 * u2 * (1 - cost), u2 * sint),
            (u1 * u2 * (1 - cost), cost + u2**2 * (1 - cost), -u1 * sint),
            (-u2 * sint, u1 * sint, cost),
        ))

        points[:, 2] = points[:, 2] - D / C
        points = (rotation_matrix @ points.T).T
        return points[:, :-1]

    res_cms = np.array([x.center_of_mass() for x in list(data.get_residues())])
    res_cms = res_cms - (res_cms.sum(axis=0) / len(res_cms))

    res_cms_plane = Plane.best_fit(res_cms)
    res_cms_projected = np.apply_along_axis(res_cms_plane.project_point, 1,
                                            res_cms)

    res_cms_xy = _points2XY(res_cms_projected, res_cms_plane)
    ell_reg = LsqEllipse().fit(res_cms_xy)
    ell_center, ell_width, ell_height, ell_phi = ell_reg.as_parameters()

    res_ell_vec = res_cms_xy - ell_center
    res_ell_vec_angle = np.arctan2(res_ell_vec[:, 1], res_ell_vec[:, 0])
    res_ell_vec_angle = np.rad2deg(res_ell_vec_angle) % 360
    res_ell_vec_angle = np.sort(res_ell_vec_angle)

    res_ell_vec_angle_diff = []
    for j in range(len(res_ell_vec_angle) - 1):
        a1 = res_ell_vec_angle[j]
        a2 = res_ell_vec_angle[j + 1]

        res_ell_vec_angle_diff.append(min(abs(a1 - a2), 360 - abs(a1 - a2)))

    a1 = res_ell_vec_angle[0]
    a2 = res_ell_vec_angle[-1]
    res_ell_vec_angle_diff.append(min(abs(a1 - a2), 360 - abs(a1 - a2)))

    d_origin = np.linalg.norm(res_cms_xy, axis=1)
    inner_circle_rad = min(d_origin)
    outer_circle_rad = max(d_origin)

    return ell_width, ell_height, outer_circle_rad, inner_circle_rad, max(
        res_ell_vec_angle_diff)
