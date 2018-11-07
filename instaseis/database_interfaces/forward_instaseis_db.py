#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python library to extract seismograms from a set of wavefields generated by
AxiSEM.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2014
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import numpy as np

from .base_netcdf_instaseis_db import BaseNetCDFInstaseisDB
from . import mesh
from .. import rotations
from ..source import Source


class ForwardInstaseisDB(BaseNetCDFInstaseisDB):
    """
    Forward Instaseis database.
    """
    def __init__(self, db_path, netcdf_files, buffer_size_in_mb=500,
                 read_on_demand=False, *args, **kwargs):
        """
        :param db_path: Path to the Instaseis Database containing
            subdirectories PZ and/or PX each containing a
            ``order_output.nc4`` file.
        :type db_path: str
        :param buffer_size_in_mb: Strain and displacement are buffered to
            avoid repeated disc access. Depending on the type of database
            and the number of components of the database, the total buffer
            memory can be up to four times this number. The optimal value is
            highly application and system dependent.
        :type buffer_size_in_mb: int, optional
        :param read_on_demand: Read several global fields on demand (faster
            initialization) or on initialization (slower
            initialization, faster in individual seismogram extraction,
            useful e.g. for finite sources, default).
        :type read_on_demand: bool, optional
        """
        BaseNetCDFInstaseisDB.__init__(
            self, db_path=db_path, buffer_size_in_mb=buffer_size_in_mb,
            read_on_demand=read_on_demand, *args, **kwargs)
        self._parse_meshes(netcdf_files)

    def _parse_meshes(self, files):
        m1_m = mesh.Mesh(
            files["MZZ"], full_parse=True, strain_buffer_size_in_mb=0,
            displ_buffer_size_in_mb=self.buffer_size_in_mb,
            read_on_demand=self.read_on_demand)
        m2_m = mesh.Mesh(
            files["MXX_P_MYY"], full_parse=False, strain_buffer_size_in_mb=0,
            displ_buffer_size_in_mb=self.buffer_size_in_mb,
            read_on_demand=self.read_on_demand)
        m3_m = mesh.Mesh(
            files["MXZ_MYZ"], full_parse=False, strain_buffer_size_in_mb=0,
            displ_buffer_size_in_mb=self.buffer_size_in_mb,
            read_on_demand=self.read_on_demand)
        m4_m = mesh.Mesh(
            files["MXY_MXX_M_MYY"], full_parse=False,
            strain_buffer_size_in_mb=0,
            displ_buffer_size_in_mb=self.buffer_size_in_mb,
            read_on_demand=self.read_on_demand)
        self.parsed_mesh = m1_m

        MeshCollection_fwd = collections.namedtuple(
            "MeshCollection_fwd", ["m1", "m2", "m3", "m4"])
        self.meshes = MeshCollection_fwd(m1_m, m2_m, m3_m, m4_m)

        self._is_reciprocal = False

    def _get_data(self, source, receiver, components, coordinates,
                  element_info, coords_rotmat=None):
        ei = element_info
        # Collect data arrays and mu in a dictionary.
        data = {}

        mesh = self.parsed_mesh.f["Mesh"]

        # Get mu.
        if not self.read_on_demand:
            mesh_mu = self.parsed_mesh.mesh_mu
        else:
            mesh_mu = mesh["mesh_mu"]

        npol = self.info.spatial_order
        data["mu"] = mesh_mu[ei.gll_point_ids[npol // 2, npol // 2]]

        if not isinstance(source, Source):
            raise NotImplementedError
        if self.info.dump_type != 'displ_only':
            raise NotImplementedError

        if "hybrid" and "strain" in components:
            if not isinstance(source, Source):
                raise NotImplementedError
            if self.info.dump_type != 'displ_only':
                raise NotImplementedError

            if ei.axis:
                G = self.parsed_mesh.G2
                GT = self.parsed_mesh.G1T
            else:
                G = self.parsed_mesh.G2
                GT = self.parsed_mesh.G2T

            # collect strain in final_strain array
            # final_strain is in Voigt notation [ss, pp, zz, sp, sz, zp]
            displ_1, strain_1 = self._get_displacement_and_strain_interp(
                self.meshes.m1, ei.id_elem, ei.gll_point_ids, G, GT,
                ei.col_points_xi, ei.col_points_eta, ei.corner_points,
                ei.eltype, ei.axis, ei.xi, ei.eta)
            displ_2, strain_2 = self._get_displacement_and_strain_interp(
                self.meshes.m2, ei.id_elem, ei.gll_point_ids, G, GT,
                ei.col_points_xi, ei.col_points_eta, ei.corner_points,
                ei.eltype, ei.axis, ei.xi, ei.eta)
            displ_3, strain_3 = self._get_displacement_and_strain_interp(
                self.meshes.m3, ei.id_elem, ei.gll_point_ids, G, GT,
                ei.col_points_xi, ei.col_points_eta, ei.corner_points,
                ei.eltype, ei.axis, ei.xi, ei.eta)
            displ_4, strain_4 = self._get_displacement_and_strain_interp(
                self.meshes.m4, ei.id_elem, ei.gll_point_ids, G, GT,
                ei.col_points_xi, ei.col_points_eta, ei.corner_points,
                ei.eltype, ei.axis, ei.xi, ei.eta)

        else:
            displ_1 = self._get_displacement(self.meshes.m1, ei.id_elem,
                                             ei.gll_point_ids, ei.col_points_xi,
                                             ei.col_points_eta, ei.xi, ei.eta)
            displ_2 = self._get_displacement(self.meshes.m2, ei.id_elem,
                                             ei.gll_point_ids, ei.col_points_xi,
                                             ei.col_points_eta, ei.xi, ei.eta)
            displ_3 = self._get_displacement(self.meshes.m3, ei.id_elem,
                                             ei.gll_point_ids, ei.col_points_xi,
                                             ei.col_points_eta, ei.xi, ei.eta)
            displ_4 = self._get_displacement(self.meshes.m4, ei.id_elem,
                                             ei.gll_point_ids, ei.col_points_xi,
                                             ei.col_points_eta, ei.xi, ei.eta)

        mij = source.tensor / self.parsed_mesh.amplitude
        # mij is [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        # final is in s, phi, z coordinates
        final = np.zeros((displ_1.shape[0], 3), dtype="float64")

        final[:, 0] += displ_1[:, 0] * mij[0]
        final[:, 2] += displ_1[:, 2] * mij[0]

        final[:, 0] += displ_2[:, 0] * (mij[1] + mij[2])
        final[:, 2] += displ_2[:, 2] * (mij[1] + mij[2])

        fac_1 = mij[3] * np.cos(coordinates.phi) + \
            mij[4] * np.sin(coordinates.phi)
        fac_2 = -mij[3] * np.sin(coordinates.phi) + \
            mij[4] * np.cos(coordinates.phi)

        final[:, 0] += displ_3[:, 0] * fac_1
        final[:, 1] += displ_3[:, 1] * fac_2
        final[:, 2] += displ_3[:, 2] * fac_1

        fac_1 = (mij[1] - mij[2]) * np.cos(2 * coordinates.phi) \
            + 2 * mij[5] * np.sin(2 * coordinates.phi)
        fac_2 = -(mij[1] - mij[2]) * np.sin(2 * coordinates.phi) \
            + 2 * mij[5] * np.cos(2 * coordinates.phi)

        final[:, 0] += displ_4[:, 0] * fac_1
        final[:, 1] += displ_4[:, 1] * fac_2
        final[:, 2] += displ_4[:, 2] * fac_1

        rotmesh_colat = np.arctan2(coordinates.s, coordinates.z)

        if "T" in components:
            # need the - for consistency with reciprocal mode,
            # need external verification still
            data["T"] = -final[:, 1]

        if "R" in components:
            data["R"] = final[:, 0] * np.cos(rotmesh_colat) \
                        - final[:, 2] * np.sin(rotmesh_colat)

        if "N" in components or "E" in components or "Z" in components:
            # transpose needed because rotations assume different slicing
            # (ugly)
            final = rotations.rotate_vector_src_to_NEZ(
                final.T, coordinates.phi,
                source.longitude_rad, source.colatitude_rad,
                receiver.longitude_rad, receiver.colatitude_rad).T

            if "N" in components:
                data["N"] = final[:, 0]
            if "E" in components:
                data["E"] = final[:, 1]
            if "Z" in components:
                data["Z"] = final[:, 2]

        if "hybrid" in components:
            if "strain" in components:
                final_strain = np.zeros((strain_1.shape[0], 6), dtype="float64")

                # monopole
                final_strain[:, 0] += strain_1[:, 0] * mij[0]
                final_strain[:, 1] += strain_1[:, 1] * mij[0]
                final_strain[:, 2] += strain_1[:, 2] * mij[0]
                final_strain[:, 4] += strain_1[:, 4] * mij[0]

                # monopole
                final_strain[:, 0] += strain_2[:, 0] * (mij[1] + mij[2])
                final_strain[:, 1] += strain_2[:, 1] * (mij[1] + mij[2])
                final_strain[:, 2] += strain_2[:, 2] * (mij[1] + mij[2])
                final_strain[:, 4] += strain_2[:, 4] * (mij[1] + mij[2])

                # dipoles
                fac_1 = mij[3] * np.cos(coordinates.phi) \
                    + mij[4] * np.sin(coordinates.phi)

                fac_2 = -mij[3] * np.sin(coordinates.phi) \
                    + mij[4] * np.cos(coordinates.phi)

                final_strain[:, 0] += strain_3[:, 0] * fac_1
                final_strain[:, 1] += strain_3[:, 1] * fac_1
                final_strain[:, 2] += strain_3[:, 2] * fac_1
                final_strain[:, 3] += strain_3[:, 3] * fac_2 # review is it
                # definitely in ss pp zz zp zs sp ???
                final_strain[:, 4] += strain_3[:, 4] * fac_1
                final_strain[:, 5] += strain_3[:, 5] * fac_2

                # quadrupoles
                fac_1 = (mij[1] - mij[2]) * np.cos(2 * coordinates.phi) \
                    + 2 * mij[5] * np.sin(2 * coordinates.phi)

                fac_2 = -(mij[1] - mij[2]) * np.sin(2 * coordinates.phi) \
                    + 2 * mij[5] * np.cos(2 * coordinates.phi)

                final_strain[:, 0] += strain_4[:, 0] * fac_1
                final_strain[:, 1] += strain_4[:, 1] * fac_1
                final_strain[:, 2] += strain_4[:, 2] * fac_1
                final_strain[:, 3] += strain_4[:, 3] * fac_2
                final_strain[:, 4] += strain_4[:, 4] * fac_1
                final_strain[:, 5] += strain_4[:, 5] * fac_2

                # rotate final_strain to tpr
                if "local" in components:
                    final_strain = \
                        rotations.hybrid_strain_tensor_src_to_local_cartesian(
                            final_strain, coords_rotmat,
                            coordinates.phi,
                            source.longitude_rad,
                            source.colatitude_rad,
                            receiver.longitude_rad,
                            receiver.colatitude_rad)
                else:
                    final_strain = \
                        rotations.hybrid_strain_tensor_src_to_tpr(
                            final_strain,
                            coordinates.phi,
                            source.longitude_rad,
                            source.colatitude_rad,
                            receiver.longitude_rad,
                            receiver.colatitude_rad)

                strain = {}
                strain['t'] = final_strain[:, 0]
                strain['p'] = final_strain[:, 1]
                strain['r'] = final_strain[:, 2]
                strain['rp'] = final_strain[:, 3]
                strain['rt'] = final_strain[:, 4]
                strain['tp'] = final_strain[:, 5]
                data["strain"] = strain

            # rotate final_displ to tpr
            if "local" in components:
                final_disp = rotations.hybrid_vector_src_to_local_cartesian(
                    final.T, coords_rotmat, coordinates.phi,
                    source.longitude_rad, source.colatitude_rad).T
            else:
                final_disp = rotations.rotate_vector_src_to_tpr(
                    final.T, coordinates.phi, source.longitude_rad,
                    source.colatitude_rad, receiver.longitude_rad,
                    receiver.colatitude_rad).T

            displacement = {}
            displacement['t'] = final_disp[:, 0]
            displacement['p'] = final_disp[:, 1]
            displacement['r'] = final_disp[:, 2]
            data["displacement"] = displacement

            dt = self.info.dt
            data["dt"] = dt

            if not self.read_on_demand:
                mesh_mu = self.parsed_mesh.mesh_mu
                mesh_rho = self.parsed_mesh.mesh_rho
                mesh_lambda = self.parsed_mesh.mesh_lambda
                mesh_xi = self.parsed_mesh.mesh_xi
                mesh_phi = self.parsed_mesh.mesh_phi
                mesh_eta = self.parsed_mesh.mesh_eta

            else:
                mesh_mu = mesh["mesh_mu"]
                mesh_rho = mesh["mesh_rho"]
                mesh_lambda = mesh["mesh_lambda"]
                mesh_xi = mesh["mesh_xi"]
                mesh_phi = mesh["mesh_phi"]
                mesh_eta = mesh["mesh_eta"]

            npol = self.info.spatial_order
            mu = mesh_mu[ei.gll_point_ids[npol // 2, npol // 2]]
            rho = mesh_rho[ei.gll_point_ids[npol // 2, npol // 2]]
            lbda = mesh_lambda[ei.gll_point_ids[npol // 2, npol // 2]]
            xi = mesh_xi[ei.gll_point_ids[npol // 2, npol // 2]]
            phi = mesh_phi[ei.gll_point_ids[npol // 2, npol // 2]]
            eta = mesh_eta[ei.gll_point_ids[npol // 2, npol // 2]]

            params = {'mu': mu, 'rho': rho, 'lambda': lbda, 'xi': xi, 'phi': phi,
                      'eta': eta}

            data["elastic_params"] = params

        return data

    def _get_params(self, element_info):

        ei = element_info

        mesh = self.parsed_mesh.f["Mesh"]

        if not self.read_on_demand:
            mesh_mu = self.parsed_mesh.mesh_mu
            mesh_rho = self.parsed_mesh.mesh_rho
            mesh_lambda = self.parsed_mesh.mesh_lambda
            mesh_xi = self.parsed_mesh.mesh_xi
            mesh_phi = self.parsed_mesh.mesh_phi
            mesh_eta = self.parsed_mesh.mesh_eta

        else:
            mesh_mu = mesh["mesh_mu"]
            mesh_rho = mesh["mesh_rho"]
            mesh_lambda = mesh["mesh_lambda"]
            mesh_xi = mesh["mesh_xi"]
            mesh_phi = mesh["mesh_phi"]
            mesh_eta = mesh["mesh_eta"]

        npol = self.info.spatial_order
        mu = mesh_mu[ei.gll_point_ids[npol // 2, npol // 2]]
        rho = mesh_rho[ei.gll_point_ids[npol // 2, npol // 2]]
        lbda = mesh_lambda[ei.gll_point_ids[npol // 2, npol // 2]]
        xi = mesh_xi[ei.gll_point_ids[npol // 2, npol // 2]]
        phi = mesh_phi[ei.gll_point_ids[npol // 2, npol // 2]]
        eta = mesh_eta[ei.gll_point_ids[npol // 2, npol // 2]]

        params = {'mu': mu, 'rho': rho, 'lambda': lbda, 'xi': xi, 'phi': phi,
                  'eta': eta}

        return params

