#!/usr/bin/env python3
"""Configurable Pinocchio/CasADi IK for the teleoperation nodes.

The original implementation embedded Piper's URDF and locked its seventh and
eighth joints.  RM75 has seven actuated joints, so that assumption silently
discarded one degree of freedom.  This module selects the model and endpoint
from ROS parameters and uses the official RM75 kinematic dimensions by
default.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, Optional

import numpy as np
import pinocchio as pin
try:
    import casadi
    from pinocchio import casadi as cpin
except ImportError:
    # The pip ``pin`` wheels do not always ship Pinocchio's CasADi bindings.
    # Standalone API2 mode falls back to the numerical damped-least-squares
    # solver below, so it still works without a ROS/conda Pinocchio build.
    casadi = None
    cpin = None
try:
    import rospkg
except ImportError:  # Standalone API2 mode does not require ROS packages.
    rospkg = None
try:
    from tf.transformations import quaternion_from_matrix
except ImportError:
    def quaternion_from_matrix(matrix):
        """Return [x, y, z, w] without depending on ROS ``tf``."""
        m = np.asarray(matrix, dtype=float)[:3, :3]
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        return np.array([x, y, z, w], dtype=float)


def matrix_to_xyzrpy(matrix):
    matrix = np.asarray(matrix, dtype=float)
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
    # Clamp the argument to avoid NaNs from small numerical errors.
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2, 0])))
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    """Create an XYZ + fixed-axis RPY homogeneous transform."""
    transformation_matrix = np.eye(4)
    A, B = np.cos(yaw), np.sin(yaw)
    C, D = np.cos(pitch), np.sin(pitch)
    E, F = np.cos(roll), np.sin(roll)
    DE, DF = D * E, D * F
    transformation_matrix[0, :3] = [A * C, A * DF - B * E, B * F + A * DE]
    transformation_matrix[1, :3] = [B * C, A * E + B * DF, B * DE - A * F]
    transformation_matrix[2, :3] = [-D, C * F, C * E]
    transformation_matrix[:3, 3] = [x, y, z]
    return transformation_matrix


def calc_pose_incre(base_pose, pose_data, zero_pose=None):
    """Convert a Quest pose into a robot-relative pose."""
    def _pose(value, name):
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.size != 6 or not np.all(np.isfinite(result)):
            raise ValueError("{} must contain six finite values (xyz+rpy)".format(name))
        return result.tolist()

    base_pose = _pose(base_pose, "base_pose")
    pose_data = _pose(pose_data, "pose_data")
    if zero_pose is None:
        # Kept for compatibility with the original Piper mapping.  RM75 users
        # can set ~teleop_zero_pose to match their mounting/calibration.
        zero_pose = [0.19, 0.0, 0.2, 0.0, 0.0, 0.0]
    zero_pose = _pose(zero_pose, "zero_pose")
    begin_matrix = create_transformation_matrix(*base_pose)
    zero_matrix = create_transformation_matrix(*zero_pose)
    end_matrix = create_transformation_matrix(*pose_data)
    return matrix_to_xyzrpy(zero_matrix.dot(np.linalg.inv(begin_matrix).dot(end_matrix)))


class Arm_IK:
    def __init__(self, name: str = "arm", params=None, urdf_path=None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.name = name
        self._param_prefix = "" if name in ("", "arm") else "{}_".format(name)
        self._params = dict(params or {})
        self.robot_model = str(self._param("robot_model", "rm75")).lower()
        urdf_value = urdf_path if urdf_path is not None else self._param("urdf_path", "")
        urdf_param = str(urdf_value) if urdf_value else ""
        if urdf_param:
            urdf_path = os.path.expanduser(urdf_param)
        elif self.robot_model in ("rm75", "rm75_6f", "rm75_6fb"):
            # Resolve the bundled model relative to this file in standalone
            # API2 mode; ROS installations can still provide package paths.
            urdf_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "urdf", "rm75.urdf"))
        else:
            if rospkg is None:
                raise RuntimeError(
                    "piper IK without ROS requires an explicit urdf_path")
            rospack = rospkg.RosPack()
            package_path = rospack.get_path("piper_description")
            urdf_path = os.path.join(package_path, "urdf", "piper_description.urdf")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError("IK URDF does not exist: {}".format(urdf_path))
        self.urdf_path = urdf_path
        # Pinocchio resolves ``package://`` mesh references relative to the
        # package root, not only the directory containing the URDF.  Include
        # both locations so bundled and official rm_description models work.
        package_dirs = [os.path.dirname(urdf_path)]
        package_root = os.path.dirname(os.path.dirname(urdf_path))
        if package_root and package_root not in package_dirs:
            package_dirs.append(package_root)
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)

        default_locks = [] if self.robot_model.startswith("rm75") else ["joint7", "joint8"]
        locks = self._param("ik_lock_joints", default_locks)
        self.mixed_jointsToLockIDs = [str(x) for x in locks]
        reference = np.zeros(self.robot.model.nq)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=reference)

        default_ee_joint = "joint7" if self.robot_model.startswith("rm75") else "joint6"
        ee_joint_name = str(self._param("ik_ee_joint", default_ee_joint))
        ee_joint_id = self.reduced_robot.model.getJointId(ee_joint_name)
        if ee_joint_id == 0:
            raise ValueError("IK endpoint joint {!r} is not in {}".format(
                ee_joint_name, urdf_path))
        default_rpy = [0.0, 0.0, 0.0] if self.robot_model.startswith("rm75") else [0.0, -1.57, 0.0]
        default_xyz = [0.0, 0.0, 0.0] if self.robot_model.startswith("rm75") else [0.13, 0.0, 0.0]
        ee_xyz = [float(v) for v in self._param("ik_ee_xyz", default_xyz)]
        ee_rpy = [float(v) for v in self._param("ik_ee_rpy", default_rpy)]
        self.last_matrix = create_transformation_matrix(*ee_xyz, *ee_rpy)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame("ee", ee_joint_id,
                      pin.SE3(pin.Quaternion(q[3], q[0], q[1], q[2]),
                              self.last_matrix[:3, 3]),
                      pin.FrameType.OP_FRAME))
        # RobotWrapper.data was created before the operational frame was
        # added; recreate it so oMf contains the new frame in Pinocchio 3.
        self.reduced_robot.data = self.reduced_robot.model.createData()

        # Collision meshes are optional.  The bundled lightweight RM75 model
        # has no mesh geometry; the official rm_description package can be
        # selected separately if collision checking is required.
        self.geom_model = None
        self.geometry_data = None
        if bool(self._param("ik_collision_check", False)):
            try:
                self.geom_model = pin.buildGeomFromUrdf(
                    self.reduced_robot.model, urdf_path, pin.GeometryType.COLLISION,
                    package_dirs=package_dirs)
                self.geometry_data = pin.GeometryData(self.geom_model)
            except Exception as exc:
                self._logwarn("IK collision geometry unavailable: %s", exc)
                self.geom_model = None
                self.geometry_data = None

        default_q = np.zeros(self.reduced_robot.model.nq)
        configured_q = np.asarray(self._param(
            "ik_initial_joint_state", self._param(
                "target_joint_state", default_q.tolist())), dtype=float).reshape(-1)
        if configured_q.size != self.reduced_robot.model.nq or not np.all(np.isfinite(configured_q)):
            self._logwarn(
                "ik_initial_joint_state has the wrong size; using zero pose")
            configured_q = default_q
        self.init_data = np.clip(
            configured_q,
            self.reduced_robot.model.lowerPositionLimit,
            self.reduced_robot.model.upperPositionLimit)
        self.history_data = self.init_data.copy()
        self.vis = None
        self._target_viz = None
        if bool(self._param("ik_visualize", False)):
            self._init_visualizer()

        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.ik_backend = "numeric"
        if cpin is None or casadi is None:
            self._logwarn(
                "Pinocchio CasADi bindings unavailable; using numerical IK")
            self.opti = None
            return

        self.ik_backend = "casadi"
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.error = casadi.Function(
            "error", [self.cq, self.cTf],
            [cpin.log6(self.cdata.oMf[self.gripper_id].inverse() *
                       cpin.SE3(self.cTf)).vector])
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_q_ref = self.opti.parameter(self.reduced_robot.model.nq)
        error_vec = self.error(self.var_q, self.param_tf)
        self.totalcost = casadi.sumsqr(error_vec[:3]) + 0.1 * casadi.sumsqr(error_vec[3:])
        # Small regularization keeps the redundant RM75 wrist near the last
        # solution and makes Quest tracking less jittery.
        # Keep the redundant wrist near the previous solution.  A plain
        # ``sumsqr(q)`` biases every frame toward the all-zero pose and can
        # cause visible wrist drift even when the Quest target is stationary.
        self.regularization = casadi.sumsqr(self.var_q - self.param_q_ref)
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit, self.var_q,
            self.reduced_robot.model.upperPositionLimit))
        self.opti.minimize(20.0 * self.totalcost + 0.01 * self.regularization)
        self.opti.solver("ipopt", {
            "ipopt": {"print_level": 0, "max_iter": int(self._param("ik_max_iter", 50)),
                      "tol": float(self._param("ik_tol", 1e-4))},
            "print_time": False})

    def _param(self, name, default):
        """Read an arm-specific private parameter, then the shared default."""
        specific_name = "{}{}".format(self._param_prefix, name)
        if self._param_prefix and specific_name in self._params:
            return self._params[specific_name]
        if name in self._params:
            return self._params[name]
        try:
            import rospy
            specific = "~{}{}".format(self._param_prefix, name)
            if self._param_prefix and rospy.has_param(specific):
                return rospy.get_param(specific)
            return rospy.get_param("~{}".format(name), default)
        except Exception:
            return default

    @staticmethod
    def _logwarn(message, *args):
        if args:
            message = message % args
        try:
            import rospy
            rospy.logwarn(message)
        except Exception:
            print("WARNING: {}".format(message))

    def _init_visualizer(self):
        try:
            import meshcat.geometry as mg
            from pinocchio.visualize import MeshcatVisualizer
            self.vis = MeshcatVisualizer(self.reduced_robot.model,
                                         self.reduced_robot.collision_model,
                                         self.reduced_robot.visual_model)
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.display(pin.neutral(self.reduced_robot.model))
            self._target_viz = "ee_target"
            self.vis.viewer[self._target_viz].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(position=0.1 * np.array(
                        [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1]], dtype=np.float32),
                                      color=np.array(
                        [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]], dtype=np.float32)),
                    mg.LineBasicMaterial(linewidth=5, vertexColors=True)))
        except Exception as exc:
            self._logwarn("IK visualizer disabled: %s", exc)
            self.vis = None
            self._target_viz = None

    def forward_pose(self, joints):
        """FK in the exact same model/tool frame used by IK."""
        q = np.asarray(joints, dtype=float)
        if q.shape != (self.reduced_robot.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError("invalid FK joints")
        pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        return self.reduced_robot.data.oMf[self.gripper_id].homogeneous.copy()

    def reset_seed(self, joints):
        self.forward_pose(joints)  # validate before changing state
        self.init_data = np.asarray(joints, dtype=float).copy()
        self.history_data = self.init_data.copy()

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        del gripper, motorV  # kept in the public signature for compatibility
        if motorstate is not None:
            self.init_data = np.asarray(motorstate, dtype=float)
        if self.init_data.shape[0] != self.reduced_robot.model.nq:
            self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.init_data = np.clip(
            self.init_data,
            self.reduced_robot.model.lowerPositionLimit,
            self.reduced_robot.model.upperPositionLimit)
        target_pose = np.asarray(target_pose, dtype=float)
        if target_pose.shape != (4, 4):
            raise ValueError("target_pose must be a 4x4 homogeneous matrix")
        if (not np.all(np.isfinite(target_pose)) or
                not np.allclose(target_pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)):
            raise ValueError("target_pose must be a finite homogeneous transform")
        if self.vis is not None and self._target_viz:
            self.vis.viewer[self._target_viz].set_transform(target_pose)
        if self.ik_backend == "numeric":
            return self._ik_numeric(target_pose)

        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, target_pose)
        self.opti.set_value(self.param_q_ref, self.init_data)
        try:
            self.opti.solve_limited()
            sol_q = np.asarray(self.opti.value(self.var_q), dtype=float).reshape(-1)
            max_diff = float(np.max(np.abs(self.history_data - sol_q)))
            if max_diff > math.radians(30.0):
                # Re-seed once after a discontinuity; this prevents a failed
                # hand-tracking frame from making the next frame jump.
                self.init_data = np.zeros_like(sol_q)
            else:
                self.init_data = sol_q
            self.history_data = sol_q
            if self.vis is not None:
                self.vis.display(sol_q)
            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data,
                              sol_q, np.zeros(self.reduced_robot.model.nv),
                              np.zeros(self.reduced_robot.model.nv))
            return sol_q, tau_ff, self.check_self_collision(sol_q)
        except Exception as exc:
            self._logwarn("RM75 IK did not converge: %s", exc)
            return None, "", False

    def _ik_numeric(self, target_pose):
        """Damped least-squares IK used by standalone pip wheels.

        Pinocchio's ``log6`` and frame Jacobian both use the [linear, angular]
        ordering, matching the original CasADi objective.  Keeping this
        implementation here avoids making API2 users install a ROS/conda
        build solely for ``pinocchio.casadi``.
        """
        try:
            target = pin.SE3(np.asarray(target_pose[:3, :3], dtype=float),
                             np.asarray(target_pose[:3, 3], dtype=float))
            q = np.array(self.init_data, dtype=float, copy=True)
            lower = self.reduced_robot.model.lowerPositionLimit
            upper = self.reduced_robot.model.upperPositionLimit
            max_iter = int(self._param("ik_max_iter", 50))
            tol = float(self._param("ik_tol", 1e-4))
            damping = float(self._param("ik_numeric_damping", 1e-3))
            damping = max(damping, 1e-8)
            for _ in range(max_iter):
                pin.forwardKinematics(self.reduced_robot.model,
                                       self.reduced_robot.data, q)
                pin.updateFramePlacements(self.reduced_robot.model,
                                          self.reduced_robot.data)
                current = self.reduced_robot.data.oMf[self.gripper_id]
                error = np.asarray(pin.log6(current.inverse() * target).vector,
                                   dtype=float).reshape(-1)
                if np.linalg.norm(error) < tol:
                    break
                jacobian = pin.computeFrameJacobian(
                    self.reduced_robot.model, self.reduced_robot.data, q,
                    self.gripper_id, pin.ReferenceFrame.LOCAL)
                # Position is more important than orientation for Quest
                # tracking; the vector order is linear then angular.
                weights = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
                weighted_j = weights[:, None] * jacobian
                weighted_error = weights * error
                lhs = weighted_j.dot(weighted_j.T) + damping * damping * np.eye(6)
                dq = weighted_j.T.dot(np.linalg.solve(lhs, weighted_error))
                q = pin.integrate(self.reduced_robot.model, q, 0.7 * dq)
                q = np.clip(q, lower, upper)
            if not np.all(np.isfinite(q)):
                raise ValueError("numerical IK produced non-finite joints")
            # Reaching max_iter is not convergence. Never publish the last
            # iterate of an unreachable target (often a joint-limit pose).
            final_pose = pin.SE3(self.forward_pose(q))
            if np.linalg.norm(pin.log6(final_pose.inverse() * target).vector) >= tol:
                raise ValueError("numerical IK residual exceeds tolerance")
            self.init_data = q
            self.history_data = q.copy()
            if self.vis is not None:
                self.vis.display(q)
            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data,
                              q, np.zeros(self.reduced_robot.model.nv),
                              np.zeros(self.reduced_robot.model.nv))
            return q, tau_ff, self.check_self_collision(q)
        except Exception as exc:
            self._logwarn("numerical RM75 IK did not converge: %s", exc)
            return None, "", False

    def check_self_collision(self, q):
        if self.geom_model is None:
            return False
        try:
            pin.forwardKinematics(self.reduced_robot.model,
                                  self.reduced_robot.data, q)
            pin.updateGeometryPlacements(self.reduced_robot.model,
                                         self.reduced_robot.data,
                                         self.geom_model, self.geometry_data)
            return bool(pin.computeCollisions(self.geom_model,
                                              self.geometry_data, False))
        except Exception as exc:
            self._logwarn("IK collision check failed: %s", exc)
            return False

    def get_dist(self, q, xyz):
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        p = self.reduced_robot.data.oMf[self.gripper_id].translation
        return float(np.linalg.norm(np.asarray(xyz) - p))

    def get_pose(self, q):
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateFramePlacements(self.reduced_robot.model, self.reduced_robot.data)
        return matrix_to_xyzrpy(self.reduced_robot.data.oMf[self.gripper_id].homogeneous)


def rospy_available():
    try:
        import rospy  # noqa: F401
        return True
    except Exception:
        return False
