from typing import Dict, List
from pathlib import Path

import casadi as cs
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from numpy import ndarray
from pinocchio.visualize import MeshcatVisualizer as visualizer
from scipy.linalg import expm
from spatialmath.base import r2q, rotz

from mm_utils import parsing

import hppfcl as fcl  # isort: skip


DEFAULT_SELF_COLLISION_GROUP_SPECS = [
    ("base", ["wrist", "tool"], None),
    ("upper_arm", ["wrist", "tool"], 2),
    ("forearm", ["tool", "rack"], 2),
]
DEFAULT_DETAILED_SELF_COLLISION_GROUP_SPECS = [
    ("base", ["wrist", "tool"], None),
    ("upper_arm", ["wrist", "tool"], None),
    ("forearm", ["tool", "rack"], None),
    ("tool", ["rack"], None),
    ("wrist", ["rack"], None),
]
DEFAULT_STATIC_OBSTACLE_TARGETS = ["base", "wrist", "forearm", "upper_arm"]
DEFAULT_GROUND_TARGETS = ["tool"]
DEFAULT_DETAILED_GROUND_TARGETS = ["wrist", "tool", "forearm"]


def rotation_matrix_from_rpy(rpy):
    """Return a 3x3 rotation matrix from roll, pitch, yaw."""
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def get_robot_collision_groups(robot_config):
    """Resolve collision groups from the new collision model or the legacy config."""
    collision_model = robot_config.get("collision_model", {})
    groups = collision_model.get("groups")
    if groups:
        return {name: list(link_names) for name, link_names in groups.items()}
    return {
        name: list(link_names)
        for name, link_names in robot_config.get("collision_link_names", {}).items()
    }


def signed_distance_sphere_sphere(c1, c2, r1, r2):
    """Signed distance between two spheres.

    Args:
        c1 (ndarray or casadi.SX): Center of sphere 1, size 3.
        c2 (ndarray or casadi.SX): Center of sphere 2, size 3.
        r1 (float): Radius of sphere 1.
        r2 (float): Radius of sphere 2.

    Returns:
        casadi.SX or float: Signed distance between the spheres.
    """
    return cs.norm_2(c1 - c2) - r1 - r2


def signed_distance_half_space_sphere(d, p, n, c, r):
    """Signed distance of a sphere to half space.

    Args:
        d (float): Offset from p along n.
        p (ndarray): Offset of the normal vector.
        n (ndarray): Normal vector of the plane.
        c (ndarray or casadi.SX): Center of the sphere, size 3.
        r (float): Radius of the sphere.

    Returns:
        casadi.SX or float: Signed distance of sphere to half space.
    """

    return (c - p).T @ n - d - r


def signed_distance_sphere_cylinder(c_sphere, c_cylinder, r_sphere, r_cylinder):
    """Signed distance between a sphere and a cylinder (with infinite height).

    Args:
        c_sphere (ndarray): Center of the sphere.
        c_cylinder (ndarray): Center of the cylinder.
        r_sphere (float): Radius of the sphere.
        r_cylinder (float): Radius of the cylinder.

    Returns:
        casadi.SX or float: Signed distance between sphere and cylinder.
    """

    return cs.norm_2(c_sphere[:2] - c_cylinder[:2]) - r_sphere - r_cylinder


class PinocchioInterface:
    """Interface to Pinocchio for robot model and collision checking."""

    def __init__(self, config):
        """Initialize Pinocchio interface.

        Args:
            config (dict): Configuration dictionary with robot and scene parameters.
        """
        # 1. build robot model
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        package_dirs = [str(Path(urdf_path).parent)]
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=package_dirs,
        )

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs=package_dirs
        )


        # 2. add scene model
        if config["scene"]["enabled"]:
            scene_urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            # here models are passed in so that scene models can be appended to robot model
            pin.buildModelFromUrdf(scene_urdf_path, self.model)
            pin.buildGeomFromUrdf(
                self.model,
                scene_urdf_path,
                pin.GeometryType.COLLISION,
                self.collision_model,
                package_dirs=[str(Path(scene_urdf_path).parent)],
            )
            pin.buildGeomFromUrdf(
                self.model,
                scene_urdf_path,
                pin.GeometryType.VISUAL,
                self.visual_model,
                package_dirs=[str(Path(scene_urdf_path).parent)],
            )
        self.addGroundCollisionObject()

        self.collision_link_names = get_robot_collision_groups(config["robot"])
        self.custom_collision_objects = {}
        self._addConfiguredCollisionObjects(config["robot"])
        if config["scene"]["enabled"]:
            self.collision_link_names.update(config["scene"]["collision_link_names"])

    def visualize(self, q):
        """Visualize robot configuration.

        Args:
            q (ndarray): Joint configuration vector.
        """
        viz = visualizer(self.model, self.collision_model, self.visual_model)
        viz.initViewer(open=True)
        viz.viewer.open()
        viz.loadViewerModel()
        viz.display(q)

    def getGeometryObject(self, link_names):
        """Get geometry objects for given link names.

        Args:
            link_names (list or str): Link name(s) to get geometry objects for.

        Returns:
            GeometryObject or list: Geometry object(s) for the link(s).
        """
        objs = []
        for name in link_names:
            if name in self.custom_collision_objects:
                objs.append(self.custom_collision_objects[name])
                continue

            o_id = self.collision_model.getGeometryId(name + "_0")
            if o_id >= self.collision_model.ngeoms:
                o = None
            else:
                o = self.collision_model.geometryObjects[o_id]
            objs.append(o)

        if len(objs) == 1:
            return objs[0]
        else:
            return objs

    def getSignedDistance(self, o1, tf1, o2, tf2):
        """Compute signed distance between two geometry objects.

        Args:
            o1 (GeometryObject): First geometry object.
            tf1 (tuple): Transformation (position, rotation) for object 1.
            o2 (GeometryObject): Second geometry object.
            tf2 (tuple): Transformation (position, rotation) for object 2.

        Returns:
            casadi.SX or float: Signed distance between the objects.
        """
        o1_geo_type = o1.geometry.getNodeType()
        o2_geo_type = o2.geometry.getNodeType()

        if o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_sphere(
                tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_HALFSPACE and o2_geo_type == fcl.GEOM_SPHERE:
            # norm and displacement of the dividing plane
            d = o1.geometry.d
            nw = tf1[1] @ o1.geometry.n
            signed_dist = signed_distance_half_space_sphere(
                d, tf1[0], nw, tf2[0], o2.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_HALFSPACE:
            # norm and displacement of the dividing plane
            d = o2.geometry.d
            nw = tf2[1] @ o2.geometry.n
            signed_dist = signed_distance_half_space_sphere(
                d, tf2[0], nw, tf1[0], o1.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_CYLINDER and o2_geo_type == fcl.GEOM_SPHERE:
            signed_dist = signed_distance_sphere_cylinder(
                tf2[0], tf1[0], o2.geometry.radius, o1.geometry.radius
            )
        elif o1_geo_type == fcl.GEOM_SPHERE and o2_geo_type == fcl.GEOM_CYLINDER:
            signed_dist = signed_distance_sphere_cylinder(
                tf1[0], tf2[0], o1.geometry.radius, o2.geometry.radius
            )
        else:
            raise NotImplementedError(
                "Unsupported signed-distance geometry pair: "
                f"{o1.geometry.getNodeType()} vs {o2.geometry.getNodeType()}"
            )

        return signed_dist

    def addCollisionObjects(self, geoms):
        """Add a list of geometry objects to the collision model.

        Args:
            geoms (list): List of pinocchio GeometryObject.
        """
        for geom in geoms:
            self.collision_model.addGeometryObject(geom)

    def addVisualObjects(self, geoms):
        """Add a list of geometry objects to the visual model.

        Args:
            geoms (list): List of pinocchio GeometryObject.
        """
        for geom in geoms:
            self.visual_model.addGeometryObject(geom)

    def addCollisionPairs(self, pairs, expand_name=True):
        """Add collision pairs to the model.

        Args:
            pairs (list): List of tuples of collision pairs.
            expand_name (bool): Append _0 to link names if True.
        """
        for pair in pairs:
            id1 = self.collision_model.getGeometryId(
                pair[0] + "_0" if expand_name else pair[0]
            )
            id2 = self.collision_model.getGeometryId(
                pair[1] + "_0" if expand_name else pair[1]
            )
            if id1 >= self.collision_model.ngeoms or id2 >= self.collision_model.ngeoms:
                raise ValueError(
                    f"Collision pair {pair} references a geometry object that does not exist"
                )
            self.collision_model.addCollisionPair(pin.CollisionPair(id1, id2))

    def removeAllCollisionPairs(self):
        """Remove all collision pairs from the collision model."""
        self.collision_model.removeAllCollisionPairs()

    def addGroundCollisionObject(self):
        """Add a ground plane collision object."""
        # add a ground plane
        ground_placement = pin.SE3.Identity()
        ground_shape = fcl.Halfspace(np.array([0, 0, 1]), 0)
        ground_geom_obj = pin.GeometryObject(
            "ground_0", self.model.frames[0].parentJoint, ground_shape, ground_placement
        )
        ground_geom_obj.meshColor = np.ones((4))

        self.addCollisionObjects([ground_geom_obj])

    def _addConfiguredCollisionObjects(self, robot_config):
        """Add config-defined collision primitives to the Pinocchio collision model."""
        collision_model = robot_config.get("collision_model", {})
        for name, spec in collision_model.get("objects", {}).items():
            geom_obj = self._buildConfiguredCollisionObject(name, spec)
            self.addCollisionObjects([geom_obj])
            self.custom_collision_objects[name] = geom_obj

    def _buildConfiguredCollisionObject(self, name, spec):
        """Create a Pinocchio GeometryObject from a collision primitive spec."""
        geometry_type = spec["type"].lower()
        translation = np.asarray(spec.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        rotation = rotation_matrix_from_rpy(spec.get("rpy", [0.0, 0.0, 0.0]))
        placement = pin.SE3(rotation, translation)

        if geometry_type == "sphere":
            geometry = fcl.Sphere(float(spec["radius"]))
        elif geometry_type == "cylinder":
            geometry = fcl.Cylinder(float(spec["radius"]), float(spec["length"]))
        elif geometry_type == "halfspace":
            normal = np.asarray(spec.get("normal", [0.0, 0.0, 1.0]), dtype=float)
            geometry = fcl.Halfspace(normal, float(spec.get("offset", 0.0)))
        else:
            raise ValueError(
                f"Unsupported configured collision primitive type '{geometry_type}' for {name}"
            )

        parent_link = spec.get("parent_link")
        if parent_link is None:
            raise ValueError(f"Configured collision object '{name}' is missing parent_link")

        frame_id = self.model.getFrameId(parent_link)
        if frame_id >= len(self.model.frames):
            raise ValueError(
                f"Configured collision object '{name}' references unknown parent_link '{parent_link}'"
            )

        parent_joint = self.model.frames[frame_id].parentJoint
        geom_obj = pin.GeometryObject(name + "_0", parent_joint, geometry, placement)
        geom_obj.meshColor = np.ones((4))
        return geom_obj

    def computeDistances(self, q):
        """Compute distances for all collision pairs.

        Args:
            q (ndarray): Joint configuration vector.

        Returns:
            tuple: (distances, names) where distances is array of minimum distances and names is list of collision pair names.
        """
        data = self.model.createData()
        geom_data = pin.GeometryData(self.collision_model)
        pin.computeDistances(self.model, data, self.collision_model, geom_data, q)
        ds = np.array([result.min_distance for result in geom_data.distanceResults])
        ps = [[cp.first, cp.second] for cp in self.collision_model.collisionPairs]
        names = [
            [
                self.collision_model.geometryObjects[p[0]].name,
                self.collision_model.geometryObjects[p[1]].name,
            ]
            for p in ps
        ]
        return ds, names


class CasadiModelInterface:
    """Interface combining robot, scene, and Pinocchio models for CasADi-based control."""

    def __init__(self, config):
        """Initialize CasADi model interface.

        Args:
            config (dict): Configuration dictionary.
        """
        self.robot = MobileManipulator3D(config)
        self.scene = Scene(config)
        self.pinocchio_interface = PinocchioInterface(config)

        collisions_enabled = (
            config.get("self_collision_avoidance_enabled", False)
            or config.get("static_obstacles_collision_avoidance_enabled", False)
            or config.get("self_collision_emergency_stop", False)
        )

        self.collision_pairs = {
            "self": [],
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }
        self.collision_pairs_detailed = {
            "self": [],
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }

        self.signedDistanceSymMdls = {}  # keyed by collision pair (tuple)
        self.signedDistanceSymMdlsPerGroup = {
            "static_obstacles": {},
            "dynamic_obstacles": {},
        }
        if collisions_enabled:
            self._setupCollisionPair(config)
            self._setupCollisionPairDetailed()
            # nested dictionary, keyed by group name
            # obstacle groups are also a dictionary, keyed by obstacle name
            self._setupSelfCollisionSymMdl()
            self._setupStaticObstaclesCollisionSymMdl()
            self._setupPinocchioCollisionMdl()

    def _addCollisionPairFromTwoGroups(self, group1, group2):
        """Add all possible collision link pairs, one from each group.

        Args:
            group1 (list): List of collision link names.
            group2 (list): Another list of collision link names.

        Returns:
            list: Nested list of possible pairs.
        """

        pairs = []
        for link_name1 in group1:
            for link_name2 in group2:
                pairs.append([link_name1, link_name2])

        return pairs

    def _expandCollisionTargets(self, target):
        """Expand a collision target which may be a group name or an object name."""
        if isinstance(target, str):
            return self.robot.collision_link_names.get(target, [target])

        expanded_targets = []
        for item in target:
            expanded_targets.extend(self._expandCollisionTargets(item))
        return expanded_targets

    def _expandCollisionPairSpecs(self, pair_specs):
        """Expand pair specs expressed in group or object names into explicit pairs."""
        pairs = []
        for spec in pair_specs:
            if len(spec) != 2:
                raise ValueError(f"Collision pair spec must contain exactly two items: {spec}")
            left_targets = self._expandCollisionTargets(spec[0])
            right_targets = self._expandCollisionTargets(spec[1])
            pairs.extend(self._addCollisionPairFromTwoGroups(left_targets, right_targets))
        return pairs

    def _expandObstaclePairTargets(self, obstacle, targets):
        """Expand obstacle collision targets into explicit pairs."""
        expanded_pairs = []
        for target in targets:
            expanded_pairs.extend(
                self._addCollisionPairFromTwoGroups(
                    [obstacle], self._expandCollisionTargets(target)
                )
            )
        return expanded_pairs

    def _buildDefaultSelfCollisionPairs(self, detailed=False):
        """Build default self-collision pairs from generic group-level rules."""
        group_specs = (
            DEFAULT_DETAILED_SELF_COLLISION_GROUP_SPECS
            if detailed
            else DEFAULT_SELF_COLLISION_GROUP_SPECS
        )
        pairs = []
        for group_name, targets, limit in group_specs:
            link_group = list(self.robot.collision_link_names.get(group_name, []))
            if limit is not None:
                link_group = link_group[:limit]
            if not link_group:
                continue
            target_group = self._expandCollisionTargets(targets)
            if not target_group:
                continue
            pairs.extend(self._addCollisionPairFromTwoGroups(link_group, target_group))
        return pairs

    def _buildDefaultStaticObstaclePairs(self, obstacle, detailed=False):
        """Build default obstacle collision pairs from generic group-level rules."""
        target_specs = (
            DEFAULT_DETAILED_GROUND_TARGETS
            if detailed and obstacle == "ground"
            else DEFAULT_GROUND_TARGETS
            if obstacle == "ground"
            else DEFAULT_STATIC_OBSTACLE_TARGETS
        )
        return self._expandObstaclePairTargets(obstacle, target_specs)

    def _setupCollisionPair(self, config):
        """Setup collision pairs from configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        robot_config = config["robot"]
        collision_model = robot_config.get("collision_model", {})
        legacy_collision_pairs = robot_config.get("collision_pairs", {})
        if legacy_collision_pairs.get("self", False):
            self.collision_pairs["self"] = legacy_collision_pairs["self"]
        elif collision_model.get("self_collision_pairs", False):
            self.collision_pairs["self"] = self._expandCollisionPairSpecs(
                collision_model["self_collision_pairs"]
            )
        else:
            self.collision_pairs["self"] = self._buildDefaultSelfCollisionPairs()

        for obstacle in self.scene.collision_link_names.get("static_obstacles", []):
            if legacy_collision_pairs.get("static_obstacles", {}).get(obstacle, False):
                self.collision_pairs["static_obstacles"][obstacle] = (
                    self._expandObstaclePairTargets(
                        obstacle,
                        legacy_collision_pairs["static_obstacles"][obstacle],
                    )
                )
            elif collision_model.get("static_obstacle_pairs", {}).get(obstacle, False):
                self.collision_pairs["static_obstacles"][obstacle] = (
                    self._expandObstaclePairTargets(
                        obstacle,
                        collision_model["static_obstacle_pairs"][obstacle],
                    )
                )
            else:
                self.collision_pairs["static_obstacles"][obstacle] = (
                    self._buildDefaultStaticObstaclePairs(obstacle)
                )

    def _setupCollisionPairDetailed(self):
        """Setup detailed collision pairs for self-collision and obstacles."""
        collision_model = self.robot.config.get("collision_model", {})
        legacy_collision_pairs = self.robot.config.get("collision_pairs", {})
        if collision_model.get("pinocchio_self_collision_pairs", False):
            self.collision_pairs_detailed["self"] = self._expandCollisionPairSpecs(
                collision_model["pinocchio_self_collision_pairs"]
            )
        elif legacy_collision_pairs.get("self", False):
            self.collision_pairs_detailed["self"] = legacy_collision_pairs["self"]
        else:
            self.collision_pairs_detailed["self"] = self._buildDefaultSelfCollisionPairs(
                detailed=True
            )

        for obstacle in self.scene.collision_link_names.get("static_obstacles", []):
            if collision_model.get("pinocchio_static_obstacle_pairs", {}).get(
                obstacle, False
            ):
                self.collision_pairs_detailed["static_obstacles"][obstacle] = (
                    self._expandObstaclePairTargets(
                        obstacle,
                        collision_model["pinocchio_static_obstacle_pairs"][obstacle],
                    )
                )
            elif legacy_collision_pairs.get("static_obstacles", {}).get(obstacle, False):
                self.collision_pairs_detailed["static_obstacles"][obstacle] = (
                    self._expandObstaclePairTargets(
                        obstacle,
                        legacy_collision_pairs["static_obstacles"][obstacle],
                    )
                )
            else:
                self.collision_pairs_detailed["static_obstacles"][obstacle] = (
                    self._buildDefaultStaticObstaclePairs(obstacle, detailed=True)
                )

    def _setupSelfCollisionSymMdl(self):
        """Setup symbolic models for self-collision signed distances."""
        sd_syms = []
        for pair in self.collision_pairs["self"]:
            os = self.pinocchio_interface.getGeometryObject(pair)
            if None in os:
                raise ValueError(
                    f"Collision pair {pair} references a collision object that does not exist"
                )

            sd_sym = self.pinocchio_interface.getSignedDistance(
                os[0],
                self.robot.collisionLinkKinSymMdls[pair[0]](self.robot.q_sym),
                os[1],
                self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym),
            )
            sd_syms.append(sd_sym)
            sd_fcn = cs.Function(
                "sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym]
            )
            self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

        if not sd_syms:
            raise ValueError("No valid self-collision pairs were configured")
        self.signedDistanceSymMdlsPerGroup["self"] = cs.Function(
            "sd_self", [self.robot.q_sym], [cs.vertcat(*sd_syms)]
        )

    def _setupStaticObstaclesCollisionSymMdl(self):
        """Setup symbolic models for static obstacle signed distances."""
        for obstacle, pairs in self.collision_pairs["static_obstacles"].items():
            sd_syms = []
            for pair in pairs:
                os = self.pinocchio_interface.getGeometryObject(pair)
                if None in os:
                    raise ValueError(
                        f"Collision pair {pair} references a collision object that does not exist"
                    )

                sd_sym = self.pinocchio_interface.getSignedDistance(
                    os[0],
                    self.scene.collisionLinkKinSymMdls[pair[0]]([]),
                    os[1],
                    self.robot.collisionLinkKinSymMdls[pair[1]](self.robot.q_sym),
                )
                sd_syms.append(sd_sym)
                sd_fcn = cs.Function(
                    "sd_" + pair[0] + "_" + pair[1], [self.robot.q_sym], [sd_sym]
                )
                self.signedDistanceSymMdls[tuple(pair)] = sd_fcn

            if not sd_syms:
                raise ValueError(
                    f"No valid collision pairs were configured for static obstacle '{obstacle}'"
                )
            self.signedDistanceSymMdlsPerGroup["static_obstacles"][obstacle] = (
                cs.Function(
                    "sd_" + obstacle, [self.robot.q_sym], [cs.vertcat(*sd_syms)]
                )
            )

    def _setupPinocchioCollisionMdl(self):
        """Setup Pinocchio collision model with collision pairs."""
        self.pinocchio_interface.removeAllCollisionPairs()
        self.pinocchio_interface.addCollisionPairs(
            self.collision_pairs_detailed["self"]
        )
        for obstacle, pairs in self.collision_pairs_detailed[
            "static_obstacles"
        ].items():
            self.pinocchio_interface.addCollisionPairs(pairs)

    def getSignedDistanceSymMdls(self, name):
        """Get signed distance function by collision link name.

        Args:
            name (str): Collision link name.

        Returns:
            casadi.Function or None: Signed distance function, or None if not found.
        """
        if name == "self":
            return self.signedDistanceSymMdlsPerGroup["self"]
        else:
            for group, name_list in self.scene.collision_link_names.items():
                if name in name_list:
                    return self.signedDistanceSymMdlsPerGroup[group][name]
        print(name + " signed distance function does not exist")
        return None

    def evaluateSignedDistance(
        self, names: List[str], qs: ndarray, params: Dict[str, List[ndarray]] = {}
    ):
        """Evaluate signed distances for collision link names.

        Args:
            names (List[str]): List of collision link names to evaluate.
            qs (ndarray): Joint configuration(s), shape (N, nq) or (nq,).
            params (Dict[str, List[ndarray]]): Additional parameters per collision link.

        Returns:
            dict: Dictionary mapping collision link names to signed distance arrays.
        """
        sd = {}
        N = len(qs)
        names.remove("static_obstacles")
        static_obstacle_names = [
            n for n in self.collision_pairs["static_obstacles"].keys()
        ]
        names += static_obstacle_names
        for name in names:
            sd_fcn = self.getSignedDistanceSymMdls(name)
            sdn_fcn = sd_fcn.map(N, "thread", 2)
            # sds dimension: num collision pairs x num time step
            if name in static_obstacle_names:
                args = [qs.T] + [p.T for p in params["static_obstacles"]]
            else:
                args = [qs.T] + [p.T for p in params[name]]
            sds = sdn_fcn(*args).toarray()
            sd_mins = np.min(sds, axis=0)
            sd[name] = sd_mins

        return sd

    def evaluateSignedDistancePerPair(
        self, names: List[str], qs: ndarray, params: Dict[str, List[ndarray]] = {}
    ):
        """Evaluate signed distances per collision pair (not aggregated).

        Args:
            names (List[str]): List of collision link names to evaluate.
            qs (ndarray): Joint configuration(s), shape (N, nq) or (nq,).
            params (Dict[str, List[ndarray]]): Additional parameters per collision link.

        Returns:
            dict: Dictionary mapping collision link names to dictionaries of per-pair signed distances.
        """
        sd = {}

        N = len(qs)
        for name in names:
            if name != "static_obstacles":
                sd[name] = {}
                for pair in self.collision_pairs[name]:
                    sd_fcn = self.signedDistanceSymMdls[tuple(pair)]
                    sdn_fcn = sd_fcn.map(N, "thread", 2)
                    # sds dimension: num collision pairs x num time step
                    args = [qs.T] + [p.T for p in params[name]]
                    sds = sdn_fcn(*args).toarray()
                    # sd_mins = np.min(sds, axis=0)
                    sd[name]["&".join(pair)] = sds.flatten()
            else:
                for obstacle, pairs in self.collision_pairs["static_obstacles"].items():
                    sd[obstacle] = {}
                    for pair in pairs:
                        sd_fcn = self.signedDistanceSymMdls[tuple(pair)]
                        sdn_fcn = sd_fcn.map(N, "thread", 2)
                        args = [qs.T] + [p.T for p in params["static_obstacles"]]

                        # sds dimension: num collision pairs x num time step
                        sds = sdn_fcn(*args).toarray()
                        # sd_mins = np.min(sds, axis=0)
                        sd[obstacle]["&".join(pair)] = sds.flatten()

        return sd


class Scene:
    def __init__(self, config):
        """Casadi symbolic model of a 3D Scene.

        Args:
            config (dict): Configuration dictionary.
        """
        if config["scene"]["enabled"]:
            urdf_path = parsing.parse_and_compile_urdf(config["scene"]["urdf"])
            self.model = pin.buildModelsFromUrdf(urdf_path)[0]
            self.cmodel = cpin.Model(self.model)
            self.cdata = self.cmodel.createData()
            self.q = cs.SX.sym("q", self.model.nq)
        else:
            self.model = None

        self.collision_link_names = config["scene"].get(
            "collision_link_names", {"static_obstacles": ["ground"]}
        )
        self._setupCollisionLinkKinSymMdl()

    def _setupCollisionLinkKinSymMdl(self):
        """Create kinematic symbolic model for collision links in scene."""
        self.collisionLinkKinSymMdls = {}

        for group, name_list in self.collision_link_names.items():
            for name in name_list:
                if name == "ground":
                    self.collisionLinkKinSymMdls[name] = cs.Function(
                        "fk_ground",
                        [cs.SX.sym("empty", 0)],
                        [cs.DM.zeros(3), cs.DM.eye(3)],
                    )
                else:
                    omf_i = self.cdata.oMf[self.cmodel.getFrameId(name)]
                    link_pos = omf_i.translation
                    # link_rot = omf_i.rotation
                    self.collisionLinkKinSymMdls[name] = cs.Function(
                        name + "_fcn", [self.q], [link_pos], ["q"], ["pos"]
                    ).expand()


class MobileManipulator3D:
    def __init__(self, config):
        """Casadi symbolic model of Mobile Manipulator.

        Args:
            config (dict): Configuration dictionary with robot parameters.
        """
        urdf_path = parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        package_dirs = [str(Path(urdf_path).parent)]
        self.config = config["robot"]
        # create Pinocchio model to get robot info such as number of joints, link names, and for collision checking
        self.model = pin.buildModelsFromUrdf(urdf_path, package_dirs=package_dirs)[0]
        # create Casadi model for symbolic kinematics and dynamics functions
        self.cmodel = cpin.Model(self.model)
        # create Casadi data for kinematics and dynamics computations
        self.cdata = self.cmodel.createData()

        self.numjoint = self.model.nq
        # DoF includes both joint DoF and 3 DoF for the mobile base position (x, y, theta)
        self.DoF = self.numjoint + 3
        self.dt = self.config["time_discretization_dt"]
        self.ub_x = parsing.parse_array(self.config["limits"]["state"]["upper"])
        self.lb_x = parsing.parse_array(self.config["limits"]["state"]["lower"])
        self.ub_u = parsing.parse_array(self.config["limits"]["input"]["upper"])
        self.lb_u = parsing.parse_array(self.config["limits"]["input"]["lower"])
        self._sync_joint_position_limits_from_urdf()
        self.base_type = self.config.get("base_type", "omnidirectional").lower()
        self.nonholonomic_mode = (
            self.config.get("nonholonomic_mode", "constraint").lower()
        )
        self.nonholonomic_lateral_damping = float(
            self.config.get("nonholonomic_lateral_damping", 20.0)
        )

        self.link_names = self.config["link_names"]
        self.tool_link_name = self.config["tool_link_name"]
        self.base_link_name = self.config["base_link_name"]
        self.collision_link_names = get_robot_collision_groups(self.config)
        self.collision_object_specs = self.config.get("collision_model", {}).get(
            "objects", {}
        )

        self.qb_sym = cs.SX.sym("qb", 3)
        self.qa_sym = cs.SX.sym("qa", self.numjoint)
        self.q_sym = cs.vertcat(self.qb_sym, self.qa_sym)

        # 用关节变量 qa 计算机器人所有 link/frame 相对于 base_link 的空间位姿，
        # 并存到 cdata 中，供后续取用（比如末端位置、碰撞检测等）
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.qa_sym)

        # create self.kinSymMdls dict:{robot links name: cs function of its forward kinematics function}
        self._setupRobotKinSymMdl()
        # create self.collisionLinkKinSymMdls dict:{collision links name: cs function of its forward kinematics function}
        self._setupCollisionLinkKinSymMdl()
        # create self.ssSymMdl robot's state space symbolic model
        self._setupSSSymMdlDI()
        # create self.jacSymMdls dict:{robot links name: cs functions of its jacobian}
        self._setupJacobianSymMdl()
        # create self.manipulability_fcn
        self._setupManipulabilitySymMdl()

    def _sync_joint_position_limits_from_urdf(self):
        """Overwrite joint position bounds with values read from the URDF.

        Only the arm joint position slice is synchronized here. Base position,
        state velocity bounds, and input bounds remain YAML-configurable since
        they encode controller design choices rather than raw URDF limits.
        """
        joint_lb = np.asarray(self.model.lowerPositionLimit).reshape(-1)
        joint_ub = np.asarray(self.model.upperPositionLimit).reshape(-1)
        self.lb_x[3 : 3 + self.numjoint] = joint_lb
        self.ub_x[3 : 3 + self.numjoint] = joint_ub

    def _setupSSSymMdlDI(self):
        """Create State-space symbolic model for MM"""
        self.va_sym = cs.SX.sym("va", self.numjoint)
        self.vb_sym = cs.SX.sym("vb", 3)
        self.v_sym = cs.vertcat(self.vb_sym, self.va_sym)

        self.x_sym = cs.vertcat(self.q_sym, self.v_sym)
        self.u_sym = cs.SX.sym("u", self.v_sym.size()[0])

        nx = self.x_sym.size()[0]
        nu = self.u_sym.size()[0]
        use_nonholonomic_dynamics = (
            self.base_type == "nonholonomic" and self.nonholonomic_mode == "dynamics"
        )
        self.ssSymMdl = {
            "x": self.x_sym,
            "u": self.u_sym,
            "mdl_type": (
                ["nonlinear", "time_invariant"]
                if use_nonholonomic_dynamics
                else ["linear", "time_invariant"]
            ),
            "nx": nx,
            "nu": nu,
            "ub_x": list(self.ub_x),
            "lb_x": list(self.lb_x),
            "ub_u": list(self.ub_u),
            "lb_u": list(self.lb_u),
        }

        if use_nonholonomic_dynamics:
            theta = self.qb_sym[2]
            c = cs.cos(theta)
            s = cs.sin(theta)
            vx, vy = self.vb_sym[0], self.vb_sym[1]
            ax, ay = self.u_sym[0], self.u_sym[1]

            v_lat = -s * vx + c * vy
            a_fwd = c * ax + s * ay
            a_lat = -s * ax + c * ay - self.nonholonomic_lateral_damping * v_lat
            a_base = cs.vertcat(c * a_fwd - s * a_lat, s * a_fwd + c * a_lat)

            qdot = self.v_sym
            vdot = cs.vertcat(a_base, self.u_sym[2:])
            xdot = cs.vertcat(qdot, vdot)
        else:
            A = cs.DM.zeros((nx, nx))
            G = cs.DM.eye(self.DoF)
            A[: self.DoF, self.DoF :] = G
            B = cs.DM.zeros((nx, nu))
            B[self.DoF :, :] = cs.DM.eye(nu)
            xdot = A @ self.x_sym + B @ self.u_sym
            self.ssSymMdl["A"] = A
            self.ssSymMdl["B"] = B

        fmdl = cs.Function(
            "ss_fcn", [self.x_sym, self.u_sym], [xdot], ["x", "u"], ["xdot"]
        )
        self.ssSymMdl["fmdl"] = fmdl.expand()
        self.ssSymMdl["fmdlk"] = self._discretizefmdl(self.ssSymMdl, self.dt)

    def _setupRobotKinSymMdl(self):
        """Create kinematic symbolic model for MM links keyed by link name"""
        self.kinSymMdls = {}
        for name in self.link_names:
            self.kinSymMdls[name] = self._getFk(name)

    def _setupCollisionLinkKinSymMdl(self):
        """Create kinematic symbolic model for collision links."""
        self.collisionLinkKinSymMdls = {}

        for collision_group, link_list in self.collision_link_names.items():
            for name in link_list:
                if name in self.collision_object_specs:
                    self.collisionLinkKinSymMdls[name] = self._getCollisionObjectFk(
                        name, self.collision_object_specs[name]
                    )
                else:
                    self.collisionLinkKinSymMdls[name] = self._getFk(name)

    def _getCollisionObjectFk(self, object_name, spec):
        """Create symbolic FK for a configured collision primitive."""
        parent_link = spec.get("parent_link")
        if parent_link is None:
            raise ValueError(
                f"Configured collision object '{object_name}' is missing parent_link"
            )

        if parent_link in self.kinSymMdls:
            parent_fk = self.kinSymMdls[parent_link]
        else:
            parent_fk = self._getFk(parent_link)

        parent_pos, parent_rot = parent_fk(self.q_sym)
        translation = cs.DM(
            np.asarray(spec.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        )
        rotation = cs.DM(
            rotation_matrix_from_rpy(spec.get("rpy", [0.0, 0.0, 0.0]))
        )
        object_pos = parent_pos + parent_rot @ translation
        object_rot = parent_rot @ rotation
        return cs.Function(
            object_name + "_fcn",
            [self.q_sym],
            [object_pos, object_rot],
            ["q"],
            ["pos", "rot"],
        )

    def _setupJacobianSymMdl(self):
        """Create jacobian symbolic model for MM links keyed by link name
        Jacobian in the reference frame of the world frame
        For tool link, creates both 3D position Jacobian and 6D spatial Jacobian
        """
        self.jacSymMdls = {}
        for name in self.link_names:
            fk_fcn = self.kinSymMdls[name]
            fk_pos_eqn, fk_rot_eqn = fk_fcn(self.q_sym)
            # Position Jacobian (3D)
            Jk_pos_eqn = cs.jacobian(fk_pos_eqn, self.q_sym)
            self.jacSymMdls[name] = cs.Function(
                name + "_jac_fcn", [self.q_sym], [Jk_pos_eqn], ["q"], ["J(q)"]
            )

            # For tool link, also create 6D spatial Jacobian (linear + angular)
            if name == self.tool_link_name:
                # Compute angular velocity Jacobian from rotation matrix
                # Angular velocity: ω = 0.5 * [R^T * dR/dq_i]_vee for each q_i
                # The vee map extracts the vector from a skew-symmetric matrix
                # For a rotation matrix R, dR/dq gives us the derivative
                # We compute: ω = [R^T * dR/dq]_vee
                # Simplified: compute the skew-symmetric part of R^T * dR/dq
                dR_dq = cs.jacobian(fk_rot_eqn, self.q_sym)
                # dR_dq is 9 x DoF (each column is flattened 3x3 matrix)
                # Reshape each column to 3x3, compute R^T * dR/dq_i, then extract skew part
                Jw_list = []
                for i in range(self.DoF):
                    dR_i = cs.reshape(dR_dq[:, i], 3, 3)
                    # Compute R^T * dR/dq_i
                    R_T_dR = cs.mtimes(fk_rot_eqn.T, dR_i)
                    # Extract skew-symmetric part: ω = [R^T * dR]_vee
                    # For skew-symmetric matrix S, [S]_vee = [S[2,1], S[0,2], S[1,0]]
                    omega_i = (
                        cs.vertcat(
                            R_T_dR[2, 1] - R_T_dR[1, 2],
                            R_T_dR[0, 2] - R_T_dR[2, 0],
                            R_T_dR[1, 0] - R_T_dR[0, 1],
                        )
                        * 0.5
                    )
                    Jw_list.append(omega_i)
                Jw_eqn = cs.horzcat(*Jw_list)
                # Stack linear and angular Jacobians
                J_spatial = cs.vertcat(Jk_pos_eqn, Jw_eqn)
                self.jacSymMdls[name + "_spatial"] = cs.Function(
                    name + "_jac_spatial_fcn",
                    [self.q_sym],
                    [J_spatial],
                    ["q"],
                    ["J_spatial(q)"],
                )

    def _setupManipulabilitySymMdl(self):
        """Setup symbolic models for end-effector and arm manipulability."""
        Jee_fcn = self.jacSymMdls[self.tool_link_name]
        qsym = cs.SX.sym("qsx", self.DoF)
        Jee_eqn = Jee_fcn(qsym)
        man_eqn = cs.det(Jee_eqn @ Jee_eqn.T) ** 0.5

        self.manipulability_fcn = cs.Function("manipulability_fcn", [qsym], [man_eqn])
        arm_man_eqn = cs.det(Jee_eqn[:, 3:] @ Jee_eqn[:, 3:].T) ** 0.5
        self.arm_manipulability_fcn = cs.Function(
            "arm_manipulability_fcn", [qsym], [arm_man_eqn]
        )

    def _getFk(self, link_name, base_frame=False):
        """Create symbolic function for a link named link_name.

        The symbolic function returns the position of its parent joint in and rotation w.r.t the world frame.
        Note this is different from link_state provided by Pybullet which provides CoM position.

        Args:
            link_name (str): Name of the link.
            base_frame (bool): If True, express pose in base frame; if False, in world frame.

        Returns:
            casadi.Function: Forward kinematics function returning (position, rotation).
        """
        if link_name == self.base_link_name:
            base_pos = cs.vertcat(self.qb_sym[0], self.qb_sym[1], 0)
            base_rot = cs.SX.eye(3)
            if not base_frame:
                ctheta = cs.cos(self.qb_sym[2])
                stheta = cs.sin(self.qb_sym[2])
                base_rot[0, 0] = ctheta
                base_rot[0, 1] = -stheta
                base_rot[1, 0] = stheta
                base_rot[1, 1] = ctheta
            else:
                base_pos = cs.SX.zeros(3)
            return cs.Function(
                link_name + "_fcn",
                [self.q_sym],
                [base_pos, base_rot],
                ["q"],
                ["pos", "rot"],
            )

        Hwb = cs.SX.eye(4)  # T related to movement of base
        if not base_frame:
            Hwb[0, 0] = np.cos(self.qb_sym[2])
            Hwb[1, 0] = np.sin(self.qb_sym[2])
            Hwb[0, 1] = -np.sin(self.qb_sym[2])
            Hwb[1, 1] = np.cos(self.qb_sym[2])
            Hwb[:2, 3] = self.qb_sym[:2]

        omf_i = self.cdata.oMf[self.cmodel.getFrameId(link_name)]
        link_pos = omf_i.translation
        link_rot = omf_i.rotation
        Hbl = cs.SX.eye(4)  # T from base to link
        Hbl[:3, :3] = link_rot
        Hbl[:3, 3] = link_pos
        Hwl = Hwb @ Hbl  # overall transformation

        return cs.Function(
            link_name + "_fcn",
            [self.q_sym],
            [Hwl[:3, 3], Hwl[:3, :3]],
            ["q"],
            ["pos", "rot"],
        )

    def _discretizefmdl(self, ss_mdl, dt):
        """Discretize state-space model for given time step.

        Args:
            ss_mdl (dict): State-space model dictionary.
            dt (float): Discretization time step.

        Returns:
            dict: Discretized state-space model dictionary.
        """
        x_sym = ss_mdl["x"]
        u_sym = ss_mdl["u"]
        if "linear" in ss_mdl["mdl_type"]:
            A = ss_mdl["A"]
            B = ss_mdl["B"]
            nx = x_sym.size()[0]
            nu = u_sym.size()[0]
            # TODO: discretization time is now hardcoded to 0.1 second. better if we could make dt symbolic too
            M = np.zeros((nx + nu, nx + nu))
            M[:nx, :nx] = A
            M[:nx, nx:] = B
            Md = expm(M * dt)
            Ad = Md[:nx, :nx]
            Bd = Md[:nx, nx:]

            xk1_eqn = Ad @ x_sym + Bd @ u_sym
        else:
            fmdl = ss_mdl["fmdl"]
            k1 = fmdl(x_sym, u_sym)
            k2 = fmdl(x_sym + 0.5 * dt * k1, u_sym)
            k3 = fmdl(x_sym + 0.5 * dt * k2, u_sym)
            k4 = fmdl(x_sym + dt * k3, u_sym)
            xk1_eqn = x_sym + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        fdsc_fcn = cs.Function("fmdlk", [x_sym, u_sym], [xk1_eqn], ["xk", "uk"], ["xk1"])
        return fdsc_fcn

    @staticmethod
    def ssIntegrate(dt, xo, u_bar, ssSymMdl):
        """Integrate state-space model.

        Args:
            dt (float): Discretization time step.
            xo (ndarray): Initial state.
            u_bar (ndarray): Control inputs, shape [N, nu].
            ssSymMdl (dict): State-space symbolic model.

        Returns:
            ndarray: State trajectory x_bar, shape [N+1, nx].
        """
        N = u_bar.shape[0]
        fk = ssSymMdl["fmdlk"]
        f_pred = fk.mapaccum(N)
        x_bar = f_pred(xo, u_bar.T)
        x_bar = np.hstack((np.expand_dims(xo, -1), x_bar)).T

        return x_bar

    def checkBounds(self, xs, us, tol=1e-2):
        """Check bounds for states and controls.

        Args:
            xs (ndarray): State trajectory.
            us (ndarray): Control trajectory.
            tol (float): Tolerance for bound checking.

        Returns:
            bool: True if within bounds, False otherwise.
        """

        # check state
        ub_x_check = xs < self.ub_x + tol
        lb_x_check = xs > self.lb_x - tol
        xs_num_violation = np.sum(1 - ub_x_check * lb_x_check, axis=1)
        print(lb_x_check.shape)

        # check input
        ub_u_check = us < self.ub_u + tol
        lb_u_check = us > self.lb_u - tol
        us_num_violation = np.sum(1 - ub_u_check * lb_u_check, axis=1)
        print(lb_u_check.shape)

        return xs_num_violation, us_num_violation

    def getEE(self, q, base_frame=False):
        """Get end-effector position and orientation.

        Args:
            q (ndarray): Joint configuration vector.
            base_frame (bool): If True, express position in base frame; if False, in world frame.

        Returns:
            tuple: (position, quaternion) where position is 3D array and quaternion is 4D array.
        """
        fee = self.kinSymMdls[self.tool_link_name]
        P, rot = fee(q)
        quat = r2q(np.array(rot), order="xyzs")
        if base_frame:
            P[:2] -= q[:2]
            Rwb = rotz(q[2])
            P = Rwb.T @ P

        return P.toarray().flatten(), quat
