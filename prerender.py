import argparse
import multiprocessing
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

roadgraph_features = {
    "roadgraph_samples/dir": tf.io.FixedLenFeature(
        [20000, 3], tf.float32, default_value=None
    ),
    "roadgraph_samples/id": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/type": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/valid": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/xyz": tf.io.FixedLenFeature(
        [20000, 3], tf.float32, default_value=None
    ),
}

# Features of other agents.
state_features = {
    "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    "state/tracks_to_predict": tf.io.FixedLenFeature(
        [128], tf.int64, default_value=None
    ),
    "state/current/bbox_yaw": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/height": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/length": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/timestamp_micros": tf.io.FixedLenFeature(
        [128, 1], tf.int64, default_value=None
    ),
    "state/current/valid": tf.io.FixedLenFeature(
        [128, 1], tf.int64, default_value=None
    ),
    "state/current/vel_yaw": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/velocity_x": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/velocity_y": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/speed": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/width": tf.io.FixedLenFeature(
        [128, 1], tf.float32, default_value=None
    ),
    "state/current/x": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/current/y": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/current/z": tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    "state/future/bbox_yaw": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/height": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/length": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/timestamp_micros": tf.io.FixedLenFeature(
        [128, 80], tf.int64, default_value=None
    ),
    "state/future/valid": tf.io.FixedLenFeature(
        [128, 80], tf.int64, default_value=None
    ),
    "state/future/vel_yaw": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/velocity_x": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/velocity_y": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/width": tf.io.FixedLenFeature(
        [128, 80], tf.float32, default_value=None
    ),
    "state/future/x": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/future/y": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/future/z": tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    "state/past/bbox_yaw": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/height": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/length": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/timestamp_micros": tf.io.FixedLenFeature(
        [128, 10], tf.int64, default_value=None
    ),
    "state/past/valid": tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    "state/past/vel_yaw": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/velocity_x": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/velocity_y": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/speed": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/width": tf.io.FixedLenFeature(
        [128, 10], tf.float32, default_value=None
    ),
    "state/past/x": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    "state/past/y": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    "state/past/z": tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    "scenario/id": tf.io.FixedLenFeature([1], tf.string, default_value=None),
}

traffic_light_features = {
    "traffic_light_state/current/state": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/current/valid": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/current/id": tf.io.FixedLenFeature(
        [1, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/current/x": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/current/y": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/current/z": tf.io.FixedLenFeature(
        [1, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/state": tf.io.FixedLenFeature(
        [10, 16], tf.int64, default_value=None
    ),
    "traffic_light_state/past/valid": tf.io.FixedLenFeature(
        [10, 16], tf.int64, default_value=None
    ),
    # "traffic_light_state/past/id":
    # tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    "traffic_light_state/past/x": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/y": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
    "traffic_light_state/past/z": tf.io.FixedLenFeature(
        [10, 16], tf.float32, default_value=None
    ),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)
MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]
idx2type = ["unset", "vehicle", "pedestrian", "cyclist", "other"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw data")
    parser.add_argument("--out", type=str, required=True, help="Path to save data")
    parser.add_argument(
        "--no-valid", action="store_true", help="Use data with flag `valid = 0`"
    )
    parser.add_argument(
        "--use-vectorize", action="store_true", help="Generate vector data"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=20, required=False, help="Number of threads"
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=8,
        required=False,
        help="Use `1/n_shards` of full dataset",
    )
    parser.add_argument(
        "--each",
        type=int,
        default=0,
        required=False,
        help="Take `each` sample in shard",
    )

    args = parser.parse_args()

    return args


def rasterize(
    tracks_to_predict,
    past_x,
    past_y,
    current_x,
    current_y,
    current_yaw,
    past_yaw,
    past_valid,
    current_valid,
    agent_type,
    roadlines_coords,
    roadlines_types,
    roadlines_valid,
    roadlines_ids,
    widths,
    lengths,
    agents_ids,
    tl_states,
    tl_ids,
    tl_valids,
    future_x,
    future_y,
    future_valid,
    scenario_id,
    validate,
    crop_size=512,
    raster_size=224,
    shift=2 ** 9,
    magic_const=3,
    n_channels=11,
):
    GRES = []
    displacement = np.array([[raster_size // 4, raster_size // 2]]) * shift
    tl_dict = {"green": set(), "yellow": set(), "red": set()}

    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    for tl_state, tl_id, tl_valid in zip(
        tl_states.flatten(), tl_ids.flatten(), tl_valids.flatten()
    ):
        if tl_valid == 0:
            continue
        if tl_state in [1, 4, 7]:
            tl_dict["red"].add(tl_id)
        if tl_state in [2, 5, 8]:
            tl_dict["yellow"].add(tl_id)
        if tl_state in [3, 6]:
            tl_dict["green"].add(tl_id)

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )

    GT_XY = np.concatenate(
        (np.expand_dims(future_x, axis=-1), np.expand_dims(future_y, axis=-1)), axis=-1
    )

    YAWS = np.concatenate((past_yaw, current_yaw), axis=1)

    agents_valid = np.concatenate((past_valid, current_valid), axis=1)

    roadlines_valid = roadlines_valid.reshape(-1)
    roadlines_coords = (
        roadlines_coords[:, :2][roadlines_valid > 0]
        * shift
        * magic_const
        * raster_size
        / crop_size
    )
    roadlines_types = roadlines_types[roadlines_valid > 0]
    roadlines_ids = roadlines_ids.reshape(-1)[roadlines_valid > 0]

    for _, (
        xy,
        current_val,
        val,
        _,
        yaw,
        agent_id,
        gt_xy,
        future_val,
        predict,
    ) in enumerate(
        zip(
            XY,
            current_valid,
            agents_valid,
            agent_type,
            current_yaw.flatten(),
            agents_ids,
            GT_XY,
            future_valid,
            tracks_to_predict.flatten(),
        )
    ):
        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue

        RES_ROADMAP = (
            np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
        )
        RES_EGO = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]
        RES_OTHER = [
            np.zeros((raster_size, raster_size, 1), dtype=np.uint8)
            for _ in range(n_channels)
        ]

        xy_val = xy[val > 0]
        if len(xy_val) == 0:
            continue

        unscaled_center_xy = xy_val[-1].reshape(1, -1)
        center_xy = unscaled_center_xy * shift * magic_const * raster_size / crop_size
        rot_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ]
        )

        centered_roadlines = (roadlines_coords - center_xy) @ rot_matrix + displacement
        centered_others = (
            XY.reshape(-1, 2) * shift * magic_const * raster_size / crop_size
            - center_xy
        ) @ rot_matrix + displacement
        centered_others = centered_others.reshape(128, n_channels, 2)
        centered_gt = (gt_xy - unscaled_center_xy) @ rot_matrix

        unique_road_ids = np.unique(roadlines_ids)
        for road_id in unique_road_ids:
            if road_id >= 0:
                roadline = centered_roadlines[roadlines_ids == road_id]
                road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

                road_color = road_colors[road_type]
                for c, rgb in zip(
                    ["green", "yellow", "red"],
                    [
                        (0, MAX_PIXEL_VALUE, 0),
                        (MAX_PIXEL_VALUE, 211, 0),
                        (MAX_PIXEL_VALUE, 0, 0),
                    ],
                ):
                    if road_id in tl_dict[c]:
                        road_color = rgb

                RES_ROADMAP = cv2.polylines(
                    RES_ROADMAP,
                    [roadline.astype(int)],
                    False,
                    road_color,
                    shift=9,
                )

        unique_agent_ids = np.unique(agents_ids)

        is_ego = False
        self_type = 0
        _tmp = 0
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            if other_agent_id < 1:
                continue
            if other_agent_id == agent_id:
                is_ego = True
                self_type = agent_type[agents_ids == other_agent_id]
            else:
                is_ego = False

            _tmp += 1
            agent_lane = centered_others[agents_ids == other_agent_id][0]
            agent_valid = agents_valid[agents_ids == other_agent_id]
            agent_yaw = YAWS[agents_ids == other_agent_id]

            agent_l = lengths[agents_ids == other_agent_id]
            agent_w = widths[agents_ids == other_agent_id]

            for timestamp, (coord, valid_coordinate, past_yaw,) in enumerate(
                zip(
                    agent_lane,
                    agent_valid.flatten(),
                    agent_yaw.flatten(),
                )
            ):
                if valid_coordinate == 0:
                    continue
                box_points = (
                    np.array(
                        [
                            -agent_l,
                            -agent_w,
                            agent_l,
                            -agent_w,
                            agent_l,
                            agent_w,
                            -agent_l,
                            agent_w,
                        ]
                    )
                    .reshape(4, 2)
                    .astype(np.float32)
                    * shift
                    * magic_const
                    / 2
                    * raster_size
                    / crop_size
                )

                box_points = (
                    box_points
                    @ np.array(
                        (
                            (np.cos(yaw - past_yaw), -np.sin(yaw - past_yaw)),
                            (np.sin(yaw - past_yaw), np.cos(yaw - past_yaw)),
                        )
                    ).reshape(2, 2)
                )

                _coord = np.array([coord])

                box_points = box_points + _coord
                box_points = box_points.reshape(1, -1, 2).astype(np.int32)

                if is_ego:
                    cv2.fillPoly(
                        RES_EGO[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )
                else:
                    cv2.fillPoly(
                        RES_OTHER[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE,
                        shift=9,
                    )

        raster = np.concatenate([RES_ROADMAP] + RES_EGO + RES_OTHER, axis=2)

        raster_dict = {
            "object_id": agent_id,
            "raster": raster,
            "yaw": yaw,
            "shift": unscaled_center_xy,
            "_gt_marginal": gt_xy,
            "gt_marginal": centered_gt,
            "future_val_marginal": future_val,
            "gt_joint": GT_XY[tracks_to_predict.flatten() > 0],
            "future_val_joint": future_valid[tracks_to_predict.flatten() > 0],
            "scenario_id": scenario_id,
            "self_type": self_type,
        }

        GRES.append(raster_dict)

    return GRES


F2I = {
    "x": 0,
    "y": 1,
    "s": 2,
    "vel_yaw": 3,
    "bbox_yaw": 4,
    "l": 5,
    "w": 6,
    "agent_type_range": [7, 12],
    "lane_range": [13, 33],
    "lt_range": [34, 43],
    "global_idx": 44,
}


def ohe(N, n, zero):
    n = int(n)
    N = int(N)
    M = np.eye(N)
    diff = 0
    if zero:
        M = np.concatenate((np.zeros((1, N)), M), axis=0)
        diff = 1
    return M[n + diff]


def make_2d(arraylist):
    n = len(arraylist)
    k = arraylist[0].shape[0]
    a2d = np.zeros((n, k))
    for i in range(n):
        a2d[i] = arraylist[i]
    return a2d


def vectorize(
    past_x,
    current_x,
    past_y,
    current_y,
    past_valid,
    current_valid,
    past_speed,
    current_speed,
    past_velocity_yaw,
    current_velocity_yaw,
    past_bbox_yaw,
    current_bbox_yaw,
    Agent_id,
    Agent_type,
    Roadline_id,
    Roadline_type,
    Roadline_valid,
    Roadline_xy,
    Tl_rl_id,
    Tl_state,
    Tl_valid,
    W,
    L,
    tracks_to_predict,
    future_valid,
    validate,
    n_channels=11,
):

    XY = np.concatenate(
        (
            np.expand_dims(np.concatenate((past_x, current_x), axis=1), axis=-1),
            np.expand_dims(np.concatenate((past_y, current_y), axis=1), axis=-1),
        ),
        axis=-1,
    )

    Roadline_valid = Roadline_valid.flatten()
    RoadXY = Roadline_xy[:, :2][Roadline_valid > 0]
    Roadline_type = Roadline_type[Roadline_valid > 0].flatten()
    Roadline_id = Roadline_id[Roadline_valid > 0].flatten()

    tl_state = [[-1] for _ in range(9)]

    for lane_id, state, valid in zip(
        Tl_rl_id.flatten(), Tl_state.flatten(), Tl_valid.flatten()
    ):
        if valid == 0:
            continue
        tl_state[int(state)].append(lane_id)

    VALID = np.concatenate((past_valid, current_valid), axis=1)

    Speed = np.concatenate((past_speed, current_speed), axis=1)
    Vyaw = np.concatenate((past_velocity_yaw, current_velocity_yaw), axis=1)
    Bbox_yaw = np.concatenate((past_bbox_yaw, current_bbox_yaw), axis=1)

    GRES = []

    ROADLINES_STATE = []

    GLOBAL_IDX = -1

    unique_road_ids = np.unique(Roadline_id)
    for road_id in unique_road_ids:

        GLOBAL_IDX += 1

        roadline_coords = RoadXY[Roadline_id == road_id]
        roadline_type = Roadline_type[Roadline_id == road_id][0]

        for i, (x, y) in enumerate(roadline_coords):
            if i > 0 and i < len(roadline_coords) - 1 and i % 3 > 0:
                continue
            tmp = np.zeros(48)
            tmp[0] = x
            tmp[1] = y

            tmp[13:33] = ohe(20, roadline_type, True)

            tmp[44] = GLOBAL_IDX

            ROADLINES_STATE.append(tmp)

    ROADLINES_STATE = make_2d(ROADLINES_STATE)

    for (
        agent_id,
        xy,
        current_val,
        valid,
        _,
        bbox_yaw,
        _,
        _,
        _,
        future_val,
        predict,
    ) in zip(
        Agent_id,
        XY,
        current_valid,
        VALID,
        Speed,
        Bbox_yaw,
        Vyaw,
        W,
        L,
        future_valid,
        tracks_to_predict.flatten(),
    ):

        if (not validate and future_val.sum() == 0) or (validate and predict == 0):
            continue
        if current_val == 0:
            continue

        GLOBAL_IDX = -1
        RES = []

        xy_val = xy[valid > 0]
        if len(xy_val) == 0:
            continue

        centered_xy = xy_val[-1].copy().reshape(-1, 2)

        ANGLE = bbox_yaw[-1]

        rot_matrix = np.array(
            [
                [np.cos(ANGLE), -np.sin(ANGLE)],
                [np.sin(ANGLE), np.cos(ANGLE)],
            ]
        ).reshape(2, 2)

        local_roadlines_state = ROADLINES_STATE.copy()

        local_roadlines_state[:, :2] = (
            local_roadlines_state[:, :2] - centered_xy
        ) @ rot_matrix.astype(np.float64)

        local_XY = ((XY - centered_xy).reshape(-1, 2) @ rot_matrix).reshape(
            128, n_channels, 2
        )

        for (
            other_agent_id,
            other_agent_type,
            other_xy,
            other_valids,
            other_speeds,
            other_bbox_yaws,
            other_v_yaws,
            other_w,
            other_l,
            other_predict,
        ) in zip(
            Agent_id,
            Agent_type,
            local_XY,
            VALID,
            Speed,
            Bbox_yaw,
            Vyaw,
            W.flatten(),
            L.flatten(),
            tracks_to_predict.flatten(),
        ):
            if other_valids.sum() == 0:
                continue

            GLOBAL_IDX += 1
            for timestamp, (
                (x, y),
                v,
                other_speed,
                other_v_yaw,
                other_bbox_yaw,
            ) in enumerate(
                zip(other_xy, other_valids, other_speeds, other_v_yaws, other_bbox_yaws)
            ):
                if v == 0:
                    continue
                tmp = np.zeros(48)
                tmp[0] = x
                tmp[1] = y
                tmp[2] = other_speed
                tmp[3] = other_v_yaw - ANGLE
                tmp[4] = other_bbox_yaw - ANGLE
                tmp[5] = float(other_l)
                tmp[6] = float(other_w)

                tmp[7:12] = ohe(5, other_agent_type, True)

                tmp[43] = timestamp

                tmp[44] = GLOBAL_IDX
                tmp[45] = 1 if other_agent_id == agent_id else 0
                tmp[46] = other_predict
                tmp[47] = other_agent_id

                RES.append(tmp)
        local_roadlines_state[:, 44] = local_roadlines_state[:, 44] + GLOBAL_IDX + 1
        RES = np.concatenate((make_2d(RES), local_roadlines_state), axis=0)
        GRES.append(RES)

    return GRES


def merge(
    data, proc_id, validate, out_dir, use_vectorize=False, max_rand_int=10000000000
):
    parsed = tf.io.parse_single_example(data, features_description)
    raster_data = rasterize(
        parsed["state/tracks_to_predict"].numpy(),
        parsed["state/past/x"].numpy(),
        parsed["state/past/y"].numpy(),
        parsed["state/current/x"].numpy(),
        parsed["state/current/y"].numpy(),
        parsed["state/current/bbox_yaw"].numpy(),
        parsed["state/past/bbox_yaw"].numpy(),
        parsed["state/past/valid"].numpy(),
        parsed["state/current/valid"].numpy(),
        parsed["state/type"].numpy(),
        parsed["roadgraph_samples/xyz"].numpy(),
        parsed["roadgraph_samples/type"].numpy(),
        parsed["roadgraph_samples/valid"].numpy(),
        parsed["roadgraph_samples/id"].numpy(),
        parsed["state/current/width"].numpy(),
        parsed["state/current/length"].numpy(),
        parsed["state/id"].numpy(),
        parsed["traffic_light_state/current/state"].numpy(),
        parsed["traffic_light_state/current/id"].numpy(),
        parsed["traffic_light_state/current/valid"].numpy(),
        parsed["state/future/x"].numpy(),
        parsed["state/future/y"].numpy(),
        parsed["state/future/valid"].numpy(),
        parsed["scenario/id"].numpy()[0].decode("utf-8"),
        validate=validate,
    )

    if use_vectorize:
        vector_data = vectorize(
            parsed["state/past/x"].numpy(),
            parsed["state/current/x"].numpy(),
            parsed["state/past/y"].numpy(),
            parsed["state/current/y"].numpy(),
            parsed["state/past/valid"].numpy(),
            parsed["state/current/valid"].numpy(),
            parsed["state/past/speed"].numpy(),
            parsed["state/current/speed"].numpy(),
            parsed["state/past/vel_yaw"].numpy(),
            parsed["state/current/vel_yaw"].numpy(),
            parsed["state/past/bbox_yaw"].numpy(),
            parsed["state/current/bbox_yaw"].numpy(),
            parsed["state/id"].numpy(),
            parsed["state/type"].numpy(),
            parsed["roadgraph_samples/id"].numpy(),
            parsed["roadgraph_samples/type"].numpy(),
            parsed["roadgraph_samples/valid"].numpy(),
            parsed["roadgraph_samples/xyz"].numpy(),
            parsed["traffic_light_state/current/id"].numpy(),
            parsed["traffic_light_state/current/state"].numpy(),
            parsed["traffic_light_state/current/valid"].numpy(),
            parsed["state/current/width"].numpy(),
            parsed["state/current/length"].numpy(),
            parsed["state/tracks_to_predict"].numpy(),
            parsed["state/future/valid"].numpy(),
            validate=validate,
        )

    for i in range(len(raster_data)):
        if use_vectorize:
            raster_data[i]["vector_data"] = vector_data[i].astype(np.float16)

        r = np.random.randint(max_rand_int)
        filename = f"{idx2type[int(raster_data[i]['self_type'])]}_{proc_id}_{str(i).zfill(5)}_{r}.npz"
        np.savez_compressed(os.path.join(out_dir, filename), **raster_data[i])


def main():
    args = parse_arguments()
    print(args)

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    files = os.listdir(args.data)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(args.data, f) for f in files], num_parallel_reads=1
    )
    if args.n_shards > 1:
        dataset = dataset.shard(args.n_shards, args.each)

    p = multiprocessing.Pool(args.n_jobs)
    proc_id = 0
    res = []
    for data in tqdm(dataset.as_numpy_iterator()):
        proc_id += 1
        res.append(
            p.apply_async(
                merge,
                kwds=dict(
                    data=data,
                    proc_id=proc_id,
                    validate=not args.no_valid,
                    out_dir=args.out,
                    use_vectorize=args.use_vectorize,
                ),
            )
        )

    for r in tqdm(res):
        r.get()


if __name__ == "__main__":
    main()
