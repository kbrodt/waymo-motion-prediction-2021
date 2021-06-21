import argparse

# chage this if you have problem
import sys
sys.path.insert(1, "~/.local/lib/python3.6/site-packages")


import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from submission_proto import motion_submission_pb2
from train import WaymoLoader, Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to CNN model"
    )
    parser.add_argument(
        "--time-limit", type=int, required=False, default=80, help="Number time steps"
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save predictions"
    )
    parser.add_argument(
        "--model-name", type=str, required=False, help="Model name"
    )

    parser.add_argument("--account-name", required=False, default="")
    parser.add_argument("--authors", required=False, default="")
    parser.add_argument("--method-name", required=False, default="SimpleCNNOnRaster")

    parser.add_argument("--batch-size", type=int, required=False, default=128)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)

    if args.model_path.endswith("pth"):
        model = Model(args.model_name)
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        model = torch.jit.load(args.model_path)

    model.cuda()
    model.eval()

    dataset = WaymoLoader(args.test_data, is_test=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=min(args.batch_size, 16)
    )

    RES = {}
    with torch.no_grad():
        for x, center, yaw, agent_id, scenario_id, _, _ in tqdm(loader):
            x = x.cuda()
            confidences_logits, logits = model(x)
            confidences = torch.softmax(confidences_logits, dim=1)

            logits = logits.cpu().numpy()
            confidences = confidences.cpu().numpy()
            agent_id = agent_id.cpu().numpy()
            center = center.cpu().numpy()
            yaw = yaw.cpu().numpy()
            for p, conf, aid, sid, c, y in zip(
                logits, confidences, agent_id, scenario_id, center, yaw
            ):
                if sid not in RES:
                    RES[sid] = []

                RES[sid].append(
                    {"aid": aid, "conf": conf, "pred": p, "yaw": -y, "center": c}
                )

    motion_challenge_submission = motion_submission_pb2.MotionChallengeSubmission()
    motion_challenge_submission.account_name = args.account_name
    motion_challenge_submission.authors.extend(args.authors.split(","))
    motion_challenge_submission.submission_type = (
        motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    )
    motion_challenge_submission.unique_method_name = args.method_name

    selector = np.arange(4, args.time_limit + 1, 5)
    for scenario_id, data in tqdm(RES.items()):
        scenario_predictions = motion_challenge_submission.scenario_predictions.add()
        scenario_predictions.scenario_id = scenario_id
        prediction_set = scenario_predictions.single_predictions

        for d in data:
            predictions = prediction_set.predictions.add()
            predictions.object_id = int(d["aid"])

            y = d["yaw"]
            rot_matrix = np.array([
                [np.cos(y), -np.sin(y)],
                [np.sin(y), np.cos(y)],
            ])

            for i in np.argsort(-d["conf"]):
                scored_trajectory = predictions.trajectories.add()
                scored_trajectory.confidence = d["conf"][i]

                trajectory = scored_trajectory.trajectory

                p = d["pred"][i][selector] @ rot_matrix + d["center"]

                trajectory.center_x.extend(p[:, 0])
                trajectory.center_y.extend(p[:, 1])

    with open(args.save, "wb") as f:
        f.write(motion_challenge_submission.SerializeToString())


if __name__ == "__main__":
    main()
