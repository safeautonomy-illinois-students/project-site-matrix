import csv
import numpy as np
import yaml


def offset_lane(points: np.ndarray, offset: float) -> np.ndarray:
    tangents = np.diff(
        points,
        axis=0,
        append=[points[-1] + (points[-1] - points[-2])]
    )
    
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    unit_normals = normals / magnitudes
    
    return points + unit_normals * offset


def closest_point_on_path(path: np.ndarray, p: np.ndarray) -> np.ndarray:
    A = path[:-1]
    B = path[1:]
    
    vec_AB = B - A
    vec_AP = p - A
    
    t = np.sum(vec_AP * vec_AB, axis=1) / np.sum(vec_AB**2, axis=1)
    t = np.clip(t, 0, 1)
    
    candidates = A + t[:, np.newaxis] * vec_AB
    
    distances = np.linalg.norm(candidates - p, axis=1)
    best_idx = np.argmin(distances)
    
    return candidates[best_idx], vec_AB[best_idx]


class WorldGT:
    def __init__(
            self,
            world: str,
            x: float=None,
            y: float=None,
            yaw: float=None) -> None:
        data = {}

        with open(f"resources/{world}_centerline.csv") as f:
            reader = csv.reader(f)
            map = {}
            for i, row in enumerate(reader):
                if i == 0:
                    for j, item in enumerate(row):
                        item = item.replace("#", "").replace(" ", "")
                        data[item] = []
                        map[j] = item
                else:
                    for j, item in enumerate(row):
                        data[map[j]].append(float(item))
        
        self._data = {
            "centerline": np.array([data["x_m"], data["y_m"]]).transpose(1, 0)
        }

        with open(f"resources/{world}_parameters.yaml") as f:
            data = yaml.safe_load(f)
        
        if "lanes" in data:
            for lane in data["lanes"]:
                self._data[lane] = offset_lane(
                    self._data["centerline"],
                    data["lanes"][lane]["offset"] / data["world"]["scale"]
                )
        if "centerline" not in data["lanes"]:
            self._data.pop("centerline")

        if x is None:
            x = data["world"]["x"]
        
        if y is None:
            y = data["world"]["y"]
        
        if yaw is None:
            yaw = data["world"]["yaw"]
        yaw = np.radians(yaw)
        
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([
            [c, -s], 
            [s,  c]
        ])
        
        for lane in self._data:
            lane = self._data[lane]
            lane *= data["world"]["scale"]
            lane = lane @ R.T
            lane[:, 0] += x
            lane[:, 1] += y

    def get_metrics(self, x: float, y: float, yaw: float) -> tuple[str, np.ndarray, float, float]:
        pt = np.array([x, y])

        lane = None
        closest = None
        tangent = None
        dist = float("inf")
        for curr_lane in self._data:
            curr_closest, curr_tangent = closest_point_on_path(self._data[curr_lane], pt)
            d = np.linalg.norm(pt - curr_closest)
            if d < dist:
                lane = curr_lane
                closest = curr_closest
                tangent = curr_tangent
                dist = d

        vec_path_to_vehicle = pt - closest
        side = tangent[0] * vec_path_to_vehicle[1] - tangent[1] * vec_path_to_vehicle[0]
        XTE = dist if side > 0 else -dist
        path_yaw = np.arctan2(tangent[1], tangent[0])
        HE = -((yaw - path_yaw + np.pi) % (2 * np.pi) - np.pi)
        return lane, closest, XTE, HE

    def show(self):
        import matplotlib.pyplot as plt
        for lane in self._data:
            pts = self._data[lane]
            plt.scatter(pts[:, 0], pts[:, 1], label=lane)
        plt.axis("equal")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    world = WorldGT("Silverstone")
    world.show()