import os
import json
import numpy as np


def main():
    # intrinsics (K) from /front_single_camera/camera_info
    K = np.array([
        [476.7030836014194, 0.0, 400.0],
        [0.0, 476.7030836014194, 300.0],
        [0.0, 0.0, 1.0]
    ])
    # extrinsics (R,t): base_footprint -> front_single_camera_link
    R = np.array([
        [0, -1, 0],
        [ 0, 0, 1],
        [ -1, 0, 0]
    ])
    t = np.array([0.160, -0.110, 1.546])

    # bird eye view image size (px); 
    # NOTE: DO NOT CHANGE THIS
    bev_img_height, bev_img_width = 600, 800

    # HYPERPARAMETERS: BEV rectangle configuration (m)
    # NOTE: Revert this to the original values for submission
    bev_height, bev_width = 15, 20

    # px -> m conversion factor
    # tells the physical scale of each bev pixel
    unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)

    # 4 points on the ground
    # X = 0 to 15 m forward, and then y = -10m to +10m sideways
    bev_world_coords = np.float32([
        [bev_height, -bev_width/2, 0],
        [0, -bev_width/2, 0],
        [0, bev_width/2, 0],
        [bev_height, bev_width/2, 0],
    ])

    # convert the bev_world_coords into pixel coordinates
    src = [] # this is basically to asnwer: where the four world-rectangle corners appear in the front-facing camera image
    for pt in bev_world_coords:
        ##### YOUR CODE STARTS HERE #####
        # pt is [X, Y, Z] in world coordinates (Z should be 0 since it's on the ground)
        Pw = pt.reshape(3, 1)                    

        # world -> camera
        Pc = R @ Pw - t.reshape(3, 1)             

        # camera -> pixel (homogeneous)
        p = K @ Pc                               

        # normalize homogeneous coords
        u = float(p[0, 0] / p[2, 0])
        v = float(p[1, 0] / p[2, 0])

        src.append([u, v])
        ##### YOUR CODE ENDS HERE #####
        # pass
    src = np.float32(src)

    output = {
        "bev_world_dim": (bev_height, bev_width),
        "unit_conversion_factor": unit_conversion_factor,
        "src": src.tolist(),
    }
    # save config to json
    save_fn = 'data/bev_config.json'
    if not os.path.isdir('data/'):
        print(f"Data directory not found. Generating...")
        os.makedirs('data/', exist_ok=False)
    if os.path.isfile(save_fn):
        if input("File already exists. Overwrite? (y/n):").lower() != 'y':
            print("Exiting...")
            import sys
            sys.exit()
    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved BEV config to {save_fn}.")


if __name__ == "__main__":
    main()
