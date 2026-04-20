import os
import torch
import numpy as np
import cv2
try:
    from .simple_enet import SimpleENet
except Exception:
    from simple_enet import SimpleENet


##### YOUR CODE STARTS HERE #####
# DO NOT CHANGE ANY FUNCTION HEADERS

# load your best model
def load_model() -> SimpleENet:
    path_to_your_model = os.environ.get(
        "LANE_MODEL_PATH",
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "checkpoints",
            "epoch100.pth",
        ),
    )
    model = SimpleENet()
    model.load_state_dict(torch.load(path_to_your_model, weights_only=True))
    return model

def inference(model: SimpleENet, image: np.ndarray, device: str) -> np.ndarray:
    """
    The main image processing pipeline for your model
    
    :param model: pytorch model
    :type model: SimpleENet
    :param image: a BGR image taken from the GEM's camera
    :type image: np.ndarray
    :param device: the device on which the model should run on ("cpu" or "cuda")
    :type device: str
    :return: binary lane-segmented image
    :rtype: ndarray
    """
    # pred = None
    # return pred
    H0, W0 = image.shape[:2]

    # resizing to model input
    img = cv2.resize(image, (640, 384), interpolation=cv2.INTER_LINEAR)

    # grayscale + normalize to [0,1] float
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # to tensor
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    dev = torch.device(device)
    model = model.to(dev)
    x = x.to(dev)

    # generates [1. 2, 384, 648]
    with torch.no_grad():
        yp = model(x)

    pred = torch.argmax(yp, dim=1).squeeze(0).to("cpu").numpy().astype(np.uint8)
    pred = pred * 255
    pred = cv2.resize(pred, (W0, H0), interpolation=cv2.INTER_NEAREST)

    return pred


##### YOUR CODE ENDS HERE #####
