import torch

def get_ofa_supernet_mbv3_w10():
    return torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_mbv3_w10', pretrained=True)

def get_ofa_supernet_mbv3_w12():
    return torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_mbv3_w12', pretrained=True)

def get_ofa_supernet_resnet50():
    return torch.hub.load('mit-han-lab/once-for-all', 'ofa_supernet_resnet50', pretrained=True)

def get_architecture_dict(ofa_supernet_name: str):
    assert ofa_supernet_name in ["ofa_supernet_mbv3_w10", "ofa_supernet_mbv3_w12", "ofa_supernet_resnet50"]
    if ofa_supernet_name == "ofa_supernet_mbv3_w10":
        return {
            "ks": {
                "length": 20,
                "choices": [3, 5, 7],
            },
            "e": {
                "length": 20,
                "choices": [3, 4, 6],
            },
            "d": {
                "length": 5,
                "choices": [2, 3, 4],
            }
        }
    elif ofa_supernet_name == "ofa_supernet_mbv3_w12":
        return {
            "ks": {
                "length": 20,
                "choices": [3, 5, 7],
            },
            "e": {
                "length": 20,
                "choices": [3, 4, 6],
            },
            "d": {
                "length": 5,
                "choices": [2, 3, 4],
            }
        }
    elif ofa_supernet_name == "ofa_supernet_resnet50":
        return {
            "d": {
                "length": 5,
                "choices": [0, 1, 2],
            },
            "e": {
                "length": 18,
                "choices": [0.2, 0.25, 0.35],
            },
            "w": {
                "length": 6,
                "choices": [0, 1, 2],
            }
        }
