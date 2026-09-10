"""
The saliency models under consideration for Vision Lab (Step 8).

WHAT DECIDES THE WINNER
    Not the highest NSS on MIT1003. The decision metric is mean NSS on ~100
    frames of OUR OWN ads, hand-annotated with what should draw the eye. The
    academic benchmarks are the sanity check; a model that reads natural photos
    beautifully and ignores headline text is no use in an ad report.

    And no model wins on accuracy alone: it also has to clear the CPU budget on
    the production VPS, and its licence has to permit commercial use.

WHY THE CENTRE BASELINE IS IN THIS LIST
    It is not a candidate, it is the control. Human fixation data is heavily
    centre-biased, so a plain Gaussian blob in the middle of the frame scores
    surprisingly well on AUC and NSS. Any model that does not clearly beat it is
    buying us nothing, and without it in the table that is invisible.

LICENCES ARE NOT RECORDED FROM MEMORY
    Every entry below says `licence: None`. fetch_models.py reads the actual
    LICENSE file out of each repository and reports what it finds. Several
    published saliency checkpoints are research-only; ScaleSerum is commercial,
    so that check is a gate before any evaluation work, not a footnote after it.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- datasets
# WHY THIS REGISTRY EXISTS
#     A repository's LICENSE file grants rights over its CODE. Whether it grants
#     anything over the trained WEIGHTS is usually unstated, and the weights were
#     produced from datasets with terms of their own. Whether those terms follow
#     the weights is unsettled law and varies by jurisdiction - which is exactly
#     why the training data is recorded per model rather than assumed harmless.
#
# NOTHING HERE IS ASSERTED FROM MEMORY
#     Every `terms` field is None. `verify_at` is where to actually read them.
#     `concern` records a specific, checkable reason to look harder - not a
#     verdict.
DATASETS = {
    "salicon": {
        "name": "SALICON",
        "verify_at": "http://salicon.net",
        "what": "~10k images from MS-COCO with mouse-tracking used as a fixation proxy",
        "terms": None,
        "concern": None,
    },
    "imagenet": {
        "name": "ImageNet (backbone pre-training)",
        "verify_at": "https://www.image-net.org/download.php",
        "what": "Classification pre-training for the encoder (VGG16 / ResNet-50 / DenseNet)",
        "terms": None,
        "concern": ("Non-commercial research terms are commonly cited, yet "
                    "ImageNet-pretrained backbones are near-universal in shipped "
                    "products. Widespread practice is not permission - flag it and "
                    "let legal decide how much it matters."),
    },
    "mit1003": {
        "name": "MIT1003",
        "verify_at": "https://people.csail.mit.edu/tjudd/WherePeopleLook/",
        "what": "1003 images with real eye-tracking from 15 observers",
        "terms": None,
        "concern": "Evaluation only for us - never redistributed and never deployed.",
    },
    "cat2000": {
        "name": "CAT2000",
        "verify_at": "https://saliency.tuebingen.ai/datasets.html",
        "what": "2000 training images across 20 scene categories",
        "terms": None,
        "concern": "Evaluation only for us - never redistributed and never deployed.",
    },
    "dhf1k": {
        "name": "DHF1K",
        "verify_at": "https://github.com/wenguanwang/DHF1K",
        "what": "1000 videos with eye-tracking, for video saliency",
        "terms": None,
        "concern": "Research dataset. Check whether its terms permit commercial derived weights.",
    },
    "hollywood2": {
        "name": "Hollywood-2",
        "verify_at": "https://www.di.ens.fr/~laptev/actions/hollywood2/",
        "what": "Action clips taken from Hollywood feature films",
        "terms": None,
        "concern": ("THE BIGGEST FLAG IN THIS FILE. The clips are excerpts from "
                    "commercial films the dataset authors did not own. Weights "
                    "trained on it carry a question no LICENSE file in the model "
                    "repo can answer."),
    },
    "ucf_sports": {
        "name": "UCF-Sports",
        "verify_at": "https://www.crcv.ucf.edu/data/UCF_Sports_Action.php",
        "what": "Sports action clips from broadcast footage",
        "terms": None,
        "concern": "Broadcast footage. Same shape of question as Hollywood-2, smaller in scale.",
    },
}

# Weight delivery. Repos that keep weights on Google Drive cannot be fetched
# unattended, and the script says so rather than appearing to succeed.
IN_REPO = "in_repo"            # committed alongside the code
DIRECT_URL = "direct_url"      # a plain HTTPS download
MANUAL = "manual"              # Drive/Dropbox - needs a human
PIP = "pip"                    # installable package that fetches its own weights

CANDIDATES = [
    {
        "id": "unisal",
        "name": "UNISAL",
        "repo": "https://github.com/rdroste/unisal",
        "paper": "Unified Image and Video Saliency Modelling (ECCV 2020)",
        "weights": IN_REPO,
        "weights_hint": "training_runs/pretrained_unisal/weights_best.pth",
        "weights_licence_declared": None,
        "weights_licence_url": None,
        "trained_on": ["salicon", "dhf1k", "hollywood2", "ucf_sports"],
        "params_m": 3.7,
        "licence": None,
        "why": ("One small model covering BOTH image and video saliency - our two "
                "input kinds from one artefact. Smallest of the candidates, so the "
                "most likely to clear the CPU budget."),
        "watch": ("Video mode expects 16-32 consecutive frames at native fps. Our "
                  "2 fps samples are not that, so we would use its image mode "
                  "unless the video path is evaluated separately."),
    },
    {
        "id": "msinet",
        "name": "MSI-Net",
        "repo": "https://github.com/alexanderkroner/saliency",
        "paper": "Contextual Encoder-Decoder Network for Visual Saliency Prediction",
        "weights": DIRECT_URL,
        "weights_hint": "GitHub Releases, plus HuggingFace and Kaggle model hubs",
        "weights_licence_declared": "MIT (declared against the model itself)",
        "weights_licence_url": "https://huggingface.co/alexanderkroner/MSI-Net",
        "trained_on": ["salicon", "imagenet"],
        "params_m": 25.0,
        "licence": None,
        "why": "Well-established baseline. The reference floor the others must beat.",
        "watch": "Originally TensorFlow; check whether the release includes an ONNX or PyTorch build.",
    },
    {
        "id": "transalnet",
        "name": "TranSalNet",
        "repo": "https://github.com/LJOVO/TranSalNet",
        "paper": "TranSalNet: Towards perceptually relevant visual saliency prediction",
        "weights": MANUAL,
        "weights_hint": "Google Drive link in the repo README",
        "weights_licence_declared": None,
        "weights_licence_url": None,
        "trained_on": ["salicon", "imagenet"],
        "params_m": 100.0,
        "licence": None,
        "why": "Transformer-based, among the strongest recent SALICON numbers.",
        "watch": ("By far the heaviest here. Likely to lose on CPU latency even if "
                  "it wins on accuracy - which is exactly the trade the bake-off "
                  "exists to make explicit."),
    },
    {
        "id": "deepgaze2e",
        "name": "DeepGaze IIE",
        "repo": "https://github.com/matthias-k/DeepGaze",
        "paper": "DeepGaze IIE: Calibrated prediction in and out-of-domain (ICCV 2021)",
        "weights": PIP,
        "weights_hint": "downloaded on first use by the package",
        "weights_licence_declared": None,
        "weights_licence_url": None,
        "trained_on": ["salicon", "mit1003"],
        "params_m": 120.0,
        "licence": None,
        "why": ("Probabilistic and explicitly principled about centre bias, which "
                "matters more for ads than for natural photos."),
        "watch": "An ensemble - heavy. Check whether a single-branch export is viable.",
    },
    {
        "id": "umsi",
        "name": "UMSI / predimportance",
        "repo": "https://github.com/diviz-mit/predimportance-public",
        "paper": "Predicting Visual Importance Across Graphic Design Types (UIST 2020)",
        "weights": MANUAL,
        "weights_hint": "linked from the repo README",
        "weights_licence_declared": None,
        "weights_licence_url": None,
        "trained_on": ["salicon", "imagenet"],
        "params_m": None,
        "licence": None,
        "why": ("Trained on DESIGNED images - ads, posters, infographics - rather "
                "than natural photos. Natural-image saliency systematically "
                "under-weights text blocks and logos, which is precisely what an "
                "ad report is about. Worth evaluating even if it loses on MIT1003."),
        "watch": "Predicts 'importance', not fixation. Compare on our ad frames, not on MIT1003.",
    },
]

# Not a candidate. The control - see the module docstring.
CENTRE_BASELINE = {
    "id": "centre_gaussian",
    "name": "Centre Gaussian (control)",
    "repo": None,
    "licence": "n/a - generated, not downloaded",
    "why": ("Human fixations are heavily centre-biased. This scores what you get "
            "for free by guessing 'the middle', and any model that does not "
            "clearly beat it is not earning its place."),
}

# The classical method Vision Lab runs today, carried into the table so the
# comparison shows what we are actually replacing.
CURRENT_BASELINE = {
    "id": "spectral_residual",
    "name": "Spectral Residual (in production now)",
    "repo": None,
    "licence": "n/a - algorithm implemented in vision_lab/saliency.py",
    "why": "What ships today. The bake-off has to show a real improvement over it.",
}


def by_id(candidate_id: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["id"] == candidate_id:
            return candidate
    raise KeyError(f"unknown candidate: {candidate_id}")
