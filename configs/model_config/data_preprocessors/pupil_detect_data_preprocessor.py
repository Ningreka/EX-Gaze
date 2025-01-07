from model.data_preprocessor.ev_pupil_preprocessor import EvPupilDataPreprocessor

pupil_detect_data_preprocessor = dict(
    type=EvPupilDataPreprocessor,
    mean=[77],
    std=[39],
    # mean=[85],
    # std=[40],
    pad_size_divisor=32,
    pad_value=0,
    boxtype2tensor=True,
    img_aug=True
)
