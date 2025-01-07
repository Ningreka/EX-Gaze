from model.data_preprocessor.ev_pupil_patch_preprocessor import EvPupilPatchPreprocessor

ev_pupil_patch_preprocessor = dict(
    type=EvPupilPatchPreprocessor,
    mean=None,
    std=None,
    patch_size=9,
    pad_value=0,
    boxtype2tensor=True,
    empty_filter=True
)
