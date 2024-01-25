import os
from itertools import chain
from pathlib import Path
from typing import Callable

from PIL import Image


def cut_ampl_and_phase(img_filename: Path | str, updated_version: bool = True):
    """coordinate system
    .→
    ↓
    """
    img_filename = Path(img_filename)
    Path("ampl&phase/").mkdir(exist_ok=True)
    img = Image.open(img_filename)

    pics_dist_x = 630  # distance between maps

    if updated_version:  # TODO: 3 versions or smth
        size_x, size_y = 171, 171  # size of map
        shift_x, shift_y = -723, -342  # amplitude's coordinates

    else:
        size_x, size_y = 160, 149  # size of map
        shift_x, shift_y = -708, -307  # amplitude's coordinates

        # size_x, size_y = 162, 151  # size of map
        # shift_x, shift_y = -701, -323  # amplitude's coordinates

    # top left corner amplitude's coordinates # TODO img.width -> -img.width
    ampl_left = img.width / 2 + shift_x
    ampl_top = img.height / 2 + shift_y

    ampl_box = (ampl_left, ampl_top, ampl_left + size_x, ampl_top + size_y)
    ampl_img = img.crop(ampl_box)
    ampl_img.save("ampl&phase/" + img_filename.stem + " (ampl)" + img_filename.suffix)

    # shift box by pics_shift_x
    phase_box = tuple(
        elem + pics_dist_x if i % 2 == 0 else elem for i, elem in enumerate(ampl_box)
    )
    phase_img = img.crop(phase_box)
    phase_img.save("ampl&phase/" + img_filename.stem + " (phase)" + img_filename.suffix)
    # phase_img.show()


def resize_image(img_filename: Path | str, pics_per_row: int = 2):
    if pics_per_row not in (1, 2):
        return

    target_width = 704
    img_filename = Path(img_filename)
    img = Image.open(img_filename)
    width, height = img.size

    if target_width < width:
        resize_coeff = width / target_width
        img = img.resize((int(target_width), int(height / resize_coeff)))

    # filename = img_filename.stem + " (compressed)" + img_filename.suffix
    path = "compressed" / Path(*img_filename.parent.parts[1:])
    path.mkdir(exist_ok=True, parents=True)
    img.save(Path(path, img_filename.name), optimize=True)


def do_in_folder(
    func: Callable, folder: Path | str = ".", pattern: str = "**/*", *args, **kwargs
):
    pathlist = Path(folder).glob(pattern)
    for file in pathlist:
        if file.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            func(file, *args, **kwargs)


if __name__ == "__main__":
    # do_in_folder(
    #     cut_ampl_and_phase, folder="./data", pattern="./fig*.*", updated_version=True
    # )
    # do_in_folder(resize_image, "./pics", pics_per_row=2)

    cut_ampl_and_phase(
        "data/#7752_2/figure_2023_11_29__23_02_04.png", updated_version=True
    )
