# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model

def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="the_end",
                        help="model architecture ")
    parser.add_argument("--weight_file", type=str, default='./weights/rain200L/weights.120-15.616-37.28831.hdf5',
                        help="trained weight file")
    parser.add_argument("--If_n", type=bool, default=False,
                        help="If normalizing the image")
    parser.add_argument("--output_dir", type=str, default='',
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return np.uint8(image)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    if_n = args.If_n
    model = get_model(args.model)
    model.load_weights(weight_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if if_n:
            image = image/255.0
        noise_image = image
        pred = model.predict(np.expand_dims(noise_image, 0))
        print(pred[1][0].max())
        print(pred[1][0].min())
        if if_n:
            denoised_image = get_image(pred[1][0]*[255])
        else:
            denoised_image = get_image(pred[1][0])



        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name[:-4] + ".png")), denoised_image)

        else:
            cv2.imshow("result", denoised_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
