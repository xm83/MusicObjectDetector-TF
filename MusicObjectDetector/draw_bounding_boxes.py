import cv2
import argparse

from PIL import ImageColor
from muscima.io import parse_cropobject_list

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_boxes_into_image(image_path: str, ground_truth_annotations_path: str, destination_path: str):
    crop_objects = parse_cropobject_list(ground_truth_annotations_path)
    img = cv2.imread(image_path, True)

    classes = list(set([c.clsname for c in crop_objects]))

    for index, crop_object in enumerate(crop_objects):
        # String to float, float to int
        x1 = crop_object.left
        y1 = crop_object.top
        x2 = crop_object.right
        y2 = crop_object.bottom

        color_name = STANDARD_COLORS[classes.index(crop_object.clsname) % len(STANDARD_COLORS)]
        color = ImageColor.getrgb(color_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        # cv2.putText(img, crop_object.clsname + '/' + str(index + 1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    cv2.imwrite(destination_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw the bounding boxes from the ground-truth data.')
    parser.add_argument('-img', dest='img_path', type=str, required=True, help='Path to the image.',
                        default="data/cvcmuscima_staff_removal/CvcMuscima-Distortions/ideal/w-14/image/p008.png")
    parser.add_argument('-gt', dest='gt_path', type=str, required=True, help='Path to the ground truth.',
                        default="data/muscima_pp_raw/v1.0/data/cropobjects_manual/CVC-MUSCIMA_W-14_N-08_D-ideal.xml")
    parser.add_argument('-save', dest='save_img', type=str, required=True, help='Path to save the processed image.',
                        default="annotated_gt_image.png")
    args = parser.parse_args()

    draw_bounding_boxes_into_image(args.img_path, args.gt_path, args.save_img)
