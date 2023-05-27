# Cameron Shipman
# 
#
# Script to convert files in Snapshots to files suitable for YOLOv7.
# Also contains Label class and related helper functions.
#
# Run like this: python dataset_helper.py


import os
import shutil
import copy
import glob

try:
    import cv2
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install opencv-python' in a Windows command shell.")       # This will install numpy if not present.
    exit(1)

try:
    import numpy
except ModuleNotFoundError as err:
    print("Please install cv2. E.g. Run 'pip install numpy' in a Windows command shell.")
    exit(1)



class_id_to_info_dict = { 'reticle' : (0, (255, 0, 0)) }                        # Map class_id string to (index, bgr colour) tuple



#-------------------------------------------------------------------------------
# Class to describe a single label in a label file.
# Yolo requires these values, and the dimensions must be normalised.
# Also used to store original labels produced by a human before normalisation.
class Label:
    def __init__(self, class_id, x_centre, y_centre, width, height):            # Loaded as file as strings.
        self.class_id = class_id                                                # The name/id pf the detection class.
        self.x_centre = int(x_centre)
        self.y_centre = int(y_centre)
        self.width = int(width)
        self.height = int(height)

    def Normalise(self, max_w, max_h):                                          # Also converts dimensions to float.
        assert(max_w > 0 and max_w >= self.width and max_w > self.x_centre)
        assert(max_h > 0 and max_h >= self.height and max_h > self.y_centre)
        self.x_centre = float(self.x_centre) / float(max_w)
        self.y_centre = float(self.y_centre) / float(max_w)
        self.width = float(self.width) / float(max_w)
        self.height = float(self.height) / float(max_w)

    def __repr__(self):
        return f"Label('{self.class_id}', {self.x_centre}, {self.y_centre}, {self.width}, {self.height})"
    def __str__(self):
        return f"Label(class='{self.class_id}', cx={self.x_centre}, cy={self.y_centre}, w={self.width}, h={self.height})"

    def __lt__(self, other):                                                    # We sort on the class_id string.
        return self.class_id < other.class_id


#-------------------------------------------------------------------------------
# Read an unnormalised label file.
# filename      Name of file to read
# Returns list of unnormalised Label objects
def ReadUnnormalisedLabelFile(filename):
    # Currently no file checking is made.
    label_strings = numpy.loadtxt(fname=filename, dtype=str, ndmin=2)           # Always return a 2 dimensional array (if data present)
    print(label_strings)
    assert(len(label_strings) > 0)                                              # There must be at least one row.
    labels = []
    for l in label_strings:
        assert(len(l) == 5)                                                     # There must be 5 columns
        labels.append(Label(l[0], l[1], l[2], l[3], l[4]))
    return labels


#-------------------------------------------------------------------------------
def ConvertImageFilenameToLabelFilename(image_filename):
    return image_filename.rsplit(".", 1)[0] + ".txt"                                # Corresponding hand crafted label file.


#-------------------------------------------------------------------------------
# Draw a label. Perhaps this should be a method of Label class.
#   image       Image to modify.
#   l           Label class instance.
def DrawLabel(image, l):
    x1 = l.x_centre - l.width // 2
    y1 = l.y_centre - l.height // 2
    x2 = x1 + l.width
    y2 = y1 + l.height
    cv2.rectangle(image, (x1, y1), (x2, y2), class_id_to_info_dict[l.class_id][1], 2)
    #cv2.circle(image, (l.x_centre, l.y_centre), l.width // 2, (255, 0, 0), 2)
    cv2.circle(image, (l.x_centre, l.y_centre), 2, (255, 0, 0), 2)


#-------------------------------------------------------------------------------
if __name__ == '__main__':

    image_files = [f.replace("\\", "/") for f in glob.glob("Snapshots/*.tiff")]
    #print(image_files)
    #exit(0)

    original_image_w = 1250
    original_image_h = 750
    original_image_shape = (original_image_h, original_image_w, 3)              # cv2 shape: h, w, colours (channels)
    padding = (original_image_w - original_image_h) // 2                        # We pad top and bottom to make image square for yolo.

    # Create yolo dataset folder structure. Delete old one first.
    dataset_folder = "YOLOv7/yolov7/data/langham_dome_dataset"
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
    mode_folders = ['train', 'test', 'val']
    type_folders = ['images', 'labels']
    for mf in mode_folders:
        for tf in type_folders:
            os.makedirs(dataset_folder + '/' + mf + '/' + tf, exist_ok=False)       # Should not exist
    dataset_train_images_folder = dataset_folder + "/train/images"
    dataset_train_labels_folder = dataset_folder + "/train/labels"
    dataset_val_images_folder = dataset_folder + "/val/images"
    dataset_val_labels_folder = dataset_folder + "/val/labels"
    dataset_test_images_folder = dataset_folder + "/test/images"
    dataset_test_labels_folder = dataset_folder + "/test/labels"
    assert(os.path.exists(dataset_train_images_folder))
    assert(os.path.exists(dataset_train_labels_folder))
    assert(os.path.exists(dataset_val_images_folder))
    assert(os.path.exists(dataset_val_labels_folder))
    assert(os.path.exists(dataset_test_images_folder))
    assert(os.path.exists(dataset_test_labels_folder))

    # Use this to check bounding box centres etc:
    if False:
        image_file = "Snapshots/screen_area_4350.tiff"
        label_file = image_file.rsplit(".", 1)[0] + ".txt"                          # Corresponding hand crafted label file.
        labels = ReadUnnormalisedLabelFile(label_file)
        print(f'labels = {labels}')
        image = cv2.imread(image_file)
        assert(original_image_shape == image.shape)

        for l in labels:
            DrawLabel(image, l)
        cv2.imshow(image_file, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit(0)



    # Create list of images that have an accompanying label file.
    image_files_with_labels = []
    for image_file in sorted(image_files):
        label_file = image_file.rsplit(".", 1)[0] + ".txt"                         # Corresponding hand crafted label file.
        if os.path.exists(label_file):
            image_files_with_labels.append((image_file, label_file))
    #print(len(image_files_with_labels))
    #print(image_files_with_labels)
    #exit(0)

    # Read labels from Snapshots (master) folder.
    num_images = len(image_files_with_labels)
    print(f'Number of images with labels = {num_images}')
    image_count = 0
    # Indexes of files not used to train. When we have a larger dataset, then we can choose a random selection of a percentage of the files
    val_frames = [2500, 2670, 2790, 3370, 3590, 3720, 4080, 4260]                   # images used for valation
    test_frames = [2510, 2680, 2800, 3380, 3600, 3730, 4100, 4290]                  # images used for test.
    val_files = [f'Snapshots/screen_area_{n}.tiff' for n in val_frames]
    test_files = [f'Snapshots/screen_area_{n}.tiff' for n in test_frames]
    #print(val_files)
    #print(test_files)

    for image_file, label_file in sorted(image_files_with_labels):
        image_count += 1
        assert(image_count <= num_images)

        file_stub = os.path.basename(image_file).rsplit(".", 1)[0]

        if image_file in val_files:
            yolo_image_file = dataset_val_images_folder + '/' + file_stub + ".tiff"
            yolo_label_file = dataset_val_labels_folder + '/' + file_stub + ".txt"      # Normalised version of label_file, created below.
        elif image_file in test_files:
            yolo_image_file = dataset_test_images_folder + '/' + file_stub + ".tiff"
            yolo_label_file = dataset_test_labels_folder + '/' + file_stub + ".txt"     # Normalised version of label_file, created below.
        else:
            yolo_image_file = dataset_train_images_folder + '/' + file_stub + ".tiff"
            yolo_label_file = dataset_train_labels_folder + '/' + file_stub + ".txt"    # Normalised version of label_file, created below.

        #print(f'image_count = {image_count}')
        #print(f'yolo_image_file = {yolo_image_file}')
        #print(f'yolo_label_file = {yolo_label_file}\n')
        #continue

        print(f'{image_file} -> {label_file}')
        if os.path.exists(label_file):
            labels = ReadUnnormalisedLabelFile(label_file)
            assert(labels)
            print(f'labels = {labels}')

            image = cv2.imread(image_file)
            assert(original_image_shape == image.shape)

            # Image needs to be padded to make it square. Yolo requires square images.
            # Copy image to yolo folder at the same time.
            if os.path.exists(yolo_image_file):
                os.remove(yolo_image_file)
            #shutil.copyfile(image_file, yolo_image_file)
            image = cv2.imread(image_file)
            image = cv2.copyMakeBorder(src=image,
                                       top=padding, bottom=padding,
                                       left=0, right=0,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=(0,0,0))                           # Add (black) padding to top and bottom to make it square.
            image = cv2.resize(image, (640, 640))                               # Resize image to 640 x 640, which is what YOLOv7 requires.
            cv2.imwrite(yolo_image_file, image)

            # Normalise a copy of the labels to for writing to the yolo labels file.
            # At the same time convert original labels to new size, and adjust for padding. So we can display them.
            normalised_labels = []
            for l in labels:
                # Order important:
                l.y_centre += padding                                           # Adjust vertical position to allow for padding.
                nl = copy.copy(l)
                nl.Normalise(original_image_w, original_image_h)
                normalised_labels.append(nl)
                # Convert dims to 640 x 640 image size
                l.x_centre = l.x_centre * 640 // original_image_w
                l.y_centre = l.y_centre * 640 // original_image_w               # Use width because image is square now.
                l.width = l.width * 640 // original_image_w
                l.height = l.height * 640 // original_image_w                   # Use width because image is square now.
            assert(normalised_labels)
            print(f'normalised_labels = {normalised_labels}')

            # Write normalised labels to yolo folder.
            if os.path.exists(yolo_label_file):
                os.remove(yolo_label_file)
            ylf = open(yolo_label_file, "w")
            for nl in normalised_labels:
                ylf.write(f'{class_id_to_info_dict[nl.class_id][0]} {nl.x_centre} {nl.y_centre} {nl.width} {nl.height}')    # Convert class_id string to index.
            ylf.close()

            # Show the dataset images with bounding boxes drawn for checking.
            if False:                                                           # Enable if required.
                for l in labels:
                    x1 = l.x_centre - l.width // 2
                    y1 = l.y_centre - l.height // 2
                    x2 = x1 + l.width
                    y2 = y1 + l.height
                    cv2.rectangle(image, (x1, y1), (x2, y2), class_id_to_info_dict[l.class_id][1], 2)
                    #cv2.circle(image, (l.x_centre, l.y_centre), l.width // 2, (255, 0, 0), 2)
                    cv2.circle(image, (l.x_centre, l.y_centre), 2, (255, 0, 0), 2)
                cv2.rectangle(image, (0, 0), (638, 638), (0,255,0), 2)
                cv2.imshow(yolo_image_file, image)
                cv2.waitKey(0)
        else:
            print(" ** WARNING **: {image_file} has no corresponding label file!")


    cv2.destroyAllWindows()

    # Generate YOLOv7 data config file:
    data_cfg_file = "YOLOv7/yolov7/data/langham_dome.yaml"
    dcf = open(data_cfg_file, "w")
    dcf.write('train: data/langham_dome_dataset/train/images\n')
    dcf.write('val: data/langham_dome_dataset/val/images\n')
    dcf.write('test: data/langham_dome_dataset/test/images\n')
    dcf.write(f'\n# number of classes\nnc: {len(class_id_to_info_dict)}\n')
    classes_str = repr(list(class_id_to_info_dict.keys())).replace("'", '"')
    dcf.write(f'\n# class names\nnames: {classes_str}\n')
    dcf.close()

