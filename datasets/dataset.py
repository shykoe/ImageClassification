from PIL import Image
import os
import os.path
import sys
import torchvision
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(dir, extensions=None):
    images = []
    dir = os.path.expanduser(dir)

    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, os.path.splitext(os.path.basename(fname))[0])
                images.append(item)
    return images
class ExtractionDatasetFolder(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, loader = pil_loader, extensions=IMG_EXTENSIONS,
                target_transform=None):
        super(ExtractionDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.samples = make_dataset(root, extensions = IMG_EXTENSIONS)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        self.loader = loader
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    def __len__(self):
        return len(self.samples)
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    import utils
    ##test make_dataset
    data = make_dataset('/Users/shykoe/imagespro/data', 
    extensions=IMG_EXTENSIONS)
    ## test ExtractionDatasetFolder
    dataFolder = ExtractionDatasetFolder('/Users/shykoe/imagespro/data')