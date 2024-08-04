import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision.transforms import functional as trans_fn

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.AB_paths = []
        self.cluster_labels = []
        for cl_num in range(opt.num_clusters):
            self.dir_AB = os.path.join(opt.dataroot, str(cl_num), opt.phase)  # get the image directory
            temp_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
            self.cluster_labels+=[cl_num]*len(temp_paths)
            self.AB_paths += temp_paths  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        real_img_transform = get_transform(self.opt, transform_params, grayscale=False)

        img_name = AB_path.split("/")[-1]
        real_img_path = "/".join(AB_path.split("/")[:3])
        real_input = Image.open(os.path.join(real_img_path, "all_images_release", img_name))
        real_input = trans_fn.resize(real_input, (w2,h))
        A = A_transform(A)
        B = B_transform(B)
        real_input = real_img_transform(real_input)

        return {'inp_image': real_input, 'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'cluster': self.cluster_labels[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
