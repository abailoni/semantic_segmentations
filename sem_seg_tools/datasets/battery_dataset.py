import os
import numpy as np
from PIL import Image

from segmfriends.utils.various import writeHDF5


from copy import deepcopy
import numpy as np

try:
    from inferno.io.core import Zip, Concatenate
    from inferno.io.transform import Compose, Transform
    from inferno.io.transform.generic import AsTorchBatch
    from inferno.io.transform.volume import RandomFlip3D, VolumeAsymmetricCrop
    from inferno.io.transform.image import RandomRotate, ElasticTransform
except ImportError:
    raise ImportError("CremiDataset requires inferno")

try:
    from neurofire.datasets.loader import RawVolume, SegmentationVolume
    from neurofire.transform.artifact_source import RejectNonZeroThreshold
    from neurofire.transform.volume import RandomSlide
except ImportError:
    raise ImportError("CremiDataset requires neurofire")

from segmfriends.utils.various import yaml2dict
from segmfriends.transform.volume import MergeExtraMasks, DuplicateGtDefectedSlices, DownSampleAndCropTensorsInBatch, \
    ReplicateTensorsInBatch
from segmfriends.transform.affinities import affinity_config_to_transform, Segmentation2AffinitiesDynamicOffsets


class BatteryDataset(Zip):
    def __init__(self, name, volume_config, slicing_config,
                 transform_config=None):
        assert isinstance(volume_config, dict)
        assert isinstance(slicing_config, dict)
        assert 'raw' in volume_config
        assert 'ground_truth' in volume_config

        volume_config = deepcopy(volume_config)

        # Get kwargs for raw volume
        raw_volume_kwargs = dict(volume_config.get('raw'))

        raw_volume_kwargs.update(slicing_config)
        # Build raw volume
        self.raw_volume = RawVolume(name=name,
                                   **raw_volume_kwargs)

        # Get kwargs for segmentation volume
        segmentation_volume_kwargs = dict(volume_config.get('ground_truth'))
        segmentation_volume_kwargs.update(slicing_config)
        self.affinity_config = segmentation_volume_kwargs.pop('affinity_config', None)
        # Build segmentation volume
        self.segmentation_volume = SegmentationVolume(name=name,
                                                      **segmentation_volume_kwargs)

        volumes_to_load = [self.raw_volume, self.segmentation_volume]

        super().__init__(*volumes_to_load,
                         sync=True)

        # Set master config (for transforms)
        self.transform_config = {} if transform_config is None else deepcopy(transform_config)
        # Get transforms
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = Compose()

        if self.transform_config.get('random_flip', False):
            transforms.add(RandomFlip3D())
            transforms.add(RandomRotate())

        # Elastic transforms can be skipped by
        # setting elastic_transform to false in the
        # yaml config file.
        if self.transform_config.get('elastic_transform'):
            elastic_transform_config = self.transform_config.get('elastic_transform')
            if elastic_transform_config.get('apply', False):
                transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                                sigma=elastic_transform_config.get('sigma', 50.),
                                                order=elastic_transform_config.get('order', 0)))

        # Replicate and downscale batch:
        nb_inputs = 1
        if self.transform_config.get("downscale_and_crop") is not None:
            ds_config = self.transform_config.get("downscale_and_crop")
            apply_to  = [conf.pop('apply_to') for conf in ds_config]
            nb_inputs = (np.array(apply_to) == 0).sum()
            transforms.add(ReplicateTensorsInBatch(apply_to))
            for indx, conf in enumerate(ds_config):
                transforms.add(DownSampleAndCropTensorsInBatch(apply_to=[indx], order=None, **conf))

        # Check if to compute binary-affinity-targets from GT labels:
        if self.transform_config.get("affinity_config") is not None:
            affs_config = deepcopy(self.transform_config.get("affinity_config"))
            global_kwargs = affs_config.pop("global", {})

            aff_transform = Segmentation2AffinitiesDynamicOffsets if affs_config.pop("use_dynamic_offsets", False) \
                else affinity_config_to_transform

            for input_index in affs_config:
                affs_kwargs = deepcopy(global_kwargs)
                affs_kwargs.update(affs_config[input_index])
                transforms.add(aff_transform(apply_to=[input_index+nb_inputs], **affs_kwargs))

        # crop invalid affinity labels and elastic augment reflection padding assymetrically
        crop_config = self.transform_config.get('crop_after_target', {})
        if crop_config:
            # One might need to crop after elastic transform to avoid edge artefacts of affinity
            # computation being warped into the FOV.
            transforms.add(VolumeAsymmetricCrop(**crop_config))


        transforms.add(AsTorchBatch(3))
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        name = config.get('dataset_name')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        transform_config = config.get('transform_config')
        return cls(name, volume_config=volume_config,
                   slicing_config=slicing_config,
                   transform_config=transform_config)


def load_stack_of_tiff_as_numpy(folder_path, output_filename):

    # path = "/home/abailoni_local/trendyTukan_localdata0/datasets/battery_data/Daten for AlbertoBailoni"
    # Z, Y, X
    # TODO: generalize
    collected_dataset = np.empty((100, 1080, 1197, 2), dtype='uint8')
    for filename in os.listdir(folder_path):
        if filename.endswith(".tiff"):
            im = Image.open(os.path.join(folder_path, filename))
            imarray = np.array(im)
            parts = filename.split("_")
            slice_nb = int(parts[-1].split(".")[0]) - 1
            if parts[0] == "2ND":
                collected_dataset[slice_nb, :, 4:, 0] = imarray
            else:
                imarray = imarray[:-4]
                collected_dataset[slice_nb, :, :-4, 1] = imarray
    #         channel_nb = 0 if parts[0] == "2ND" else 1
    #         collected_dataset[slice_nb, :, ]
    #         print(slice_nb, channel_nb)
    #         print(imarray.shape, parts[0])
    #         all_arrays[]
    #          print(os.path.join(directory, filename))
    #         continue
        else:
            continue
    print(collected_dataset.max())

    writeHDF5(collected_dataset,
              os.path.join(folder_path, output_filename),
              "data")
