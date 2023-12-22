import os
from PIL import Image
import webdataset as wds
from ad_invento.datasets.datasets.base_dataset import BaseDataset
from ad_invento.datasets.datasets.caption_datasets import CaptionDataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = ann["image_id"]
        image_path = img_file
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }



# import os
# from PIL import Image
# import webdataset as wds
# from ad_invento.datasets.datasets.base_dataset import BaseDataset
# from ad_invento.datasets.datasets.caption_datasets import CaptionDataset
# import pydicom as dicom
# import matplotlib.pylab as plt
# from PIL import Image
# import numpy as np

# class CCSBUDataset(BaseDataset):
#     def __init__(self, vis_processor, text_processor, location):
#         super().__init__(vis_processor=vis_processor, text_processor=text_processor)

#         self.inner_dataset = wds.DataPipeline(
#             wds.ResampledShards(location),
#             wds.tarfile_to_samples(handler=wds.warn_and_continue),
#             wds.shuffle(1000, handler=wds.warn_and_continue),
#             wds.decode("pilrgb", handler=wds.warn_and_continue),
#             wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
#             wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
#             wds.map(self.to_dict, handler=wds.warn_and_continue),
#         )

#     def to_dict(self, sample):
#         return {
#             "image": sample[0],
#             "text_input": self.text_processor(sample[1]["caption"]),
#         }


# class CCSBUAlignDataset(CaptionDataset):

#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation[index]
        
#         caption = ann["caption"]
#         img_path = ann["image_id"]
#         ds = dicom.dcmread(img_path)
#         image_array = ds.pixel_array
#         pixel_data_normalized = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
#         rgb_image = Image.fromarray(pixel_data_normalized, mode='L').convert('RGB')
#         image = self.vis_processor(rgb_image)
        
        

#         return {
#             "image": image,
#             "text_input": caption,
#             "image_id": self.img_ids[ann["image_id"]],
#         }