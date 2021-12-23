from torch.utils.data.dataset import Dataset as BaseDataset
from torch.utils.data.dataloader import DataLoader
class Reader():
    """
    to read the data

    feed_dict for dataloader:
        use dict to load (convenient for later use)
        should contain:
            im_data,im_info,gt_boxes,num_box
            gt_boxes: ground true roi and bbox (0-3bbox, 4target)
            im_info: w,h,scale
    """
    def __init__(self,args):
        self.dataset=args.dataset_name
        self.dataset_path=args.dataset_path

    def preprocess(self):
        """
        preprocess the file (dataset path)
        :return:
        """

    def get_loader(self,phase:str,batch_size:int)->DataLoader:
        """
        :param phase: train or test (may be has val set
        :param batch_size: the batch size of dataloader
        :return: a dataloader for train or test
        """