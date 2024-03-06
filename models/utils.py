from models.sync_batchnorm import DataParallelWithCallback
from config import Config



def put_on_multi_gpus(model):
    model = DataParallelWithCallback(model, device_ids=Config.gpu_ids).cuda()

    # assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model
