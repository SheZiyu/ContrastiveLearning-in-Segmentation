import os
import sys
from datetime import datetime
from utils import *
import torch.backends.cudnn as cudnn
import time
import random
from network.unet2d import UNet2D
from skimage.transform import resize
from data.test_mmwhs import MMWHS
import numpy as np
import torch.nn.functional as F
from metrics import SegmentationMetric
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger
import SimpleITK as sitk
#from data.generate_mmwhs import maybe_mkdir_p
from skimage.transform import resize
import nibabel as nib
from thop import profile
# macs, params = profile(model, inputs=(inputs,))
# macs_new = macs/inputs.szie(0)
from skimage.measure import marching_cubes_lewiner
#from stl import mesh
def dataToMesh(vert, faces):
    mm = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mm.vectors[i][j] = vert[f[j],:]
    return mm

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60

    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def maybe_mkdir_k(dir_1, dir_2):
    if not os.path.exists(dir_1):
        os.mkdir(dir_1)
    if not os.path.exists(os.path.join(dir_1, dir_2)):
        os.mkdir(os.path.join(dir_1, dir_2))


# def get_size(split = "test"):
#     json_path = "/practice/AMOS22/AMOS22/data/task1_dataset.json"
#     with open(json_path, "r") as file:
#         json_dict = json.load(file)
#         json_dict_test = json_dict[split]
#     size_list = []
#     affine_list = []
#
#     os.chdir('/practice/AMOS22/AMOS22/data')
#     #print("********************zshe len(json_dict_test): {} *********************".format(len(json_dict_test)))
#     for i in range(len(json_dict_test)):
#         print('get_size ct_test_{}...'.format(i))
#         image_path = json_dict_test[i]#["image"]
#         #print(os.listxattr(image_path))
#         #name = os.path.split(os.path.splitext(os.path.splitext(image_path)[0])[0])[1]
#         img = nib.load(image_path)
#
#
#
#
#         size = img.get_data().shape[0:2]
#         affine = img.affine
#         size_list.append(size)
#         affine_list.append(affine)
#
#
#
#         #print('original image size:{}'.format(itk_image.GetSize()[0:2]))
#     os.chdir('/practice/AMOS22/AMOS22')
#     return size_list, affine_list
def get_size(split="test"):
    json_path = "/practice/AMOS22/AMOS22/data/task1_dataset.json"
    with open(json_path, "r") as file:
        json_dict = json.load(file)
        json_dict_test = json_dict[split]

    origin_list = []
    direction_list = []
    spacing_list = []
    size_list = []
    m_list = []
    d_list = []

    os.chdir('/practice/AMOS22/AMOS22/data')
    # print("********************zshe len(json_dict_test): {} *********************".format(len(json_dict_test)))
    for i in range(len(json_dict_test)):
        print('get_size ct_test_{}...'.format(i))
        image_path = json_dict_test[i]  # ["image"]
        # print(os.listxattr(image_path))
        # name = os.path.split(os.path.splitext(os.path.splitext(image_path)[0])[0])[1]
        image = sitk.ReadImage(image_path)


        origin = image.GetOrigin()
        direction = image.GetDirection()
        spacing = image.GetSpacing()
        size = image.GetSize()[::-1][1:3]
        origin_list.append(origin)
        direction_list.append(direction)
        spacing_list.append(spacing)
        size_list.append(size)
        # print(image)
        # print(origin)
        # print(direction)
        # print(spacing)
        #print(size)


        m_l = []
        d_l = []
        for m in image.GetMetaDataKeys():
            m_l.append(m)
            d_l.append(image.GetMetaData(m))
            # print("\"{0}\":\"{1}\"".format(m, image.GetMetaData(m)))
        # print("size:", size)
        # print("\n")
        m_list.append(m_l)
        d_list.append(d_l)


        # print('original image size:{}'.format(itk_image.GetSize()[0:2]))
    os.chdir('/practice/AMOS22/AMOS22')
    #print(m_list, d_list, size_list)
    return origin_list, direction_list, spacing_list, size_list, m_list, d_list

# def batch_resize_image(batch_images, new_shape, order=3):
#     results = []
#     for i in batch_images:
#         result = resize(i, new_shape, order=order, mode='edge')
#         results.append(result[None])
#     image = np.vstack(results).transpose()#.argmax(0)
#     #print(image.shape)
#     return image

def convert_to_one_hot(seg):
    #print("********************zshe seg.shape: {} *********************".format(seg.shape))
    vals = np.unique(seg)

    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

# def batch_resize_image(batch_images, new_shape, order=3):
#     tmp = convert_to_one_hot(batch_images)
#     vals = np.unique(batch_images)
#     # print("********************zshe vals.shape: {} *********************".format(vals.shape))
#     results = []
#     for i in range(len(tmp)):
#         results.append(resize(tmp[i].astype(float), new_shape, order=order, mode='edge')[None])
#     image = vals[np.vstack(results).argmax(0)]
#     # print(np.vstack(results).shape)
#     # print(np.vstack(results).argmax(0).shape)
#     # print("********************zshe seg.shape: {} *********************".format(image.shape))
#
#     return image

def batch_resize_images(batch_images, new_shape, order=3):
    new_shape = (16,)+new_shape
    #print("******************************new_shape:********************:", new_shape)
    results = []
    for i in range(len(batch_images)):
        resize_image = resize(batch_images[i].astype(float), new_shape, order=order, mode='edge')
        #print("******************************resize_image:********************:", resize_image.shape)
        resize_image = resize_image.argmax(0)
        #print("******************************resize_image:********************:", resize_image.shape)
        results.append(resize_image[None])
    image = np.vstack(results)#.transpose((1,0,2,3))
    # print(np.vstack(results).shape)
    # print(np.vstack(results).argmax(0).shape)
    # print("********************zshe seg.shape: {} *********************".format(image.shape))

    return image

# def run(fold, new_shape, affine, key, writer, args):
def run(fold, origin, direction, spacing, size, m_l, d_l, key, model, args):
    maybe_mkdir_k(args.save_path, str(fold))
    logger = PytorchExperimentLogger(os.path.join(args.save_path, str(fold)), "elog", ShowTerminal=True)
    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device:{args.device}")
    torch.manual_seed(args.seed)
    if 'cuda' in str(args.device):
        torch.cuda.manual_seed_all(args.seed)
    logger.print(f"starting testing for fold {fold} ...")


    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.print(f"number of parameters: {num_parameters}")


    test_dataset = MMWHS(key=key, purpose='test', args=args)
    logger.print('testing data dir ' + test_dataset.data_dir)


    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_works, drop_last=False)
    start_time = time.time()
    model.eval()
    x_out_list = []
    with torch.no_grad():
        for batch_idx, tup in enumerate(test_loader):
            img = tup
            #print("img.shape: {}".format(img.shape))
            image_var = img.float().to(args.device)
            x_out = model(image_var)
            #print("x_out.shape###############channel: {}".format(x_out.shape))
            x_out = F.softmax(x_out, dim=1).detach().cpu()
            print("x_out.shape###############channel: {}".format(x_out.shape))
            x_out_list.extend(x_out)
            logger.print(
                f"Test: batch:{batch_idx + 1}/{len(test_loader)}")
    batch_images = np.stack(x_out_list)
    print("stack", batch_images.shape)

    image = batch_resize_images(batch_images, size, order=1)
    print(image.shape)


    time_took = time.time() - start_time
    print("\nTest completed, time took: {}.\n".format(hms_string(time_took)))

    maybe_mkdir_k('test_result', str(fold))
    # vertices, faces, _, _ = marching_cubes_lewiner(np.array(image))
    # mm = dataToMesh(vertices, faces)
    # mm.save(os.path.join('test_result', str(fold), str(key) + ".stl"))

    image = image.astype(np.uint8)
    image = sitk.GetImageFromArray(image)
    print("old size:", image.GetSize())
    # print(size)
    # print(direction)
    # print(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    image.SetSpacing(spacing)


    for m, d in zip(m_l, d_l):
        image.SetMetaData(m, d)
    # image.SetMetaData("bitpix", "16")
    # image.SetMetaData("datatype", "4")
    # image.SetMetaData("scl_inter", "-1024")
    # image.SetMetaData("xyzt_units", "10")

    # for m in image.GetMetaDataKeys():
    #     print("\"{0}\":\"{1}\"".format(m, image.GetMetaData(m)))


    sitk.WriteImage(image, os.path.join('test_result', str(fold), str(key)+".nii.gz"))

    i = sitk.ReadImage(os.path.join('test_result', str(fold), str(key)+".nii.gz"))
    print("new size:", i.GetSize())
    print("new origin:", i.GetOrigin())
    print("new direction:", i.GetDirection())
    print("new spacing:", i.GetSpacing())
    # for m in i.GetMetaDataKeys():
    #     print("\"{0}\":\"{1}\"".format(m, i.GetMetaData(m)))


if __name__ == '__main__':
    # initialize config
    args = get_config()
    args.restart = True
    args.batch_size = 64
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.results_dir = 'test_result'
    args.pretrained_model_path = "/practice/AMOS22/AMOS22/results/contrast_mmwhs_simclr_2022-07-07_14-27-07/0.20cross_val_0/model/latest.pth"
    args.save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    args.classes = 16
    maybe_mkdir_k(args.results_dir, args.experiment_name + args.save)

    writer = SummaryWriter(os.path.join(args.runs_dir, args.experiment_name + args.save))

    model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes,
                   do_instancenorm=True)
    if args.restart:
        print('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]

        model.load_state_dict(save_model)
    model.to(args.device)

    origin_list, direction_list, spacing_list, size_list, m_list, d_list = get_size("test")
    keys = os.listdir(os.path.join(args.data_dir, 'test'))
    for i in range(0, 1):
        for origin, direction, spacing, size, m_l, d_l,  key in zip(origin_list, direction_list, spacing_list, size_list, m_list, d_list,  keys):
            run(i, origin, direction, spacing, size, m_l, d_l, key, model, args)
















