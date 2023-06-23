import os
from datetime import datetime
from utils import *
import torch.backends.cudnn as cudnn
import time
from torch.autograd import Variable
from loss.contrast_loss import SupConLoss
from network.unet import UNET_classification, get_default_config
#from dataset.chd import CHD
from data.mmwhs import *
#from dataset.acdc import ACDC
from myconfig import get_config
from batchgenerators.utilities.file_and_folder_operations import *
from lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from experiment_log import PytorchExperimentLogger

def maybe_mkdir_b(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(props=None):
    # initialize config
    #args = get_config()
    min_loss = np.inf
    sum_patients = 50
    patient = 0
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    save_path = os.path.join(args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    logger = PytorchExperimentLogger(save_path, "elog", ShowTerminal=True)
    model_result_dir = join(save_path, 'model')
    maybe_mkdir_b(model_result_dir)
    args.model_result_dir = model_result_dir

    logger.print(f"saving to {save_path}")
    writer = SummaryWriter('runs/' + args.experiment_name + args.save)

    # setup cuda
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # logger.print(f"the model will run on device {args.device}")

    # create model
    logger.print("creating model ...")
    model = UNET_classification(input_feature=1,  base_feature=args.initial_filter_size, vector_feature=args.classes, props=props)

    if args.restart:
        logger.print('loading from saved model'+args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model.load_state_dict(save_model)

    model.to(args.device)
    model = torch.nn.DataParallel(model)

    num_parameters = sum([l.nelement() for l in model.module.parameters()])
    logger.print(f"number of parameters: {num_parameters}")
    #
    if args.dataset == 'chd':
        training_keys = os.listdir(os.path.join(args.data_dir,'train'))
        training_keys.sort()
        train_dataset = CHD(keys=training_keys, purpose='train', args=args)
    elif args.dataset == 'mmwhs':
        train_dataset = MMWHS(keys=list(range(0,160)), purpose='train', args=args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_works, drop_last=False)

    # define loss function (criterion) and optimizer
    criterion = SupConLoss(threshold=args.slice_threshold, temperature=args.temp, contrastive_method=args.contrastive_method).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader))

    for epoch in range(args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, epoch, optimizer, scheduler, logger, args)

        logger.print('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \n'
                     .format(epoch+1, train_loss=train_loss))

        writer.add_scalar('training_loss'+str(args.slice_threshold), train_loss, epoch+1)
        writer.add_scalar('lr'+str(args.slice_threshold), optimizer.param_groups[0]['lr'], epoch+1)

        # save model
        if train_loss <= min_loss:
            min_loss = train_loss
            save_dict = {"net": model.module.state_dict()}
            torch.save(save_dict, os.path.join(args.model_result_dir, str(args.slice_threshold)+"latest.pth"))
            patient = 0
        else:
            patient += 1
        if patient >= sum_patients:
            print("\nTrain loss didn't improve last {} epochs.\n".format(sum_patients))
            break

def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args):
    model.train()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        scheduler(optimizer, batch_idx, epoch)
        img1, img2, slice_position, partition = tup
        # print("img1.shape:", img1.shape)
        # print("img2.shape:", img2.shape)
        # print("slice_position.shape:", slice_position.shape)
        # print("partition.shape:", partition.shape)
        image1_var = Variable(img1.float(), requires_grad=False).to(args.device)
        image2_var = Variable(img2.float(), requires_grad=False).to(args.device)
        f1_1 = model(image1_var)
        print(f1_1.shape)
        f2_1 = model(image2_var)
        bsz = img1.shape[0]
        features = torch.cat([f1_1.unsqueeze(1), f2_1.unsqueeze(1)], dim=1)
        if args.contrastive_method == 'pcl':
            loss = criterion(features, labels=slice_position)
        elif args.contrastive_method == 'gcl':
            loss = criterion(features, labels=partition)
        else: # simclr
            loss = criterion(features)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.print(f"epoch:{epoch+1}, batch:{batch_idx+1}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}")
    return losses.avg

if __name__ == '__main__':
    # os.chdir('/practice/AMOS22/AMOS22')
    # print("Current working directory: {}".format(os.getcwd()))

    # print("simclr")
    # props = get_default_config(dim=2, droprate=0.1, nonlin="LeakyReLU", norm_type="bn")#3
    # args = get_config()
    # args.contrastive_method = 'simclr'
    # args.classes = 64
    # main(props)

    print("gcl")
    props = get_default_config(dim=2, droprate=0.1, nonlin="LeakyReLU", norm_type="bn")  # 3
    args = get_config()
    args.contrastive_method = 'gcl'
    args.classes = 64
    main(props)

    # print("pcl_2")
    # props = get_default_config(dim=2, droprate=0.1, nonlin="LeakyReLU", norm_type="bn")  # 3
    # args = get_config()
    # args.contrastive_method = 'pcl'
    # args.classes = 64
    # for i in np.arange(0.20, 0.40, 0.05):
    #     args.slice_threshold = i
    #     main(props)
    #

