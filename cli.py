import timm
import torch

def add_all_parsers(parser):
    _add_loss_parser(parser)
    _add_training_parser(parser)
    _add_model_parser(parser)
    _add_hardware_parser(parser)
    _add_misc_parser(parser)


def _add_loss_parser(parser):
    group_loss = parser.add_argument_group('Loss parameters')
    group_loss.add_argument('--mu', type=float, default=0.0001, help='weight decay parameter')


def _add_training_parser(parser):
    group_training = parser.add_argument_group('Training parameters')
    group_training.add_argument('--lr', type=float, help='learning rate to use')
    group_training.add_argument('--batch_size', type=int, default=32, help='default is 32')
    group_training.add_argument('--n_epochs', type=int)
    group_training.add_argument('--pretrained', action='store_true')
    group_training.add_argument('--image_size', type=int, default=256)
    group_training.add_argument('--crop_size', type=int, default=224)
    group_training.add_argument('--epoch_decay', nargs='+', type=int, default=[])
    group_training.add_argument('--k', nargs='+', help='value of k for computing the topk loss and computing topk accuracy',
                                required=True, type=int)


def _add_model_parser(parser):
    group_model = parser.add_argument_group('Model parameters')
    group_model.add_argument('--model', choices=timm.list_models(),
                             default='resnet50', help='choose the model you want to train on')
    group_model.add_argument('--model_file', type=str,
                             help='choose location of file which contains model weights')


def _add_hardware_parser(parser):
    group_hardware = parser.add_argument_group('Hardware parameters')
    group_hardware.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available())


def _add_misc_parser(parser):
    group_misc = parser.add_argument_group('Miscellaneous parameters')
    group_misc.add_argument('--seed', type=int, help='set the seed for reproductible experiments')
    group_misc.add_argument('--num_workers', type=int, default=4,
                            help='number of workers for the data loader. Default is one. You can bring it up. '
                                 'If you have memory errors go back to one')
    group_misc.add_argument('--root', help='location of the train val and test directories')
    group_misc.add_argument('--save_name_xp', help='name of the saving file')

