import os
import random
import sacred
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from visdom_logger import VisdomLogger
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from src.utils import warp_tqdm, save_checkpoint
from src.trainer import Trainer, trainer_ingredient
from src.eval import Evaluator
from src.eval import eval_ingredient
from src.tim import tim_ingredient
from src.optim import optim_ingredient, get_optimizer, get_scheduler
from src.datasets.ingredient import dataset_ingredient
from src.models.ingredient import get_model, model_ingredient

ex = sacred.Experiment('FSL training',
                       ingredients=[trainer_ingredient, eval_ingredient,
                                    optim_ingredient, dataset_ingredient,
                                    model_ingredient, tim_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    ckpt_path = os.path.join('checkpoints')
    seed = 2020
    pretrain = False
    resume = False
    evaluate = False
    make_plot = False
    epochs = 90
    disable_tqdm = False
    visdom_port = None
    print_runtime = False
    cuda = True


@ex.automain
def main(seed, pretrain, resume, evaluate, print_runtime,
         epochs, disable_tqdm, visdom_port, ckpt_path,
         make_plot, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    callback = None if visdom_port is None else VisdomLogger(port=visdom_port)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
    torch.cuda.set_device(0)
    # create model
    print("=> Creating model '{}'".format(ex.current_run.config['model']['arch']))
    model = torch.nn.DataParallel(get_model()).cuda()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = get_optimizer(model)

    if pretrain:
        pretrain = os.path.join(pretrain, 'checkpoint.pth.tar')
        if os.path.isfile(pretrain):
            print("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            print('[Warning]: Did not find pretrained model {}'.format(pretrain))

    if resume:
        resume_path = ckpt_path + '/checkpoint.pth.tar'
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
        else:
            print('[Warning]: Did not find checkpoint {}'.format(resume_path))
    else:
        start_epoch = 0
        best_prec1 = -1

    cudnn.benchmark = True

    # Data loading code
    evaluator = Evaluator(device=device, ex=ex)
    if evaluate:
        results = evaluator.run_full_evaluation(model=model,
                                                model_path=ckpt_path,
                                                callback=callback)
        return results

    # If this line is reached, then training the model
    trainer = Trainer(device=device, ex=ex)
    scheduler = get_scheduler(optimizer=optimizer,
                              num_batches=len(trainer.train_loader),
                              epochs=epochs)
    tqdm_loop = warp_tqdm(list(range(start_epoch, epochs)),
                          disable_tqdm=disable_tqdm)
    for epoch in tqdm_loop:
        # Do one epoch
        trainer.do_epoch(model=model, optimizer=optimizer, epoch=epoch,
                         scheduler=scheduler, disable_tqdm=disable_tqdm,
                         callback=callback)

        # Evaluation on validation set
        if (epoch) % trainer.meta_val_interval == 0:
            prec1 = trainer.meta_val(model=model, disable_tqdm=disable_tqdm,
                                     epoch=epoch, callback=callback)
            print('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # Save checkpoint
        save_checkpoint(state={'epoch': epoch + 1,
                               'arch': ex.current_run.config['model']['arch'],
                               'state_dict': model.state_dict(),
                               'best_prec1': best_prec1,
                               'optimizer': optimizer.state_dict()},
                        is_best=is_best,
                        folder=ckpt_path)
        if scheduler is not None:
            scheduler.step()

    # Final evaluation on test set
    results = evaluator.run_full_evaluation(model=model, model_path=ckpt_path)
    return results
