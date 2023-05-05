import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, train_loader_aug_1, train_loader_aug_2, optimizer_aug, scheduler_aug):

    log_period = args.log_period
    eval_period = args.eval_period
    device = f"cuda:{args.GPU_ID}"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        num_train = len(train_loader)
        num_train_aug_1 = len(train_loader_aug_1)
        num_train_aug_2 = len(train_loader_aug_2)

        # print(len(train_loader), len(train_loader_aug))

        train_iter = iter(train_loader)
        train_iter_aug_1 = iter(train_loader_aug_1)
        train_iter_aug_2 = iter(train_loader_aug_2)

        pred = 0

        cur_train = 0
        cur_train_aug_1 = 0
        cur_train_aug_2 = 0

        # while cur_train_aug < num_train_aug:
        # while cur_train < num_train and cur_train_aug < num_train_aug:
        while cur_train < num_train and cur_train_aug_1 < num_train_aug_1 and cur_train_aug_2 < num_train_aug_2:
            if pred < 2:
                batch = next(train_iter)
                pred += 1
                cur_train += 1
                cur_opt = optimizer
            elif pred >= 2 and pred < 3:
                batch = next(train_iter_aug_1)
                pred += 1
                cur_train_aug_1 += 1
                cur_opt = optimizer_aug
            else:
                batch = next(train_iter_aug_2)
                pred = 0
                cur_train_aug_2 += 1
                cur_opt = optimizer_aug
            
            n_iter = cur_train+cur_train_aug_1+cur_train_aug_2-1
                
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            cur_opt.zero_grad()
            total_loss.backward()
            cur_opt.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                # info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader_aug)}]"
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)+len(train_loader_aug_1)+len(train_loader_aug_2)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                info_str += f", Base Lr Aug: {scheduler_aug.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('lr_aug', scheduler_aug.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        scheduler_aug.step()

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        # (train_loader_aug.batch_size) / time_per_batch))
                        (train_loader.batch_size+train_loader_aug_1.batch_size+train_loader_aug_2.batch_size) / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval(), True)
