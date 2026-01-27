from opt import parse_args
from utils import seed_all
import os
import wandb

from experiment import Experiment
from utils import save_checkpoint

def main():
    log_wandb = True
    args = parse_args()
    project_run_path=os.path.dirname(__file__)

    # security to not waste run
    if args.put_lambda_max and args.minPruning ==0.0:
        print("if put_lambda_max==TRUE, args.minPruning must be >0.0, because no mask are generated for 0.0percent of pruning")
        return

    # if not args.mask and args.pruningType!="adapter":
    #     print("Mask is 0 or False, not adapter run — exiting program.")
    #     return
    
    if args.seed is not None:
        seed_all(args.seed)
        args.save_dir = f'{args.save_dir}_seed_{args.seed}'

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name=f'{args.nameRun}',
            config=vars(args)
        )

    exp = Experiment(args,project_run_path)

    # RD before training
    if args.mask :
        exp.make_baseline_plot()

    for epoch in range(exp.ctx.last_epoch, args.epochs):
        
        #train 
        exp.train(epoch)

        # test 
        # exp.validate(epoch,exp.ctx.val_dataloader,'val')

        # kodak
        # exp.validate(epoch,exp.ctx.kodak_dataloader,'kodak')

        if epoch%5==0:
        #     exp.make_plot(epoch)

    # save model for the last epoch in order to use later
            save_checkpoint(
                                {
                                    "epoch": epoch,
                                    "state_dict": exp.ctx.net.state_dict(),
                                    "best_val_loss": exp.ctx.best_val_loss,
                                    "best_kodak_loss":exp.ctx.best_kodak_loss,
                                    "optimizer": exp.ctx.optimizer.state_dict(),
                                    "aux_optimizer": exp.ctx.aux_optimizer.state_dict() if exp.ctx.aux_optimizer is not None else None,
                                    "lr_scheduler": exp.ctx.lr_scheduler.state_dict(),
                                },
                                True, # is best
                                out_dir=exp.ctx.model_dir,
                                #filename=f"{str(args.lmbda).replace('0.','')}_checkpoint.pth.tar"
                                filename=f"{exp.args.nameRun}_checkpoint_{epoch}.pth.tar"
                            )



    if log_wandb:
        wandb.run.finish()
        
if __name__ == "__main__":
    main()